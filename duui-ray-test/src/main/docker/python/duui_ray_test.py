import ray
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    AutoConfig,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple
import logging
from ray import train
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer
from ray.air import ScalingConfig
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OLMpicsMLMDataset(Dataset):
    """Dataset for MLM training with multiple choice questions."""
    
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        stem = item['stem']
        choices = item['choices']
        answer_key = item['answerKey']
          
        answer_idx = answer_key
            
        # Get the correct answer
        correct_answer = choices[answer_idx]
        
        # Tokenize the stem with [MASK] tokens
        stem_encoding = self.tokenizer(
            stem,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize the correct answer separately
        answer_encoding = self.tokenizer(
            correct_answer,
            add_special_tokens=False,
            return_tensors='pt'
        )
        
        input_ids = stem_encoding['input_ids'].squeeze()
        attention_mask = stem_encoding['attention_mask'].squeeze()
        answer_tokens = answer_encoding['input_ids'].squeeze()
        
        # Create labels: only predict the masked tokens
        labels = input_ids.clone()
        
        # Find positions of [MASK] tokens in the input
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
        
        # Set all positions to -100 (ignore) except mask positions
        labels[:] = -100
        
        # For each mask position, set the corresponding answer token as label
        if len(mask_positions) > 0:
            # Handle cases where there are multiple masks or single mask
            if answer_tokens.dim() == 0:
                answer_tokens = answer_tokens.unsqueeze(0)
            
            # Map answer tokens to mask positions
            for i, mask_pos in enumerate(mask_positions):
                if i < len(answer_tokens):
                    labels[mask_pos] = answer_tokens[i]
                else:
                    # If we have more masks than answer tokens, ignore extra masks
                    break
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'answer_idx': answer_idx
        }

def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'answer_idx': torch.tensor([item['answer_idx'] for item in batch])
    }

def train_func(config):
    """Training function to be executed on each worker."""
    
    # Get distributed training setup
    device = train.torch.get_device()
    
    # Load tokenizer and model
    model_name = config.get('model_name', 'bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    
    # Load dataset
    dataset = load_dataset('KevinZ/oLMpics', 'Age_Comparison')['train']

    # Create train/validation split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataset objects
    train_mlm_dataset = OLMpicsMLMDataset(train_dataset, tokenizer, config['max_length'])
    val_mlm_dataset = OLMpicsMLMDataset(val_dataset, tokenizer, config['max_length'])
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_mlm_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_dataloader = DataLoader(
        val_mlm_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Prepare model for distributed training
    model = train.torch.prepare_model(model)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    num_training_steps = len(train_dataloader) * config['num_epochs']
    num_warmup_steps = int(0.1 * num_training_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        for batch in train_dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k != 'answer_idx'}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
        
        # Validation
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items() if k != 'answer_idx'}
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()
                num_val_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches
        avg_val_loss = total_val_loss / num_val_batches
        
        # Report metrics
        train.report({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        })
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model")
            model.module.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            checkpoint = Checkpoint.from_directory(temp_dir)
            train.report({}, checkpoint=checkpoint)
        
        logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

def main():
    """Main function to start Ray training."""
    
    # Initialize Ray
    ray.init()
    
    # Training configuration
    config = {
        'model_name': 'bert-base-uncased',  # You can change this to other models
        'batch_size': 16,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'num_epochs': 3,
        'max_length': 512,
    }
    
    # Create Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=ScalingConfig(
            num_workers=1,  # Adjust based on your available GPUs
            use_gpu=False,
            resources_per_worker={
                "CPU": 2,
                "GPU": 0
            }
        )
    )
    
    # Start training
    result = trainer.fit()
    
    # Get the best checkpoint
    best_checkpoint = result.checkpoint
    
    print("Training completed!")
    print(f"Best checkpoint: {best_checkpoint}")
    
    # Optionally, load and use the trained model
    # You can load the model from the checkpoint for inference
    
    ray.shutdown()

if __name__ == "__main__":

    main()

# Additional utility functions for evaluation and inference

class MLMEvaluator:
    """Evaluator for the trained MLM model on multiple choice questions."""
    
    def __init__(self, model_path, tokenizer_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.eval()
        
    def predict_masked_choice(self, stem: str, choices: List[str]) -> Tuple[int, float]:
        """
        Predict the best choice for a masked question.
        
        Args:
            stem: Question stem with [MASK] token
            choices: List of possible answers
            
        Returns:
            Tuple of (predicted_index, confidence_score)
        """
        scores = []
        
        for choice in choices:
            # Replace [MASK] with the choice
            text = stem.replace('[MASK]', choice)
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors='pt')
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Calculate perplexity as a score
            loss = outputs.loss
            perplexity = torch.exp(loss)
            scores.append(-perplexity.item())  # Negative because we want higher scores for better fits
        
        # Get the choice with the highest score
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        return best_idx, best_score
    
    def evaluate_dataset(self, dataset):
        """Evaluate the model on a dataset."""
        correct = 0
        total = 0
        
        for item in dataset:
            stem = item['stem']
            choices = item['choices']
            answer_key = item['answerKey']
            
            # Convert answer key to index
            if isinstance(answer_key, str):
                correct_idx = ord(answer_key.upper()) - ord('A')
            else:
                correct_idx = answer_key
                
            # Predict
            predicted_idx, score = self.predict_masked_choice(stem, choices)
            
            if predicted_idx == correct_idx:
                correct += 1
            total += 1
        
        accuracy = correct / total
        return accuracy

# Example usage for evaluation
def evaluate_trained_model(checkpoint_path):
    """Evaluate a trained model checkpoint."""
    
    # Load test dataset
    test_dataset = load_dataset('KevinZ/oLMpics', 'Age_Comparison')["test"]
    
    # Create evaluator
    evaluator = MLMEvaluator(checkpoint_path)
    
    # Evaluate
    accuracy = evaluator.evaluate_dataset(test_dataset)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return accuracy