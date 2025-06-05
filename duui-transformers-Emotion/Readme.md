[![Version](https://img.shields.io/static/v1?label=duui-transformers-emotion&message=0.3.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-transformers-emotion-finetuned-twitter-xlm-roberta-base-emotion/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.10&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.41.2&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.5.1&color=red)]()

# Transformers Emotion

DUUI implementation for selected Hugging-Face-based transformer [Emotion tools](https://huggingface.co/models?sort=trending&search=emotion) models,  [pol_emo_mDeBERTa](https://github.com/tweedmann/pol_emo_mDeBERTa2), [pysentimiento](https://github.com/pysentimiento/pysentimiento/) and [EmoAtlas](https://github.com/alfonsosemeraro/emoatlas).
## Included Models

| Name                           | link                                                                        | Revision                                | Languages      |
|--------------------------------|-----------------------------------------------------------------------------|-----------------------------------------|----------------|
| finetuned-twitter-xlm-roberta-base-emotion | https://huggingface.co/02shanky/finetuned-twitter-xlm-roberta-base-emotion  | 28e6d080e9f73171b574dd88ac768da9e6622c36 | Multilingual   |
| dreamy-xlm-roberta-emotion     | https://huggingface.co/DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence | b3487623ec2dd4b9bd0644d8266291afb9956e9f | Multilingual   |
| pol_emo_mDeBERTa               | https://github.com/tweedmann/pol_emo_mDeBERTa2                              | 523da7dc2523631787ef0712bad53bfe2ac46840 | Multilingual   |
| xlm-emo-t                      | https://huggingface.co/MilaNLProc/xlm-emo-t                                 | a6ee7c9fad08d60204e7ae437d41d392381496f0 | Multilingual   |
| emotion-english-distilroberta-base | https://huggingface.co/j-hartmann/emotion-english-distilroberta-base        | 0e1cd914e3d46199ed785853e12b57304e04178b | EN             |
| emotion_text_classifier        | https://huggingface.co/michellejieli/emotion_text_classifier                | dc4df5597fcda82589511c3900fedbe1c0ffec82 | EN             |
| cardiffnlp-twitter-roberta-base-emotion | https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion              | 2848306ad936b7cd47c76c2c4e14d694a41e0f54 | EN             |
| bertweet-base-emotion-analysis | https://huggingface.co/finiteautomata/bertweet-base-emotion-analysis        | c482c9e1750a29dcc393234816bcf468ff77cd2d | EN             |
| distilbert-base-uncased-finetuned-emotion | https://huggingface.co/ActivationAI/distilbert-base-uncased-finetuned-emotion | dbf4470880ff3b73f22975241cd309bdf8e2195f | EN             |
| roberta-base-go_emotions       | https://huggingface.co/SamLowe/roberta-base-go_emotions                     | 58b6c5b44a7a12093f782442969019c7e2982299 | EN             |
| t5-base-finetuned-emotion      | https://huggingface.co/mrm8488/t5-base-finetuned-emotion                    | e44a316825f11230724b36412fbf1899c76e82de | EN             |
| emoatlas                       | https://github.com/alfonsosemeraro/emoatlas                                 | adae44a80dd55c1d1c467c4e72bdb2d8cf63bf28 | EN             |
| pysentimiento                  | https://github.com/pysentimiento/pysentimiento/                             | 60822acfd805ad5d95437c695daa33c18dbda060 | EN, ES, IT, PT |
| exalt-baseline                 | https://huggingface.co/pranaydeeps/EXALT-Baseline                           | 4b5e2a38b4e72823c428891170aec8930f580bad | Multi          |
| feel-it                        | https://huggingface.co/MilaNLProc/feel-it-italian-emotion                                   | 6efdabf62230414aeba764986b4ae317ce7c5c47         | IT             |
| cardiffnlp-multilabel                   | https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion-multilabel-latest                                 | 30a56d88e47e493f08f93c786d49c526550b55b9        | EN             |
| bert-emotion                   | https://huggingface.co/boltuix/bert-emotion                                 | 00b4ef11958dd607b2ede29f6ed6d02338782c94        | EN             |
| beto-es-analysis                               | https://huggingface.co/finiteautomata/beto-emotion-analysis           | 9b628b0bd91471ad9bd709c10522c379ce09c32a        | ES             |



# How To Use

For using duui-transformers-emotion as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run -p 9714:9714 docker.texttechnologylab.org/duui-transformers-emotion-[modelname]:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-transformers-emotion/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-transformers-emotion-[modelname]:latest")
        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
);
```

### Parameters

| Name | Description |
| ---- | ----------- |
| `model_name` | Model to use, see table above |
| `selection`  | Use `text` to process the full document text or any selectable UIMA type class name |

## Building
Before build download [pol_emo_mDeBERTa2.zip](https://github.com/tweedmann/pol_emo_mDeBERTa2/releases/download/v.1.0.0/pol_emo_mDeBERTa2.zip) and save the folder pol_emo_DeBERTa under the python directory.
Also download the nltk wordnet with nltk.download('wordnet', download_dir="nltk_data") and save it under the python directory.

# Cite

If you want to use the DUUI image please quote this as follows:

Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler. (2023). "Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI." Findings of the Association for Computational Linguistics: EMNLP 2023, 385–399. [[LINK](https://aclanthology.org/2023.findings-emnlp.29)] [[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]

## BibTeX

```
@inproceedings{Leonhardt:et:al:2023,
  title     = {Unlocking the Heterogeneous Landscape of Big Data {NLP} with {DUUI}},
  author    = {Leonhardt, Alexander and Abrami, Giuseppe and Baumartz, Daniel and Mehler, Alexander},
  editor    = {Bouamor, Houda and Pino, Juan and Bali, Kalika},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
  year      = {2023},
  address   = {Singapore},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2023.findings-emnlp.29},
  pages     = {385--399},
  pdf       = {https://aclanthology.org/2023.findings-emnlp.29.pdf},
  abstract  = {Automatic analysis of large corpora is a complex task, especially
               in terms of time efficiency. This complexity is increased by the
               fact that flexible, extensible text analysis requires the continuous
               integration of ever new tools. Since there are no adequate frameworks
               for these purposes in the field of NLP, and especially in the
               context of UIMA, that are not outdated or unusable for security
               reasons, we present a new approach to address the latter task:
               Docker Unified UIMA Interface (DUUI), a scalable, flexible, lightweight,
               and feature-rich framework for automatic distributed analysis
               of text corpora that leverages Big Data experience and virtualization
               with Docker. We evaluate DUUI{'}s communication approach against
               a state-of-the-art approach and demonstrate its outstanding behavior
               in terms of time efficiency, enabling the analysis of big text
               data.}
}

@misc{Bagci:2024,
  author         = {Bagci, Mevlüt},
  title          = {Emotion models as {DUUI} component},
  year           = {2024},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-transformers-Emotion}
}

```
