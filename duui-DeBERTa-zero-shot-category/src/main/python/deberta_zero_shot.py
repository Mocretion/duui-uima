from typing import List, Optional

from fastapi import FastAPI, Response
from fastapi.openapi.utils import get_openapi
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from transformers import pipeline

import uvicorn
import threading
import torch 

# A label containing the label and its zero-short score
class Label(BaseModel):
    label: str
    score: float
    iBegin: int
    iEnd: int

class Selection(BaseModel):
    text: str
    iBegin: int
    iEnd: int

# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    doc_text: str
    labels: List[str]
    selection: Optional[List[Selection]]
    multi_label: bool
    clear_gpu_cache_after: int


# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of Sentiment
    labels: List[Label]


# Creates an instance of the pipeline.
# Device = -1 forces cpu usage
device = torch.cuda.current_device() if torch.cuda.is_available() else -1
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=device)

lock = threading.Lock()

def run_classifier(text, labels, multi_label):
    with lock:
        return classifier(text, labels, multi_label=multi_label)

def analyse(doc_text, selection, labels, multi_label, clear_gpu_cache_after):
    analyzed_labels = []
    print("Start Analyse")

    if len(selection) > 0:
         print("Selection is set...")
         for i, s in enumerate(selection):

            result = run_classifier(s.text, labels, multi_label)

            sel_labels = result["labels"]
            sel_scores = result["scores"]

            for r in range(len(sel_labels)):
                analyzed_labels.append(Label(label=sel_labels[r], score=sel_scores[r], iBegin=s.iBegin, iEnd=s.iEnd))

            if(torch.cuda.is_available() and i % clear_gpu_cache_after == 0):
                torch.cuda.empty_cache() 

#             if i % 1000 == 0:
#                 print(f"[DEBUG] {i} / {len(selection)}")
#                 print(sel_labels)
#                 print(sel_scores)
#                 print("===========================================")

    else:
        print("Analyse full text")

        text_length = len(doc_text)

        result = classifier(doc_text, labels, multi_label=True)

        labels = result["labels"]
        scores = result["scores"]

        for i in range(len(labels)):
            analyzed_labels.append(Label(label=labels[i], score=scores[i], iBegin=0, iEnd=text_length))

    return analyzed_labels


# Start fastapi
app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
)

# Get input and output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": [""],
        "outputs": ["org.hucompute.textimager.uima.type.category.CategoryCoveredTagged"]
    }
    return json_item


# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'dkpro-core-types.xml'
with open(typesystem_filename, 'rb') as f:
    typesystem = f.read()
# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(
        content=typesystem,
        media_type="application/xml"
    )


# Load the Lua communication script
communication = "communication.lua"
with open(communication, 'rb') as f:
    communication = f.read().decode("utf-8")

# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return communication


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:
    doc_text = request.doc_text
    labels = request.labels
    selection = request.selection
    multi_label = request.multi_label
    clear_gpu_cache_after = request.clear_gpu_cache_after

    analysed_labels = analyse(doc_text, selection, labels, multi_label, clear_gpu_cache_after)

    # Return data as JSON
    return DUUIResponse(
        labels=analysed_labels
    )


# Documentation for api
openapi_schema = get_openapi(
    title="DeBERTa-Zero-Shot Classification",
    description="A implementation of the MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli Modell for TTLab DUUI",
    version="0.1",
    routes=app.routes
)

# Extra Documentation not supported by FastApi
# https://fastapi.tiangolo.com/how-to/extending-openapi/#self-hosting-javascript-and-css-for-docs
# https://spec.openapis.org/oas/v3.1.0#infoObject
openapi_schema["info"]["contact"] = {"name": "TTLab Team", "url": "https://texttechnologylab.org", "email": "abrami@em.uni-frankfurt.de"}
openapi_schema["info"]["termsOfService"] = "https://www.texttechnologylab.org/legal_notice/"
openapi_schema["info"]["license"] = {"name": "AGPL", "url": "http://www.gnu.org/licenses/agpl-3.0.en.html"}
app.openapi_schema = openapi_schema


# For starting the script locally
if __name__ == "__main__":
    uvicorn.run("deberta_zero_shot:app", host="0.0.0.0", port=9714, workers=1)
