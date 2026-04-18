from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from src.utils.logger import log_performance

app = FastAPI(title="Credit-Sentiment MLOps API")

class AnalysisInput(BaseModel):
    text: str

model_path = "./model_artifacts"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval() 
except Exception as e:
    print(f"Error crítico cargando el modelo: {e}")
    model = None

@app.get("/")
def read_root():
    return {"status": "API is running", "engine": "PyTorch Manual Inference"}

@app.post("/predict")
@log_performance
async def predict(input_data: AnalysisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible.")
    
    try:
        inputs = tokenizer(
            input_data.text, 
            return_tensors="pt", 
            return_token_type_ids=False  
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        probs = F.softmax(logits, dim=-1)
        prediction_idx = torch.argmax(probs).item()
        
        label = "POSITIVE" if prediction_idx == 1 else "NEGATIVE"
        score = probs[0][prediction_idx].item()
        
        return {
            "input": input_data.text,
            "prediction": [{"label": label, "score": round(score, 4)}]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia manual: {str(e)}")