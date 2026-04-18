from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from src.utils.logger import log_performance

app = FastAPI(title="Credit-Shield MLOps API")

class AnalysisInput(BaseModel):
    text: str

model_path = "./model_artifacts"

tokenizer = None
model = None

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval() 
except Exception as e:
    print(f"Aviso: Modelo no cargado localmente ({e}). Ignorar si estás en CI/CD.")

@app.get("/")
def read_root():
    return {"status": "API is running", "engine": "PyTorch Manual Inference"}

@app.post("/predict")
@log_performance
async def predict(input_data: AnalysisInput):
    # Verificación de seguridad
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible.")
    
    try:
        # Tokenización
        inputs = tokenizer(
            input_data.text, 
            return_tensors="pt", 
            return_token_type_ids=False  
        )
        
        # Inferencia
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Procesamiento de resultados
        probs = F.softmax(logits, dim=-1)
        prediction_idx = torch.argmax(probs).item()
        
        label = "POSITIVE" if prediction_idx == 1 else "NEGATIVE"
        score = probs[0][prediction_idx].item()
        
        return {
            "input": input_data.text,
            "prediction": [{"label": label, "score": round(score, 4)}]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {str(e)}")