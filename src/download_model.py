from transformers import pipeline
import os

def download():
    model_path = "./model_artifacts"
    if not os.path.exists(model_path):
        print("Descargando modelo de Hugging Face...")
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        classifier.save_pretrained(model_path)
        print(f"Modelo guardado en {model_path}")
    else:
        print("El modelo ya existe localmente.")

if __name__ == "__main__":
    download()