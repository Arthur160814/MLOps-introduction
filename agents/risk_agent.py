import os
import requests
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

class CreditShieldAgent:
    def __init__(self, api_url="http://localhost:8000/predict"):
        """
        Inicializa el agente de riesgo.
        :param api_url: URL de la API local de sentimiento.
        """
        self.api_url = api_url
        self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            print("ADVERTENCIA: GROQ_API_KEY no encontrada en el entorno.")
            self.llm = None
        else:
            try:
                self.llm = ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.2,
                    groq_api_key=self.api_key
                )
            except Exception as e:
                print(f"ERROR: No se pudo conectar con Groq: {e}")
                self.llm = None

    def analyze_customer_sentiment(self, text):
        """
        Ejecuta el pipeline de análisis: API Local -> Groq Reasoning.
        """
        # 1. Llamada a la API de sentimiento local
        try:
            response = requests.post(self.api_url, json={"text": text}, timeout=5)
            response.raise_for_status()
            sentiment_data = response.json()
            prediction = sentiment_data["prediction"][0]
        except requests.exceptions.RequestException as e:
            return f"ERROR CRÍTICO: No se pudo conectar con el servicio de análisis técnico (FastAPI/Docker): {e}"
        except (KeyError, IndexError):
            return "ERROR: Formato de respuesta inesperado de la API local."

        # 2. Preparación del razonamiento con Groq
        if not self.llm:
            return (
                f"--- ANÁLISIS TÉCNICO (Sin LLM) ---\n"
                f"Sentimiento: {prediction['label']} (Confianza: {prediction['score']:.2%})\n"
                f"Nota: Configura GROQ_API_KEY para obtener el análisis de riesgo profesional."
            )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "Actúa como un Analista de Riesgo Senior de un Banco con 20 años de experiencia. "
                "Tu tarea es interpretar los resultados de sentimiento de los clientes. "
                "Debes explicar el impacto directo en el riesgo crediticio o reputacional del banco "
                "y proponer una acción comercial o de mitigación de riesgo concreta. "
                "Usa un lenguaje formal, corporativo y profesional."
            )),
            HumanMessage(content=(
                f"El sistema ha detectado el siguiente sentimiento para este comentario: '{text}'\n"
                f"Resultado técnico: {prediction['label']} con un {prediction['score']:.2%} de confianza.\n\n"
                "Por favor, genera un reporte ejecutivo breve con:\n"
                "1. Impacto de Riesgo.\n"
                "2. Propuesta de Acción Bancaria."
            ))
        ])

        # 3. Invocación de Groq
        try:
            chain = prompt | self.llm
            report = chain.invoke({})
            return report.content
        except Exception as e:
            return (
                f"ERROR EN RAZONAMIENTO: {e}\n\n"
                f"Datos Técnicos: {prediction['label']} ({prediction['score']:.2%})"
            )

if __name__ == "__main__":
    # Prueba rápida del agente
    agent = CreditShieldAgent()
    test_text = "I am very unhappy with the high interest rates on my loan, I am thinking of moving my funds to another bank."
    
    print("\n" + "="*50)
    print("EJECUCIÓN DEL AGENTE DE RIESGO (GROQ ENGINE)")
    print("="*50)
    
    result = agent.analyze_customer_sentiment(test_text)
    print(result)
    print("="*50 + "\n")
