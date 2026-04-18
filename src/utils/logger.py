import logging
import time
import json
import sys
from functools import wraps

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app_performance.log")
    ]
)

logger = logging.getLogger("Credit-Shield")

def log_performance(func):
    """
    Decorator to log the performance (latency and results) of API endpoints.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            latency = time.perf_counter() - start_time
            
            # Log structured data
            log_data = {
                "function": func.__name__,
                "latency_seconds": round(latency, 4),
                "status": "success",
                "result_preview": str(result)[:100] # Evitar logs masivos
            }
            logger.info(f"Performance Metrics: {json.dumps(log_data)}")
            
            return result
        except Exception as e:
            latency = time.perf_counter() - start_time
            log_data = {
                "function": func.__name__,
                "latency_seconds": round(latency, 4),
                "status": "error",
                "error_message": str(e)
            }
            logger.error(f"Performance Error: {json.dumps(log_data)}")
            raise e
    return wrapper

def get_logger():
    return logger
