import yaml
import logging

def setup_logger():
    logging.basicConfig(filename='/logs/app.log', level=logging.INFO)

def load_config():
    with open("student-ai-ml-tool\config\settings.yaml", 'r') as f:
        return yaml.safe_load(f)
