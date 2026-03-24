from huggingface_hub import HfApi
import os

print("Files in current directory:", os.listdir())

HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi()

api.upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id="Arjuna3667/tourism-model",
    repo_type="model",
    token=HF_TOKEN
)
