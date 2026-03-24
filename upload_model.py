import os
from huggingface_hub import HfApi

print("FILES BEFORE UPLOAD:", os.listdir())

if not os.path.exists("model.pkl"):
    raise FileNotFoundError(" model.pkl NOT FOUND before upload")

api = HfApi()

api.upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id="Arjuna3667/tourism-model",
    repo_type="model",
    token=os.environ["HF_TOKEN"]
)

print("Model uploaded successfully!")
