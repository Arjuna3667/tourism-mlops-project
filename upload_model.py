from huggingface_hub import HfApi
import os

print("FILES BEFORE UPLOAD:", os.listdir())

api = HfApi()

api.upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id="Arjuna3667/tourism-package-data",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)

print("Model uploaded successfully!")
