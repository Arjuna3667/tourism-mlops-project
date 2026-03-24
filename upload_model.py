from huggingface_hub import HfApi
import os

api = HfApi()

api.upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id="Arjuna3667/tourism-model",
    repo_type="model",
    token=os.environ["HF_TOKEN"]
)

print("Model uploaded successfully!")
