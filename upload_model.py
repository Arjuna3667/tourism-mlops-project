from huggingface_hub import upload_file
import os

upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id="Arjuna3667/tourism-model",
    repo_type="model",
    token=os.environ["HF_TOKEN"]
)
