import os
from huggingface_hub import HfApi

model_path = os.path.join(os.getcwd(), "model.pkl")

print("Current working dir:", os.getcwd())
print("Files here:", os.listdir())
print("Looking for:", model_path)

if not os.path.exists(model_path):
    raise FileNotFoundError(f" model.pkl NOT FOUND at {model_path}")

api = HfApi()

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="model.pkl",
    repo_id="Arjuna3667/tourism-model",
    repo_type="model",
    token=os.environ["HF_TOKEN"]
)

print(" Model uploaded successfully!")
