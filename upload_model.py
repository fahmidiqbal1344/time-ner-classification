from huggingface_hub import login, create_repo, upload_folder

REPO_ID = "mdg-nlp/time-ner-bert-base-cased"  # change to datasets/... if you truly want it in Datasets
MODEL_DIR = "outputs/bert-base-cased-timeNER"

login()  # will prompt once; or login(token="hf_...")

# For MODELS section:
create_repo(repo_id=REPO_ID, exist_ok=True)

# For DATASETS section instead:
# create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

upload_folder(
    repo_id=REPO_ID,
    folder_path=MODEL_DIR,
    path_in_repo=".",
    commit_message="Upload time-ner model",
)
print(" Uploaded!")