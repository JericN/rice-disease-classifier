from huggingface_hub import HfApi, delete_repo

# 🔐 Replace with your Hugging Face token
HF_TOKEN = ""

# Initialize API
api = HfApi()
user_info = api.whoami(token=HF_TOKEN)
username = user_info['name']

# Get all your model repos
model_repos = api.list_models(author=username, token=HF_TOKEN)

# Loop through and delete repos with "v3" or "test" in the ID
for model in model_repos:
    model_id = model.modelId
    if "v3" in model_id.lower() or "test" in model_id.lower():
        print(f"Deleting model: {model_id}")
        
        # Delete the model
        delete_repo(
            token=HF_TOKEN,
            repo_id=model_id,
            repo_type="model"
        )
    # else:
    #     print(f"Skipping model: {model_id}")
