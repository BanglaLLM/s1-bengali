import os
from huggingface_hub import HfApi, login, create_repo
from tqdm import tqdm
import time
import glob

# Function to upload files one by one with a progress bar
def upload_with_progress(api, folder_path, repo_id):
    # Get all files in the directory and subdirectories
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    
    # Calculate total size
    total_size = sum(os.path.getsize(f) for f in all_files)
    print(f"Found {total_size/1024/1024:.2f} MB of files to upload")
    
    # Sort files by size (smaller files first for quick feedback)
    all_files.sort(key=lambda x: os.path.getsize(x))
    
    # Create progress bar
    pbar = tqdm(total=len(all_files), desc="Uploading files")
    
    # Upload each file
    for file_path in all_files:
        # Get the relative path for the HF repo
        relative_path = os.path.relpath(file_path, folder_path)
        file_size = os.path.getsize(file_path)/1024/1024
        
        # Update progress bar description to show current file
        pbar.set_description(f"Uploading {relative_path} ({file_size:.2f} MB)")
        
        # Upload the file
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=relative_path,
            repo_id=repo_id,
            repo_type="model"
        )
        
        # Update progress
        pbar.update(1)
    
    pbar.close()

# Get repository information
repo_id = input("Repo ID (e.g. username/model-name): ")
folder_path = input("Local model folder path: ")

# Validate input
if not os.path.isdir(folder_path):
    print(f"Error: '{folder_path}' is not a valid directory")
    exit(1)

# Create API instance
api = HfApi()

# Create the repository if it doesn't exist
try:
    print(f"Creating repository '{repo_id}'...")
    create_repo(repo_id, repo_type="model", exist_ok=True)
    print("Repository created successfully")
except Exception as e:
    print(f"Note: {e}")

try:
    # Start the upload with progress tracking
    print(f"Starting upload to {repo_id}...")
    upload_with_progress(api, folder_path, repo_id)
    print(f"✅ Upload complete! Model is now available at: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"❌ Error during upload: {e}")
    print("Try updating huggingface_hub with: pip install -U huggingface_hub")