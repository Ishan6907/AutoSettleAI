from huggingface_hub import HfApi

api = HfApi()

for repo in ["ParthG09/Car_Accidents_and_deformation_dataset", "TornikeO/CarAccidents", "M-ArslanArshad/Car_Accidents_and_deformation_dataset"]:
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
        print(f"{repo}: {len(files)} files. Sample: {files[:5]}")
    except Exception as e:
        print(f"{repo}: {e}")
