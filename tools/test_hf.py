from huggingface_hub import HfApi

api = HfApi()
datasets = list(api.list_datasets(search="car accident"))
print(f"Datasets matching 'car accident': {len(datasets)}")
for d in datasets[:10]:
    print(d.id)

datasets_india = list(api.list_datasets(search="accident india"))
print(f"Datasets matching 'accident india': {len(datasets_india)}")
for d in datasets_india[:10]:
    print(d.id)

datasets_damage = list(api.list_datasets(search="vehicle damage"))
print(f"Datasets matching 'vehicle damage': {len(datasets_damage)}")
for d in datasets_damage[:10]:
    print(d.id)
