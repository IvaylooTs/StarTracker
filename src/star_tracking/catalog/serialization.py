import pickle

def load_catalog_hash(file_path):
    with open(file_path, "rb") as f:
        catalog_hash = pickle.load(f)
    
    return catalog_hash