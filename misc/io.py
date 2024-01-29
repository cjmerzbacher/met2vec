import os
import pickle
import json

def join(folder, subpath, raise_exists=False, make_folder=False):
    joined = os.path.join(folder, subpath)
    if not os.path.exists(joined):
        if raise_exists:
            raise FileNotFoundError(f"'Error {subpath}' not found at '{folder}'!")
        elif make_folder:
            os.makedirs(joined)

    return joined

def safe_pkl_load(path):
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except:
        return None
    
def safe_pkl_dump(path, obj, show=False):
    try:
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        if show:
            print(f"Failed pickle dump, {e}")
    
def safe_json_load(path):
    try:
        with open(path, 'r') as file:
            return json.load(file)
    except:
        return None
        
def safe_json_dump(path, obj, show=False):
    try:
        with open(path, 'w') as file:
            json.dump(obj, file, indent=4)
    except Exception as e:
        if show:
            print(f"Failed pickle dump, {e}")

def save_args(folder, args):
    with open(os.path.join(folder, "args.json"), "w+") as file:
        json.dump(vars(args), file, indent=4)