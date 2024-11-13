import hashlib
import os
from pathlib import Path
import requests
import torch
from tqdm import tqdm

# ********************************************************************
# *************** Utilities to download pretrained LDM ***************
# ********************************************************************

URL_MAP_LDM = {
    "ldm": "<URL_OF_THE_PRETRAINED_LDM_MODEL>"
}

CKPT_MAP_LDM = {
    "ldm": "ldm.pth"
}

MD5_MAP_LDM = {
    "ldm": "<MD5_CHECKSUM_OF_THE_LDM_MODEL>"
}

def download(url: str, local_path: str, chunk_size: int = 1024) -> None:
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

def md5_hash(path: str) -> str:
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()

def get_ldm_ckpt_path(name: str, root: str, check: bool = False) -> str:
    assert name in URL_MAP_LDM
    path = os.path.join(root, CKPT_MAP_LDM[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP_LDM[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP_LDM[name], path))
        download(URL_MAP_LDM[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP_LDM[name], md5
    return path

def load_ldm_model(ckpt_path: str) -> torch.nn.Module:
    # This is a placeholder for loading the actual LDM model
    # You should replace it with the appropriate model architecture and loading logic
    ldm_model = torch.load(ckpt_path, map_location=torch.device("cpu"))
    return ldm_model

if __name__ == "__main__":
    root = Path.home() / ".cache/iris/ldm_pretrained"
    ckpt_path = get_ldm_ckpt_path("ldm", root=str(root))
    ldm_model = load_ldm_model(ckpt_path)
    print("LDM model loaded successfully!")
