from pathlib import Path
import json, os
from dataloader import build_manifest  

jsonl = Path("data/processed/index_val.jsonl")
cache_root = "data/cache_npy"
manifest_out = "manifests/sr16000/val_pairs.json"

paths = [json.loads(l)["path"] for l in jsonl.read_text().splitlines() if l.strip()]
common_root = os.path.commonpath([str(Path(p).parent) for p in paths])

build_manifest(common_root, cache_root, manifest_out)
print("Manifest scritto in:", manifest_out)
