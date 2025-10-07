from pathlib import Path
import json, os
from dataloader import build_manifest  # viene da src/dataloader.py

jsonl = Path("data/processed/index_val.jsonl")  # <-- il tuo file JSONL
cache_root = "data/cache_npy"                        # dove salvare le .npy (va bene)
manifest_out = "manifests/sr16000/val_pairs.json"  # nuovo manifest "entries"

# Trova la root comune dei .npz elencati nell'index
paths = [json.loads(l)["path"] for l in jsonl.read_text().splitlines() if l.strip()]
common_root = os.path.commonpath([str(Path(p).parent) for p in paths])

# Costruisci il manifest: scansiona 'common_root' e scrivi il JSON "entries"
build_manifest(common_root, cache_root, manifest_out)
print("Manifest scritto in:", manifest_out)
