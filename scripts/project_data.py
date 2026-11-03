"""Shared paths and serialization for the single-date study."""
from pathlib import Path
import hashlib,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ORIGIN=np.array([825800.,842100.,0.])
def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
