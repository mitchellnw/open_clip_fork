import fsspec
import torch

def pt_save(fs, pt_obj, file_path):
    of = fs.open(file_path, "wb")
    with of as f:
        torch.save(pt_obj, file_path)

def pt_load(fs, file_path, map_location=None):
    of = fs.open(file_path, "rb")
    with of as f:
        out = torch.load(f)
    return out

