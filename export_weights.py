"""Convert a PyTorch DQN checkpoint (.pt) to a NumPy .npz without importing torch.

Usage: python export_weights.py runs/flappybirdv0.pt runs/flappybirdv0_weights.npz
"""
import zipfile, pickle, numpy as np, io, sys
path = sys.argv[1]
z = zipfile.ZipFile(path)
names = z.namelist()
prefix = names[0].split('/')[0]
class Stor:  # placeholder for torch storage
    def __init__(self, key, dtype): self.key, self.dtype = key, dtype
class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if name == '_rebuild_tensor_v2':
            def rebuild(storage, offset, size, stride, *a):
                raw = z.read(f"{prefix}/data/{storage.key}")
                arr = np.frombuffer(raw, dtype=np.float32)
                n = int(np.prod(size)) if size else 1
                return arr[offset:offset+n].reshape(size).copy()
            return rebuild
        if name == 'OrderedDict':
            import collections; return collections.OrderedDict
        if name.endswith('Storage'):
            return name
        return super().find_class(mod, name)
    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        return Stor(pid[2], pid[1])
sd = U(io.BytesIO(z.read(f"{prefix}/data.pkl"))).load()
for k,v in sd.items(): print(k, v.shape, v.dtype)
np.savez(sys.argv[2], **{k.replace('.','_'):v for k,v in sd.items()})
