import torch
import torch.nn as nn
import zipfile
import io
import numpy as np


class DQNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(23, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 7)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


TENSOR_SHAPES = [
    ('fc1.weight', (64, 23)),
    ('fc1.bias',   (64,)),
    ('fc2.weight', (64, 64)),
    ('fc2.bias',   (64,)),
    ('fc3.weight', (7, 64)),
    ('fc3.bias',   (7,)),
]


def _try_zip_extract(path):
    """Directly extract float32 tensors from a LibTorch .pt ZIP archive."""
    with zipfile.ZipFile(path, 'r') as zf:
        names = zf.namelist()
        data_files = sorted(
            [n for n in names if n.split('/')[-1].isdigit()],
            key=lambda x: int(x.split('/')[-1])
        )
        if len(data_files) < 6:
            return None

        model = DQNNet()
        state_dict = {}
        for i, (key, shape) in enumerate(TENSOR_SHAPES):
            raw = zf.read(data_files[i])
            arr = np.frombuffer(raw, dtype=np.float32)
            expected = 1
            for d in shape:
                expected *= d
            if arr.size != expected:
                return None
            state_dict[key] = torch.from_numpy(arr.reshape(shape).copy())

        model.load_state_dict(state_dict)
        model.eval()
        return model


def load_model(path):
    """Load a DQN model saved by C++ LibTorch or Python PyTorch."""
    errors = []

    # Strategy 1: standard torch.load (works for Python-saved models)
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = DQNNet()
        if isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        elif hasattr(ckpt, 'state_dict'):
            sd = ckpt.state_dict()
            # remap keys if needed (C++ might prefix differently)
            clean = {k.replace('policy_net_.', ''): v for k, v in sd.items()}
            model.load_state_dict(clean)
        else:
            raise ValueError("Unknown checkpoint format")
        model.eval()
        return model, None
    except Exception as e:
        errors.append(f"torch.load: {e}")

    # Strategy 2: TorchScript / JIT load
    try:
        scripted = torch.jit.load(path, map_location='cpu')
        model = DQNNet()
        params = dict(scripted.named_parameters())
        sd = {k: v.detach() for k, v in params.items() if k in dict(model.named_parameters())}
        if len(sd) == 6:
            model.load_state_dict(sd)
            model.eval()
            return model, None
    except Exception as e:
        errors.append(f"jit.load: {e}")

    # Strategy 3: raw ZIP tensor extraction (for C++ LibTorch saves)
    try:
        model = _try_zip_extract(path)
        if model is not None:
            return model, None
    except Exception as e:
        errors.append(f"zip_extract: {e}")

    return None, " | ".join(errors)


def predict(model, state):
    with torch.no_grad():
        t = torch.FloatTensor(state).unsqueeze(0)
        q = model(t).squeeze()
        q_list = q.tolist()
        action = int(q.argmax().item())
    return action, q_list
