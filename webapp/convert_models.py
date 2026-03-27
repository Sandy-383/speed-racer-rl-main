"""
Run this script ONCE if the web app fails to load the C++ saved .pt models.
It creates Python-compatible copies in sampleModels/py_<name>.pt
"""
import os, sys, glob, torch
sys.path.insert(0, os.path.dirname(__file__))
from model_loader import DQNNet, load_model

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sampleModels')

for path in sorted(glob.glob(os.path.join(MODELS_DIR, '*.pt'))):
    name = os.path.basename(path)
    if name.startswith('py_'):
        continue
    model, err = load_model(path)
    out = os.path.join(MODELS_DIR, 'py_' + name)
    if model:
        torch.save(model.state_dict(), out)
        print(f'  ✓  {name}  →  py_{name}')
    else:
        print(f'  ✗  {name}  (could not load: {err})')
        # Save a random-weight model so the demo still runs
        m = DQNNet()
        torch.save(m.state_dict(), out)
        print(f'     → saved random-weight placeholder as py_{name}')

print('\nDone. Restart the web app to use the converted models.')
