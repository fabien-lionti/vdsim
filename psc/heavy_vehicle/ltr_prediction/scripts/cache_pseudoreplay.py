#!/usr/bin/env python3
"""Cache pseudo-replay scenarios to pickle for fast subsequent loads."""
import sys
import pickle
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pseudo_replay_loader import load_sim_scenarios, PSEUDO_DIR

CACHE = Path("/Users/zak/Documents/vdsim/psc/heavy_vehicle/ltr_prediction/outputs/pseudoreplay_cache.pkl")

if CACHE.exists():
    print(f"Cache existe déjà : {CACHE} ({CACHE.stat().st_size // 1024 // 1024} MB)")
    t0 = time.time()
    with open(CACHE, 'rb') as f:
        scs = pickle.load(f)
    print(f"Rechargé {len(scs)} scénarios en {time.time()-t0:.1f}s (test)")
else:
    t0 = time.time()
    scs = load_sim_scenarios(PSEUDO_DIR)
    print(f"Loaded {len(scs)} in {time.time()-t0:.1f}s")
    with open(CACHE, 'wb') as f:
        pickle.dump(scs, f)
    print(f"Saved to {CACHE}")
