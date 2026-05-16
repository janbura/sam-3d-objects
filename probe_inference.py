"""Probe: what's actually inside Inference?"""
import sys
sys.path.append("notebook")
from inference import Inference

print(">> instantiating Inference (this takes a moment)...")
inf = Inference("checkpoints/hf/pipeline.yaml", compile=False)

print(">> type(inf):", type(inf).__mro__)
print(">> all attributes of inf (incl. underscored):")
for a in dir(inf):
    if a.startswith("__"):
        continue
    try:
        v = getattr(inf, a)
        print(f"   {a:30s} -> {type(v).__name__}")
    except Exception as e:
        print(f"   {a:30s} -> <error: {e}>")

print()
print(">> __dict__ keys:")
for k, v in inf.__dict__.items():
    print(f"   {k:30s} -> {type(v).__name__}")
