"""Smoke test for LayerSurgeon on cached gpt2."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from layer_surgery import LayerSurgeon

model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
tok = AutoTokenizer.from_pretrained("gpt2")
s = LayerSurgeon(model)

print(s.describe())
print(s.stack_overview()[:400])

inp = tok("The quick brown fox", return_tensors="pt")

def current_logits():
    with torch.no_grad():
        return model(**inp).logits

base_logits = current_logits()

def maxlogit_diff(ref=None):
    return (current_logits() - (ref if ref is not None else base_logits)).abs().max().item()

# --- insert identity layer at index 6 ---
print("\n-- insert identity @6 --")
print(s.insert_layer(position=6, init="identity"))
print("depth:", len(model.transformer.h), "config:", model.config.n_layer)
d = maxlogit_diff()
print("max |logit diff| vs original:", d)
assert d < 1e-3, f"identity init should preserve logits, got {d}"

# --- insert copy layer at end ---
print("\n-- insert copy at end --")
print(s.insert_layer(position=14, source_index=0, init="copy"))
assert len(model.transformer.h) == 14
assert model.config.n_layer == 14
d = maxlogit_diff()
print("max |logit diff| after copy-insert:", d)

# --- graft layer 0 -> layer 2 ---
print("\n-- graft 0 -> 2 --")
print(s.copy_layer_weights(0, 2))
sd0 = model.transformer.h[0].state_dict()
sd2 = model.transformer.h[2].state_dict()
assert all(torch.equal(sd0[k], sd2[k]) for k in sd0), "graft mismatch"
print("graft verified: layer 2 == layer 0")

# --- random insert ---
print("\n-- insert random @1 --")
print(s.insert_layer(position=1, init="random"))
assert len(model.transformer.h) == 15

# --- remove ---
print("\n-- remove @1 --")
print(s.remove_layer(1))
assert len(model.transformer.h) == 14
assert model.config.n_layer == 14
try:
    s.remove_layer(99)
    print("ERROR: no IndexError")
except IndexError as e:
    print("out-of-range guard ok:", e)

# --- insert MLP sublayer as residual layer ---
print("\n-- insert mlp sublayer @7 --")
print("submodules of block 0:", s.list_submodules(0))
print(s.insert_sublayer(position=7, block_index=2, submodule_name="mlp", init="copy"))
assert len(model.transformer.h) == 15
d = maxlogit_diff()
print("max |logit diff| after mlp-insert:", d)

# --- insert mlp sublayer with identity init ---
print("\n-- insert mlp sublayer identity @8 --")
pre = current_logits()  # delta vs state just before insertion
print(s.insert_sublayer(position=8, block_index=2, submodule_name="mlp", init="identity"))
d = maxlogit_diff(pre)
print("max |logit diff| caused by identity mlp:", d)
assert d < 1e-3, f"identity mlp should preserve logits, got {d}"

# --- module-level graft: mlp of block 4 -> mlp of block 5 ---
print("\n-- graft h.4.mlp -> h.5.mlp --")
print(s.copy_module_weights("transformer.h.4.mlp", "transformer.h.5.mlp"))
sd4 = model.transformer.h[4].mlp.state_dict()
sd5 = model.transformer.h[5].mlp.state_dict()
assert all(torch.equal(sd4[k], sd5[k]) for k in sd4), "mlp graft mismatch"
print("mlp graft verified")

# --- shape-mismatch graft should error cleanly ---
try:
    s.copy_module_weights("transformer.h.0.mlp", "transformer.wte")
    print("ERROR: shape mismatch graft did not fail")
except Exception as e:
    print("shape-mismatch guard ok:", str(e)[:80])

print("\ngraftable modules sample:", s.graftable_modules()[:8])

# --- copy+noise insert: weights differ slightly from template ---
print("\n-- insert copy+noise @3 --")
print(s.insert_layer(position=3, source_index=1, init="copy_noise", noise_scale=0.05))
new_block = model.transformer.h[3]
src_block = model.transformer.h[1]
diffs = []
for (n1, p_new), (_, p_src) in zip(
    new_block.named_parameters(), src_block.named_parameters()
):
    rel = (p_new - p_src).abs().max().item() / (p_src.std().item() + 1e-9)
    diffs.append(rel)
print(f"max relative deviation from template: {max(diffs):.3f} (expect ~5-25% for 5% noise)")
assert all(d > 0 for d in diffs), "copy+noise produced an identical copy"
assert max(diffs) < 0.5, "noise way too large"

# --- move layer ---
print("\n-- move layer 3 -> 10 --")
moved = model.transformer.h[3]
print(s.move_layer(3, 10))
assert model.transformer.h[10] is moved, "moved block is not the same object"
assert len(model.transformer.h) == 17, "depth changed on move"
print("move verified: same module object at new index, depth preserved")

# --- jitter existing layer ---
print("\n-- jitter layer 4 @ 0.1 --")
target = model.transformer.h[4]
before = {n: p.detach().clone() for n, p in target.named_parameters()}
print(s.jitter_layer(4, 0.1))
changed = all(
    not torch.equal(before[n], p) for n, p in target.named_parameters()
)
assert changed, "jitter did not change weights"
rel = {
    n: ((p - before[n]).abs().mean() / (before[n].std() + 1e-9)).item()
    for n, p in target.named_parameters()
}
print(f"mean relative change per tensor: ~{sum(rel.values())/len(rel):.3f} (expect ~0.08 for scale=0.1)")

# --- generate still works ---
out = model.generate(**inp, max_new_tokens=5, do_sample=False)
print("\ngen ok:", tok.decode(out[0])[:60])

print("\n" + s.stack_overview())
print("\nALL CHECKS PASSED")
