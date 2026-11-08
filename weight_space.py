"""
Weight-space arithmetic on the live model — stage 1 of Weight Space
Modelling (see progress_and_plan.md).

Treats weights as vectors: load a second checkpoint of the same
architecture and form task vectors / merges:

- task vector add    W += lam * (R - W)      (approach/move toward R)
- task vector negate W -= lam * (R - W)      (move away from R)
- soup/average       W =  (1-lam)*W + lam*R  (== task add; lam in [0,1])
- delta basis        returns R - W without applying (for later PCA/viz)

Only parameters whose names AND shapes match are merged — safe across
variants of the same architecture. Touched parameters are backed up so a
merge can be undone. The merge writes weights in place, so it composes
with inversion, sculpt ops, and steering hooks like any other edit.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

MERGE_MODES = ("task_add", "task_negate", "soup")


class WeightMerger:
    """Task-vector merges between the live model and a reference checkpoint."""

    def __init__(self, model):
        self.model = model
        self.ref_model = None
        self.ref_name: Optional[str] = None
        self._backup: Dict[str, torch.Tensor] = {}
        self.history: List[Dict[str, Any]] = []
        # stage-2 ensemble state
        self.delta_sets: List[Dict[str, Any]] = []
        self._baseline: Optional[Dict[str, torch.Tensor]] = None
        self._baseline_filter: str = ""
        self._basis: Optional[Dict[str, Any]] = None
        self._latent_anchor: Optional[Dict[str, torch.Tensor]] = None

    # ---------- reference checkpoint ----------

    def load_reference(self, name_or_path: str) -> str:
        """Load a second checkpoint to merge against (same arch as live model)."""
        from transformers import AutoModelForCausalLM

        if self.model is None:
            return "Load a base model first."
        try:
            logger.info(f"Loading reference model {name_or_path}")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                torch_dtype=next(self.model.parameters()).dtype,
            )
            self.ref_name = name_or_path
            n = sum(p.numel() for p in self.ref_model.parameters())
            return f"Reference loaded: {name_or_path} ({n:,} params)"
        except Exception as e:
            self.ref_model = None
            self.ref_name = None
            return f"Could not load reference: {e}"

    def release_reference(self) -> str:
        """Free the reference model's memory."""
        if self.ref_model is None:
            return "No reference loaded."
        name = self.ref_name
        self.ref_model = None
        self.ref_name = None
        import gc
        gc.collect()
        return f"Released reference {name}."

    # ---------- merging ----------

    def _shared_params(self) -> Tuple[List[Tuple[str, torch.Tensor, torch.Tensor]], int]:
        """(name, live_param, ref_param) for matching names+shapes; + skip count."""
        if self.ref_model is None:
            return [], 0
        ref_map = dict(self.ref_model.named_parameters())
        shared, skipped = [], 0
        for name, p in self.model.named_parameters():
            r = ref_map.get(name)
            if r is None or r.shape != p.shape:
                skipped += 1
                continue
            shared.append((name, p, r))
        return shared, skipped

    def merge(self, mode: str = "task_add", lam: float = 1.0,
              name_filter: str = "") -> Dict[str, Any]:
        """Apply a merge in place. Returns stats.

        name_filter optionally limits to params whose name contains it
        (e.g. 'mlp' or 'transformer.h.3').
        """
        if self.model is None:
            return {"error": "Load a base model first."}
        if self.ref_model is None:
            return {"error": "Load a reference model first."}
        if mode not in MERGE_MODES:
            return {"error": f"Unknown mode {mode}; use one of {MERGE_MODES}."}

        shared, skipped = self._shared_params()
        if name_filter:
            shared = [(n, p, r) for n, p, r in shared if name_filter in n]
        if not shared:
            return {"error": "No matching parameters to merge."}

        with torch.no_grad():
            for name, p, r in shared:
                if name not in self._backup:
                    self._backup[name] = p.data.clone()
                if mode == "task_negate":
                    p.data -= lam * (r.data - p.data)
                elif mode == "soup":
                    p.data.mul_(1.0 - lam).add_(r.data, alpha=lam)
                else:  # task_add
                    p.data += lam * (r.data - p.data)

        n_params = len(shared)
        self.history.append({
            "mode": mode, "lam": lam, "ref": self.ref_name,
            "params": n_params, "filter": name_filter,
        })
        return {
            "merged": n_params,
            "skipped_mismatch": skipped,
            "filter": name_filter or "(all)",
            "mode": mode,
            "lam": lam,
        }

    def delta(self, name_filter: str = "") -> Dict[str, torch.Tensor]:
        """R - W per matching param (no apply) — raw material for later
        PCA/latent work."""
        shared, _ = self._shared_params()
        out = {}
        with torch.no_grad():
            for name, p, r in shared:
                if not name_filter or name_filter in name:
                    out[name] = (r.data - p.data).float().cpu()
        return out

    # ---------- undo ----------

    def undo(self) -> str:
        if not self._backup:
            return "Nothing to undo."
        live = dict(self.model.named_parameters())
        n = 0
        with torch.no_grad():
            for name, orig in self._backup.items():
                if name in live:
                    live[name].data.copy_(orig)
                    n += 1
        self._backup = {}
        self.history.append({"mode": "undo", "params": n})
        return f"Restored {n} parameters."

    def has_edits(self) -> bool:
        return bool(self._backup)

    # ---------- latent basis (stage 2 starter) ----------

    @staticmethod
    def latent_basis(
        delta_sets: List[Dict[str, torch.Tensor]], n_components: int = 8
    ) -> Dict[str, Any]:
        """PCA over a set of weight-delta dicts (from ``delta()`` or elsewhere).

        delta_sets: list of {param_name: delta_tensor} dicts — e.g. one per
        merge, sweep variant, or marshal. Params common to all sets are
        stacked into rows and reduced jointly.

        Returns per-component variance share and per-param component slices
        so callers can reassemble latent-space edits.
        """
        if len(delta_sets) < 2:
            return {"error": "Need at least 2 delta sets for a basis."}
        # union of names — a param absent from a set is a zero delta there
        shapes = {}
        for d in delta_sets:
            for n, t in d.items():
                shapes.setdefault(n, tuple(t.shape))
        names = [
            n for n in shapes
            if all(tuple(d[n].shape) == shapes[n] for d in delta_sets if n in d)
        ]
        if not names:
            return {"error": "No parameter names across delta sets."}
        rows = []
        for d in delta_sets:
            rows.append(torch.cat([
                d[n].flatten().float() if n in d
                else torch.zeros(int(torch.tensor(shapes[n]).prod()))
                for n in names
            ]))
        mean_delta_norm = float(torch.stack(rows).norm(dim=1).mean())
        X = torch.stack(rows)
        X = X - X.mean(dim=0, keepdim=True)
        _, S, V = torch.pca_lowrank(X, q=min(n_components, X.shape[0] - 1, X.shape[1]))
        var = (S ** 2) / max(1, X.shape[0] - 1)
        total = float(var.sum()) or 1.0

        # split each component back into per-param slices
        comp_slices, offset = [], 0
        sizes = [int(torch.tensor(shapes[n]).prod()) for n in names]
        for c in range(V.shape[1]):
            slices, offset = {}, 0
            for n, sz in zip(names, sizes):
                slices[n] = V[offset:offset + sz, c].reshape(shapes[n])
                offset += sz
            comp_slices.append(slices)
        return {
            "names": names,
            "shapes": shapes,
            "variance_share": [float(v / total) for v in var.tolist()],
            "components": comp_slices,  # component c -> {name: tensor}
            "mean_delta_norm": mean_delta_norm,
        }

    # ---------- delta ensembles (stage 2) ----------

    def set_baseline(self, name_filter: str = "") -> str:
        """Capture current weights (matching filter) as the ensemble baseline.
        Clears any recorded delta sets."""
        if self.model is None:
            return "Load a base model first."
        self._baseline = {
            n: p.data.detach().clone()
            for n, p in self.model.named_parameters()
            if not name_filter or name_filter in n
        }
        if not self._baseline:
            return f"Baseline is empty — no params match '{name_filter}'."
        self._baseline_filter = name_filter
        self.delta_sets = []
        self._basis = None
        self._latent_anchor = None
        return (
            f"Baseline set: {len(self._baseline)} params "
            f"[{name_filter or 'all'}]. Delta sets cleared."
        )

    def add_delta_set(self, label: str = "") -> str:
        """Record (current weights - baseline) for baseline params as a delta
        set. Apply a weight-editing op first; structural changes (inserted or
        removed layers) simply won't match baseline names."""
        if self._baseline is None:
            return "Set a baseline first."
        live = dict(self.model.named_parameters())
        delta = {}
        with torch.no_grad():
            for n, b in self._baseline.items():
                p = live.get(n)
                if p is not None and p.shape == b.shape:
                    d = (p.data - b).float().cpu()
                    if float(d.abs().max()) > 0:
                        delta[n] = d
        if not delta:
            return "No changed params vs baseline — apply a weight edit first."
        label = label or f"set{len(self.delta_sets)}"
        self.delta_sets.append({"label": label, "delta": delta})
        self._basis = None
        return (
            f"Recorded '{label}': {len(delta)} changed params. "
            f"{len(self.delta_sets)} sets total."
        )

    def clear_delta_sets(self) -> str:
        self.delta_sets = []
        self._basis = None
        return "Delta sets cleared."

    def auto_ensemble(self, k: int = 6, edits_per_set: int = 3,
                      name_filter: Optional[str] = None,
                      seed: int = 0) -> str:
        """Auto-generate K delta sets via structured edits: for each set, pick
        ``edits_per_set`` random >=2D params and rescale a random slice along
        the last dim (like a neuron scale_write). Each variant is snapshotted
        then exactly restored — structured deltas give the basis real
        directions instead of isotropic noise."""
        import random

        if self._baseline is None:
            msg = self.set_baseline(name_filter or "")
            if "empty" in msg or "first" in msg:
                return msg
        rnd = random.Random(seed)
        live = dict(self.model.named_parameters())
        cand = [
            n for n, b in self._baseline.items()
            if n in live and live[n].dim() >= 2
            and live[n].shape == b.shape
        ]
        if not cand:
            return "No eligible >=2D params for auto-ensemble."
        made = 0
        for i in range(k):
            picks = rnd.sample(cand, min(edits_per_set, len(cand)))
            touched, desc = {}, []
            with torch.no_grad():
                for n in picks:
                    p = live[n]
                    j = rnd.randrange(p.shape[-1])
                    factor = rnd.uniform(0.0, 2.0)
                    touched[n] = (j, p.data[..., j].clone())
                    p.data[..., j].mul_(factor)
                    desc.append(f"{'.'.join(n.split('.')[-4:])}[{j}]x{factor:.2f}")
                r = self.add_delta_set(f"auto{i}: " + "; ".join(desc))
                for n, (j, orig) in touched.items():
                    live[n].data[..., j].copy_(orig)
            if "Recorded" in r:
                made += 1
        return f"Auto ensemble: {made}/{k} sets recorded ({len(cand)} candidate params)."

    def compute_basis(self, n_components: int = 8) -> Dict[str, Any]:
        if len(self.delta_sets) < 2:
            return {"error": "Need at least 2 delta sets — record or auto-generate."}
        r = self.latent_basis([s["delta"] for s in self.delta_sets], n_components)
        if "error" not in r:
            self._basis = r
            self._latent_anchor = None
        return r

    def apply_component(self, comp_idx: int, scale: float) -> str:
        """One-shot add of ``scale * PC[comp_idx]`` — accumulates. For
        slider-driven control prefer ``set_component_mix`` (idempotent)."""
        return self._apply_mix({int(comp_idx): float(scale)},
                               incremental=True)

    def set_component_mix(self, scales: List[float]) -> str:
        """Set absolute PC slider positions — idempotent: writes
        ``W = anchor + Σ s_i · mean_delta_norm · PC_i`` where anchor is the
        weights captured on the first mix call. scale 1.0 ≈ one typical
        delta-set magnitude. Touched params join the merge backup so Undo
        fully restores."""
        return self._apply_mix({i: float(s) for i, s in enumerate(scales)},
                               incremental=False)

    def _apply_mix(self, scale_map: Dict[int, float],
                   incremental: bool) -> str:
        if not self._basis:
            return "Compute a basis first."
        comps = self._basis["components"]
        norm = self._basis.get("mean_delta_norm", 1.0)
        live = dict(self.model.named_parameters())

        if not incremental and self._latent_anchor is None:
            self._latent_anchor = {
                n: live[n].data.clone() for n in self._basis["names"]
                if n in live and live[n].shape == self._basis["shapes"][n]
            }

        with torch.no_grad():
            if not incremental:
                for n, a in self._latent_anchor.items():
                    if n in live and live[n].shape == a.shape:
                        live[n].data.copy_(a)
            applied = 0
            for idx, s in scale_map.items():
                if not 0 <= idx < len(comps) or s == 0:
                    continue
                for n, comp in comps[idx].items():
                    p = live.get(n)
                    if p is None or p.shape != comp.shape:
                        continue
                    if n not in self._backup:
                        self._backup[n] = p.data.clone()
                    p.data += s * norm * comp.to(p.data.dtype)
                    applied += 1
        desc = ", ".join(f"PC{i}x{s:g}" for i, s in scale_map.items() if s)
        self.history.append({
            "mode": "latent_mix" if not incremental else "latent_pc",
            "lam": norm, "ref": desc, "params": applied, "filter": "",
        })
        return f"Applied [{desc}] (norm={norm:.4g}) to {applied} param-slices (undo restores)."

    def basis_summary(self) -> str:
        if not self._basis:
            return "No basis computed."
        shares = self._basis["variance_share"]
        line = "  ".join(f"PC{i}:{s * 100:.1f}%" for i, s in enumerate(shares))
        return f"{len(self._basis['names'])} shared params\n{line}"

    def sets_summary(self) -> str:
        if not self.delta_sets:
            return "No delta sets recorded."
        return "\n".join(
            f"{i}: {s['label']} ({len(s['delta'])} params)"
            for i, s in enumerate(self.delta_sets)
        )

    def describe(self) -> str:
        if not self.history:
            return "No merges applied."
        lines = []
        for h in self.history[-10:]:
            if h["mode"] == "undo":
                lines.append(f"undo ({h['params']} params)")
            else:
                lines.append(
                    f"{h['mode']} lam={h['lam']} ref={h['ref']} "
                    f"on {h['params']} params"
                    + (f" [{h['filter']}]" if h["filter"] else "")
                )
        pending = f" ({len(self._backup)} params pending undo)" if self._backup else ""
        return "\n".join(lines) + pending
