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
        names = [
            n for n in delta_sets[0]
            if all(n in d and d[n].shape == delta_sets[0][n].shape for d in delta_sets[1:])
        ]
        if not names:
            return {"error": "No common parameter names across delta sets."}
        rows, shapes = [], {}
        for d in delta_sets:
            rows.append(torch.cat([d[n].flatten().float() for n in names]))
        for n in names:
            shapes[n] = tuple(delta_sets[0][n].shape)
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
        }

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
