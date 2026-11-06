"""
Layer Surgery — add, remove, and graft decoder blocks on a loaded HF model.

Blocks are deepcopied from an existing layer and spliced into the model's
repeating block stack (GPT-2 `transformer.h`, Llama `model.layers`, OPT
`model.decoder.layers`, ...). Init modes:

- ``copy``     keep the source block's weights (behaves like running that
               layer twice)
- ``copy_noise`` copy weights plus gaussian noise scaled to each tensor's
               std — breaks the identical-function symmetry of a plain copy
- ``identity`` zero the block's output projections (weight + bias), so a
               residual block passes its input through almost unchanged —
               the llama-pro / depth-up-scaling trick
- ``random``   small-std normal init (scrambles behavior; tinker option)

Deferred / TODO (see progress_and_plan.md):
- Cross-checkpoint grafting: load a second HF model's state dict and pull
  blocks or individual tensors from it, with shape checks.
- Gated residual wrapper  y = x + a * block(x)  with a tunable/trainable a.
- Training for added layers on RunPod: freeze all other params, LM loss on
  pasted text or distillation against a pre-surgery output snapshot.
"""

import copy
import inspect
import logging
import torch
import torch.nn as nn
from datetime import datetime
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ResidualWrap(nn.Module):
    """Wraps a copied submodule so it can sit in the decoder block stack.

    Computes ``y = x + inner(x)``. Extra positional/keyword args the parent
    loop passes to blocks (attention masks, position ids, cache flags) are
    filtered to what ``inner.forward`` actually accepts; tuple outputs from
    attention-style submodules are reduced to their hidden-state element.

    ``returns_tuple`` must match the sibling blocks' convention (GPT-2 style
    blocks return tuples; newer Llama-style blocks return bare tensors) or the
    parent loop will misread the output. Pad the tuple with Nones so
    ``outputs[1]``/``outputs[2]`` indexing for cache/attentions does not crash.
    """

    def __init__(self, inner: nn.Module, returns_tuple: bool):
        super().__init__()
        self.inner = inner
        self.returns_tuple = returns_tuple
        try:
            sig = inspect.signature(inner.forward)
            params = sig.parameters
            self._accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            self._accepted_kw = set(params) - {"self"}
        except (TypeError, ValueError):
            self._accepts_var_kw = True
            self._accepted_kw = set()

    def forward(self, hidden_states, *args, **kwargs):
        if self._accepts_var_kw:
            inner_kwargs = kwargs
        else:
            inner_kwargs = {k: v for k, v in kwargs.items() if k in self._accepted_kw}
        res = self.inner(hidden_states, **inner_kwargs)
        if isinstance(res, (tuple, list)):
            res = res[0]
        out = hidden_states + res
        if self.returns_tuple:
            return (out, None, None)
        return out


class LayerSurgeon:
    """Structural edits on a loaded causal LM's decoder block stack."""

    # config attribute names that store decoder depth, by arch family
    DEPTH_CONFIG_ATTRS = (
        "num_hidden_layers",  # llama, opt, bloom, ...
        "n_layer",            # gpt2, gpt-neo, gpt-j
        "num_layers",
        "n_layers",
    )

    # param-name substrings for a block's *output* projections; zeroing
    # weight+bias makes a residual block an identity function
    OUT_PROJ_PATTERNS = (
        "o_proj",               # llama/mistral attn out
        "out_proj",             # opt/gpt-neo attn out
        "c_proj",               # gpt2 attn out AND mlp out
        "down_proj",            # llama/mistral mlp out
        "fc2",                  # opt mlp out
        "dense_4h_to_h",        # bloom/falcon mlp out
        "self_attention.dense", # bloom/falcon attn out
        "attention.dense",      # gpt-j attn out
    )

    def __init__(self, model):
        self.model = model
        self.history = []
        self._added_ids = set()  # id() of blocks inserted this session
        self._hooks_stripped = False
        self._stack_cache: Optional[Tuple[str, nn.ModuleList]] = None

    # ---------- stack discovery ----------

    def _expected_depth(self) -> Optional[int]:
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return None
        for attr in self.DEPTH_CONFIG_ATTRS:
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        return None

    def find_block_stack(self) -> Tuple[str, nn.ModuleList]:
        """Locate the repeating decoder block ModuleList.

        Picks the uniform-child-type ModuleList whose length matches the
        configured depth; falls back to the longest uniform ModuleList.
        Result is cached — a ResidualWrap insertion makes the stack
        non-uniform, but the ModuleList object itself persists.
        """
        if self._stack_cache is not None:
            return self._stack_cache
        expected = self._expected_depth()
        best = None
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.ModuleList) or len(module) < 2:
                continue
            if len({type(m) for m in module}) != 1:
                continue
            score = (1 if expected and len(module) == expected else 0, len(module))
            if best is None or score > best[0]:
                best = (score, name, module)
        if best is None:
            raise RuntimeError(
                "No repeating block stack (uniform nn.ModuleList) found in this model."
            )
        self._stack_cache = (best[1], best[2])
        return best[1], best[2]

    def describe(self) -> str:
        path, blocks = self.find_block_stack()
        cls = type(blocks[0]).__name__
        return f"Block stack: {path} — {len(blocks)} x {cls}"

    def stack_overview(self) -> str:
        try:
            path, blocks = self.find_block_stack()
        except Exception as e:
            return str(e)
        lines = [f"{path}:"]
        for i, block in enumerate(blocks):
            n_params = sum(p.numel() for p in block.parameters())
            cls = type(block).__name__
            if isinstance(block, ResidualWrap):
                cls = f"ResidualWrap({type(block.inner).__name__})"
            mark = "  [added]" if id(block) in self._added_ids else ""
            lines.append(f"{i:3d}: {cls} ({n_params / 1e6:.2f}M){mark}")
        return "\n".join(lines)

    # ---------- structural ops ----------

    def insert_layer(
        self,
        position: Optional[int] = None,
        source_index: Optional[int] = None,
        init: str = "copy",
        noise_scale: float = 0.01,
    ) -> str:
        """Deepcopy a block and splice it into the stack at ``position``."""
        path, blocks = self.find_block_stack()
        n = len(blocks)
        position = n if position is None else max(0, min(int(position), n))
        if source_index is None:
            # default template: the block just before the insertion point
            source_index = position - 1
        source_index = max(0, min(int(source_index), n - 1))

        self._strip_dispatch_hooks()
        new_block = copy.deepcopy(blocks[source_index])

        init = (init or "copy").lower().replace("+", "_")
        detail = ""
        if init == "identity":
            zeroed = self._zero_output_projections(new_block)
            detail = f", identity init ({len(zeroed)} out-proj tensors zeroed)"
            if not zeroed:
                detail = (
                    ", identity init FAILED (no out-projection params matched; "
                    "block keeps a copy of the source weights)"
                )
        elif init == "random":
            self._randomize_block(new_block)
            detail = ", random init"
        elif init == "copy_noise":
            self._add_noise(new_block, noise_scale)
            detail = f", copy+noise ({noise_scale} x each tensor's std)"
        elif init != "copy":
            raise ValueError(
                f"Unknown init '{init}' (use copy/copy_noise/identity/random)"
            )

        blocks.insert(position, new_block)
        self._added_ids.add(id(new_block))
        self._after_structure_change(blocks)
        msg = (
            f"Inserted {type(new_block).__name__} at {path}[{position}] "
            f"(template: layer {source_index}{detail}). Depth now {len(blocks)}."
        )
        self._record("insert", msg)
        return msg

    def remove_layer(self, index: int) -> str:
        """Delete the block at ``index``. Refuses to remove the last block."""
        path, blocks = self.find_block_stack()
        n = len(blocks)
        if n <= 1:
            raise ValueError("Refusing to remove the last remaining block.")
        index = int(index)
        if not 0 <= index < n:
            raise IndexError(f"Index {index} out of range (0..{n - 1}).")

        self._strip_dispatch_hooks()
        removed = blocks[index]
        del blocks[index]
        self._after_structure_change(blocks)
        msg = (
            f"Removed {type(removed).__name__} at {path}[{index}]. "
            f"Depth now {len(blocks)}."
        )
        self._record("remove", msg)
        return msg

    def copy_layer_weights(self, source_index: int, target_index: int) -> str:
        """Copy all of one block's weights onto another (same model graft)."""
        path, _ = self.find_block_stack()
        n = self.block_count()
        for idx in (int(source_index), int(target_index)):
            if not 0 <= idx < n:
                raise IndexError(f"Index {idx} out of range (0..{n - 1}).")
        return self.copy_module_weights(
            f"{path}.{source_index}", f"{path}.{target_index}"
        )

    def copy_module_weights(self, source_name: str, target_name: str) -> str:
        """Graft weights between any two same-shaped submodules.

        Works at any granularity: whole blocks, ``mlp``, ``attn``, single
        Linear/Embedding, LayerNorm — wherever ``state_dict`` keys and shapes
        match. Cross-checkpoint grafting is a TODO (see module docstring).
        """
        modules = dict(self.model.named_modules())
        src = modules.get(source_name)
        dst = modules.get(target_name)
        if src is None:
            raise KeyError(f"Source module '{source_name}' not found.")
        if dst is None:
            raise KeyError(f"Target module '{target_name}' not found.")
        if not any(True for _ in src.parameters()):
            raise ValueError(f"Source module '{source_name}' owns no parameters.")
        sd = src.state_dict()
        dst.load_state_dict(sd, strict=True)
        msg = f"Copied {len(sd)} tensors: {source_name} -> {target_name}."
        self._record("graft", msg)
        return msg

    def graftable_modules(self) -> list:
        """Module names that own parameters — valid graft src/dst targets.

        Excludes ancestors of the block stack (grafting the whole stack onto
        something would only ever fail a shape check).
        """
        try:
            stack_path, _ = self.find_block_stack()
        except Exception:
            stack_path = None
        names = []
        for name, module in self.model.named_modules():
            if not name:
                continue
            if stack_path and (
                stack_path == name or stack_path.startswith(name + ".")
            ):
                continue
            if any(True for _ in module.parameters()):
                names.append(name)
        return names

    def block_count(self) -> int:
        _, blocks = self.find_block_stack()
        return len(blocks)

    def list_submodules(self, block_index: int) -> list:
        """Param-owning submodule names inside a block (mlp, attn, ln_1, ...)."""
        _, blocks = self.find_block_stack()
        n = len(blocks)
        block_index = int(block_index)
        if not 0 <= block_index < n:
            raise IndexError(f"Index {block_index} out of range (0..{n - 1}).")
        return [
            name
            for name, module in blocks[block_index].named_modules()
            if name and any(True for _ in module.parameters())
        ]

    def insert_sublayer(
        self,
        position: Optional[int],
        block_index: int,
        submodule_name: str,
        init: str = "copy",
        noise_scale: float = 0.01,
    ) -> str:
        """Deepcopy a submodule (e.g. ``mlp``) out of a block, wrap it
        residually, and splice it into the stack as a new layer.

        Note: attention submodules technically work (their kwargs are passed
        through) but double-applying attention is rarely what you want —
        MLP / norm-ish pieces are the useful targets.
        """
        path, blocks = self.find_block_stack()
        n = len(blocks)
        position = n if position is None else max(0, min(int(position), n))
        block_index = int(block_index)
        if not 0 <= block_index < n:
            raise IndexError(f"Block index {block_index} out of range (0..{n - 1}).")
        submodules = dict(blocks[block_index].named_modules())
        if submodule_name not in submodules:
            raise KeyError(
                f"'{submodule_name}' not found in block {block_index}. "
                f"Options: {sorted(submodules)}"
            )

        self._strip_dispatch_hooks()
        inner = copy.deepcopy(submodules[submodule_name])
        init = (init or "copy").lower().replace("+", "_")
        detail = ""
        if init == "identity":
            zeroed = self._zero_output_projections(inner)
            detail = f", identity init ({len(zeroed)} out-proj tensors zeroed)"
            if not zeroed:
                detail = ", identity init found no out-projections to zero"
        elif init == "random":
            self._randomize_block(inner)
            detail = ", random init"
        elif init == "copy_noise":
            self._add_noise(inner, noise_scale)
            detail = f", copy+noise ({noise_scale} x each tensor's std)"
        elif init != "copy":
            raise ValueError(
                f"Unknown init '{init}' (use copy/copy_noise/identity/random)"
            )

        wrapped = ResidualWrap(inner, returns_tuple=self._block_returns_tuple(blocks))
        blocks.insert(position, wrapped)
        self._added_ids.add(id(wrapped))
        self._after_structure_change(blocks)
        msg = (
            f"Inserted ResidualWrap({type(inner).__name__}) at {path}[{position}] "
            f"(source: block {block_index}.{submodule_name}{detail}). "
            f"Depth now {len(blocks)}."
        )
        self._record("insert_sublayer", msg)
        return msg

    def move_layer(self, source_index: int, target_index: int) -> str:
        """Move the block at ``source_index`` to ``target_index`` in the stack.

        Reorders without copying — same module object, new position.
        """
        path, blocks = self.find_block_stack()
        n = len(blocks)
        source_index = int(source_index)
        if not 0 <= source_index < n:
            raise IndexError(f"Index {source_index} out of range (0..{n - 1}).")
        block = blocks[source_index]
        del blocks[source_index]
        target_index = max(0, min(int(target_index), len(blocks)))
        blocks.insert(target_index, block)
        self._after_structure_change(blocks)
        cls = type(block).__name__
        if isinstance(block, ResidualWrap):
            cls = f"ResidualWrap({type(block.inner).__name__})"
        msg = f"Moved {cls}: {path}[{source_index}] -> position {target_index}."
        self._record("move", msg)
        return msg

    def jitter_layer(self, index: int, scale: float) -> str:
        """Add gaussian noise to an existing block's weights in place.

        Noise per tensor is N(0, std(tensor) * scale), so e.g. scale=0.01
        jitters each weight by ~1% of that tensor's spread.
        """
        path, blocks = self.find_block_stack()
        n = len(blocks)
        index = int(index)
        if not 0 <= index < n:
            raise IndexError(f"Index {index} out of range (0..{n - 1}).")
        self._add_noise(blocks[index], scale)
        msg = (
            f"Jittered {path}[{index}] with noise at {scale} x each tensor's std."
        )
        self._record("jitter", msg)
        return msg

    # ---------- internals ----------

    def _block_returns_tuple(self, blocks: nn.ModuleList) -> bool:
        """Probe whether sibling blocks return tuples (GPT-2/OPT style) or
        bare tensors (newer Llama style), by hooking block 0 on a 1-token
        forward. Cached after first call.
        """
        if getattr(self, "_returns_tuple", None) is not None:
            return self._returns_tuple
        captured = {}
        handle = blocks[0].register_forward_hook(
            lambda m, i, o: captured.setdefault("out", o)
        )
        try:
            device = next(blocks[0].parameters()).device
            dummy = torch.zeros((1, 1), dtype=torch.long, device=device)
            with torch.no_grad():
                self.model(input_ids=dummy)
        finally:
            handle.remove()
        self._returns_tuple = isinstance(
            captured.get("out"), (tuple, list)
        )
        return self._returns_tuple

    def _add_noise(self, module: nn.Module, scale: float):
        """Add gaussian noise proportional to each tensor's own std.

        Zero-std tensors (e.g. identity-zeroed projections) fall back to a
        small absolute noise floor so they still get perturbed.
        """
        scale = float(scale)
        with torch.no_grad():
            for param in module.parameters():
                std = float(param.data.std())
                param.data.add_(
                    torch.randn_like(param) * ((std if std > 0 else 0.02) * scale)
                )

    def _zero_output_projections(self, block: nn.Module) -> list:
        zeroed = []
        for name, param in block.named_parameters():
            if any(pat in name for pat in self.OUT_PROJ_PATTERNS):
                param.data.zero_()
                zeroed.append(name)
        return zeroed

    def _randomize_block(self, block: nn.Module):
        with torch.no_grad():
            for name, param in block.named_parameters():
                if "bias" in name:
                    param.zero_()
                elif param.ndim >= 2:
                    param.normal_(0.0, 0.02)
                elif "ln" in name or "norm" in name or "layernorm" in name:
                    param.fill_(1.0)  # norm weights stay at 1
                else:
                    param.normal_(0.0, 0.02)

    def _after_structure_change(self, blocks: nn.ModuleList):
        self._renumber_layer_indices(blocks)
        self._sync_depth_config(len(blocks))

    def _renumber_layer_indices(self, blocks: nn.ModuleList):
        """Fix self_attn.layer_idx / similar after insert/remove.

        Newer transformers use per-layer indices for KV-cache bookkeeping;
        a deepcopied block would otherwise collide with its template.
        """
        for i, block in enumerate(blocks):
            for sub in block.modules():
                if hasattr(sub, "layer_idx"):
                    try:
                        sub.layer_idx = i
                    except Exception:
                        pass

    def _sync_depth_config(self, count: int):
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return
        for attr in self.DEPTH_CONFIG_ATTRS:
            if hasattr(cfg, attr):
                try:
                    setattr(cfg, attr, count)
                except Exception:
                    pass

    def _strip_dispatch_hooks(self):
        """Remove accelerate device hooks before mutating the module tree.

        deepcopy would clone AlignDevicesHooks whose execution-device maps
        are keyed by the *old* module names. Single-device models are
        unaffected; multi-device sharded models lose auto input dispatch.
        """
        if self._hooks_stripped:
            return
        try:
            device_map = getattr(self.model, "hf_device_map", None)
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(self.model)
            self._hooks_stripped = True
            if device_map and len({str(d) for d in device_map.values()}) > 1:
                logger.warning(
                    "Layer surgery on a multi-device model: accelerate dispatch "
                    "hooks were removed; params keep their devices but inputs "
                    "are no longer auto-moved between devices."
                )
        except Exception as e:
            logger.warning(f"Could not strip accelerate hooks: {e}")

    def _record(self, op: str, msg: str):
        self.history.append(
            {"op": op, "message": msg, "timestamp": datetime.now().isoformat()}
        )
