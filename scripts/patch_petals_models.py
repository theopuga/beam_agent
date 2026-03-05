#!/usr/bin/env python3
"""
Patch the installed petals package with support for frontier open-source models:
  - Qwen3.5 MoE  (model_type: qwen3_5_moe)
  - GLM-5         (model_type: glm_moe_dsa)
  - Kimi K2/K2.5  (model_type: kimi_k2 / kimi_k25)

Run with the petals venv Python:
  $petals_venv/bin/python scripts/patch_petals_models.py
"""
import pathlib
import sys
import sysconfig
import textwrap

site_pkgs = pathlib.Path(sysconfig.get_paths()["purelib"])
models_dir = site_pkgs / "petals" / "models"
utils_dir = site_pkgs / "petals" / "utils"

if not models_dir.exists():
    print("petals models dir not found; skipping model patches")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Shared block/model template helpers
# ---------------------------------------------------------------------------
# The block pattern is identical across MoE models — only the import
# paths and class names differ.  We generate the code from a template.

BLOCK_TEMPLATE = textwrap.dedent("""\
from typing import Optional, Tuple
import torch
from transformers.cache_utils import DynamicCache

{decoder_import}

if _DecoderLayer is not None:
    class {block_class}(_DecoderLayer):
        def __init__(self, config, layer_idx=0):
            super().__init__(config, layer_idx)
            # Ensure attn_implementation is set (Qwen3Config may leave it None)
            self._attn_implementation = getattr(config, "_attn_implementation", "eager") or "eager"
            if not getattr(config, "_attn_implementation", None):
                config._attn_implementation = "eager"
            self.layer_idx = layer_idx
            # Pre-compute RoPE for transformers>=5.x (position_embeddings required by attention)
            self._beam_rope = None
            try:
                import importlib, inspect
                mod = importlib.import_module(type(self).__mro__[1].__module__)
                for name in dir(mod):
                    if "rotary" in name.lower() and "embedding" in name.lower():
                        cls = getattr(mod, name)
                        if isinstance(cls, type):
                            sig = inspect.signature(cls.__init__)
                            if "config" in sig.parameters:
                                self._beam_rope = cls(config=config)
                            else:
                                self._beam_rope = cls(
                                    config.hidden_size // config.num_attention_heads,
                                    config.max_position_embeddings,
                                )
                            break
            except Exception:
                pass

        def forward(self, hidden_states, *args, attention_mask=None, layer_past=None, use_cache=False, **kwargs):
            bs, sl, _ = hidden_states.shape
            pkvl = 0
            sl_wp = sl

            # Always create a fresh DynamicCache; populate with past KV when available
            pkv = DynamicCache()
            if layer_past is not None:
                # beam format: k=(bs*nkv, hd, pkvl), v=(bs*nkv, pkvl, hd)
                pkvl = layer_past[0].shape[2]
                sl_wp = sl + pkvl
                k_m, v_m = self._rcfb(layer_past, bs, pkvl)
                pkv.update(k_m, v_m, layer_idx=self.layer_idx)

            pos_ids = torch.arange(pkvl, sl + pkvl, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            if "position_embeddings" not in kwargs and self._beam_rope is not None:
                kwargs["position_embeddings"] = self._beam_rope(hidden_states, pos_ids)

            # transformers>=5.x: past_key_values (plural), DynamicCache updated in-place,
            # returns tuple (hidden_states, [opt: attn_weights]) — no past_key_values in output
            outputs = super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=pos_ids,
                past_key_values=pkv,
                use_cache=use_cache,
                **kwargs
            )
            h = outputs[0]  # hidden_states is always first element

            if use_cache:
                # DynamicCache updated in-place; extract via key_cache/value_cache lists
                k_out = pkv.key_cache[self.layer_idx]    # (bs, nkv, sl_wp, hd)
                v_out = pkv.value_cache[self.layer_idx]  # (bs, nkv, sl_wp, hd)
                beam_kv = self._rctb((k_out, v_out), bs, sl_wp)
                return (h, beam_kv)
            return (h,)

        def _rcfb(self, kv, bs, sl):
            # beam format -> model format: k (bs*nkv, hd, sl) -> (bs, nkv, sl, hd)
            k, v = kv
            nkv = k.shape[0] // bs   # derived from tensor shape (avoids missing attr)
            hd = k.shape[1]
            k = k.permute(0, 2, 1).view(bs, nkv, sl, hd)
            v = v.view(bs, nkv, sl, hd)
            return k, v

        def _rctb(self, kv, bs, sl):
            # model format (bs, nkv, sl, hd) -> beam format
            k, v = kv
            nkv = k.shape[1]   # derived from tensor shape
            hd = k.shape[3]
            k = k.view(bs * nkv, sl, hd).permute(0, 2, 1)
            v = v.view(bs * nkv, sl, hd)
            return k, v
else:
    class {block_class}:
        def __init__(self, *a, **kw):
            raise ImportError("{model_name} requires a newer transformers version")
""")

CONFIG_TEMPLATE = textwrap.dedent("""\
import os
from hivemind import get_logger
from petals.client.config import ClientConfig
from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig
from petals.models.{pkg}.block import {block_class}
logger = get_logger(__name__)

{base_config_import}
{attn_import}

class {config_class}(_BaseConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = {block_class}
    attn_class = _AttnClass
    block_prefix = "model.layers"

    @property
    def num_key_value_groups(self):
        nkv = getattr(self, "num_key_value_heads", self.num_attention_heads)
        return self.num_attention_heads // nkv

    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, dht_prefix=None, **kwargs):
        if model_name_or_path and not os.path.isdir(str(model_name_or_path)) and not dht_prefix:
            dht_prefix = str(model_name_or_path).split("/")[-1].replace(".", "-")
            if not dht_prefix.endswith("-hf"):
                dht_prefix += "-hf"
            logger.info(f"Using DHT prefix: {{dht_prefix}}")
        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        config.use_cache = True
        if config.pad_token_id is None:
            config.pad_token_id = 0
        return result
""")

MODEL_TEMPLATE = textwrap.dedent("""\
from typing import Optional
import torch, torch.nn as nn
from hivemind import DHT
from hivemind.utils.logging import get_logger
from transformers.modeling_outputs import {output_class}

{base_model_import}

from petals.client.from_pretrained import FromPretrainedMixin
from petals.client.lm_head import LMHead
from petals.client.ptune import PTuneMixin
from petals.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from petals.client.remote_sequential import RemoteSequential
from petals.models.{pkg}.config import {config_class}
from petals.utils.auto_config import DefaultRevisionMixin
logger = get_logger(__name__)

if _BaseModel is not None:
    class {dist_model}(DefaultRevisionMixin, FromPretrainedMixin, PTuneMixin, _BaseModel):
        _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
        _keys_to_ignore_on_load_unexpected = [r"^model\\\\.layers\\\\."]
        config_class = {config_class}

        def __init__(self, config, *, dht=None):
            n, config.num_hidden_layers = config.num_hidden_layers, 0
            super().__init__(config)
            assert len(self.layers) == 0
            config.num_hidden_layers = n
            self.layers = RemoteSequential(config, dht=dht)
            self.requires_grad_(False)
            self.init_prompts(config)

        def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                    position_ids=None, head_mask=None, inputs_embeds=None, use_cache=None,
                    output_attentions=None, output_hidden_states=None,
                    return_dict=None, cache_position=None, **kwargs):
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("Cannot specify both input_ids and inputs_embeds")
            elif input_ids is not None:
                input_shape = input_ids.size(); input_ids = input_ids.view(-1, input_shape[-1])
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("Must specify input_ids or inputs_embeds")
            assert attention_mask is None or (attention_mask == 1).all()
            assert use_cache is None or use_cache
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            use_prompts = self.config.tuning_mode and "ptune" in self.config.tuning_mode and self.h.position == 0
            if use_prompts:
                prompts, ip = self.get_prompt(inputs_embeds.shape[0])
                inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
            else:
                prompts = ip = None
            hs = inputs_embeds; out_shape = input_shape + (hs.size(-1),)
            if past_key_values is None:
                past_key_values = RemotePastKeyValues()
            hs = self.layers(hs, prompts=ip, hypo_ids=past_key_values.hypo_ids if past_key_values else None)
            if use_prompts:
                hs = hs[:, self.pre_seq_len:]
            hs = self.norm(hs).view(out_shape)
            return {output_class}(last_hidden_state=hs, past_key_values=past_key_values, hidden_states=None, attentions=None)

        @property
        def word_embeddings(self): return self.embed_tokens
        @property
        def word_embeddings_layernorm(self): return nn.Identity()
        @property
        def h(self): return self.layers
        @property
        def ln_f(self): return self.norm

    class {dist_causal_lm}(FromPretrainedMixin, RemoteGenerationMixin, _BaseCausalLM):
        _keys_to_ignore_on_load_missing = {dist_model}._keys_to_ignore_on_load_missing
        _keys_to_ignore_on_load_unexpected = {dist_model}._keys_to_ignore_on_load_unexpected
        config_class = {config_class}

        def __init__(self, config):
            _BasePreTrained.__init__(self, config)
            self.model = {dist_model}(config)
            self.lm_head = LMHead(config)
            self.post_init()

        def get_output_embeddings(self): return self.lm_head

        @property
        def transformer(self): return self.model
else:
    class {dist_model}:
        def __init__(self, *a, **kw): raise ImportError("{model_name} requires a newer transformers")
    class {dist_causal_lm}:
        def __init__(self, *a, **kw): raise ImportError("{model_name} requires a newer transformers")
""")

INIT_TEMPLATE = textwrap.dedent("""\
from petals.models.{pkg}.block import {block_class}
from petals.models.{pkg}.config import {config_class}
from petals.models.{pkg}.model import {dist_causal_lm}, {dist_model}
from petals.utils.auto_config import register_model_classes
register_model_classes(
    config={config_class},
    model={dist_model},
    model_for_causal_lm={dist_causal_lm},
)
""")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS = [
    {
        "pkg": "qwen3",
        "model_name": "Qwen3",
        "block_class": "WrappedQwen3Block",
        "config_class": "DistributedQwen3Config",
        "dist_model": "DistributedQwen3Model",
        "dist_causal_lm": "DistributedQwen3ForCausalLM",
        "output_class": "BaseModelOutputWithPast",
        "decoder_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer as _DecoderLayer
            except ImportError:
                _DecoderLayer = None"""),
        "base_config_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3 import Qwen3Config as _BaseConfig
            except ImportError:
                _BaseConfig = None
            if _BaseConfig is None:
                raise ImportError("Qwen3 requires transformers >= 5.0.0")"""),
        "attn_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention as _AttnClass
            except ImportError:
                _AttnClass = None"""),
        "base_model_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3 import Qwen3Model as _BaseModel, Qwen3ForCausalLM as _BaseCausalLM, Qwen3PreTrainedModel as _BasePreTrained
            except ImportError:
                _BaseModel = _BaseCausalLM = _BasePreTrained = None"""),
    },
    {
        "pkg": "qwen3_5_moe",
        "model_name": "Qwen3.5 MoE",
        "block_class": "WrappedQwen3_5MoeBlock",
        "config_class": "DistributedQwen3_5MoeConfig",
        "dist_model": "DistributedQwen3_5MoeModel",
        "dist_causal_lm": "DistributedQwen3_5MoeForCausalLM",
        "output_class": "MoeModelOutputWithPast",
        "decoder_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextDecoderLayer as _DecoderLayer
            except ImportError:
                try:
                    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer as _DecoderLayer
                except ImportError:
                    _DecoderLayer = None"""),
        "base_config_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig as _BaseConfig
            except ImportError:
                from transformers.models.qwen3_5_moe import Qwen3_5MoeConfig as _BaseConfig"""),
        "attn_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextAttention as _AttnClass
            except ImportError:
                _AttnClass = None"""),
        "base_model_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3_5_moe import Qwen3_5MoeTextModel as _BaseModel, Qwen3_5MoeForCausalLM as _BaseCausalLM, Qwen3_5MoePreTrainedModel as _BasePreTrained
            except ImportError:
                _BaseModel = _BaseCausalLM = _BasePreTrained = None"""),
    },
    {
        "pkg": "glm_moe_dsa",
        "model_name": "GLM-5",
        "block_class": "WrappedGlmMoeDsaBlock",
        "config_class": "DistributedGlmMoeDsaConfig",
        "dist_model": "DistributedGlmMoeDsaModel",
        "dist_causal_lm": "DistributedGlmMoeDsaForCausalLM",
        "output_class": "BaseModelOutputWithPast",
        "decoder_import": textwrap.dedent("""\
            try:
                from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer as _DecoderLayer
            except ImportError:
                _DecoderLayer = None"""),
        "base_config_import": textwrap.dedent("""\
            try:
                from transformers.models.glm_moe_dsa import GlmMoeDsaConfig as _BaseConfig
            except ImportError:
                _BaseConfig = None
            if _BaseConfig is None:
                raise ImportError("GLM-5 requires transformers >= 5.2.0")"""),
        "attn_import": textwrap.dedent("""\
            try:
                from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention as _AttnClass
            except ImportError:
                _AttnClass = None"""),
        "base_model_import": textwrap.dedent("""\
            try:
                from transformers.models.glm_moe_dsa import GlmMoeDsaModel as _BaseModel, GlmMoeDsaForCausalLM as _BaseCausalLM, GlmMoeDsaPreTrainedModel as _BasePreTrained
            except ImportError:
                _BaseModel = _BaseCausalLM = _BasePreTrained = None"""),
    },
]


def patch_model(models_dir: pathlib.Path, spec: dict):
    pkg = spec["pkg"]
    pkg_dir = models_dir / pkg
    pkg_dir.mkdir(exist_ok=True)

    (pkg_dir / "__init__.py").write_text(INIT_TEMPLATE.format(**spec))
    (pkg_dir / "block.py").write_text(BLOCK_TEMPLATE.format(**spec))
    (pkg_dir / "config.py").write_text(CONFIG_TEMPLATE.format(**spec))
    (pkg_dir / "model.py").write_text(MODEL_TEMPLATE.format(**spec))
    print(f"  {spec['model_name']} ({pkg}) overlay written")


def patch_auto_config(utils_dir: pathlib.Path):
    """Add trust_remote_code and model_type aliases to auto_config.py."""
    auto_config_path = utils_dir / "auto_config.py"
    if not auto_config_path.exists():
        print("  auto_config.py not found; skipping")
        return

    text = auto_config_path.read_text()

    # Add _MODEL_TYPE_ALIASES dict definition if not present
    ALIASES_DEF = '_MODEL_TYPE_ALIASES = {"kimi_k25": "kimi_k2"}'
    if ALIASES_DEF not in text:
        # Insert before _CLASS_MAPPING = {} (the actual petals code uses {}, not ()
        text = text.replace(
            "_CLASS_MAPPING = {}",
            f'{ALIASES_DEF}\n\n_CLASS_MAPPING = {{}}',
            1,
        )

    # Add trust_remote_code=True to AutoConfig.from_pretrained call
    if "trust_remote_code=True" not in text:
        text = text.replace(
            "config = AutoConfig.from_pretrained(model_name_or_path, *args, **kwargs)",
            "config = AutoConfig.from_pretrained(model_name_or_path, *args, trust_remote_code=True, **kwargs)",
        )

    # Add alias resolution
    if "_MODEL_TYPE_ALIASES.get" not in text:
        text = text.replace(
            'if config.model_type not in _CLASS_MAPPING:',
            'model_type = _MODEL_TYPE_ALIASES.get(config.model_type, config.model_type)\n        if model_type not in _CLASS_MAPPING:',
        )
        text = text.replace(
            "proper_cls = getattr(_CLASS_MAPPING[config.model_type], cls._mapping_field)",
            "proper_cls = getattr(_CLASS_MAPPING[model_type], cls._mapping_field)",
        )

    # Make register_model_classes more lenient (skip duplicates instead of asserting)
    if "is already registered" in text and "logger.warning" not in text:
        text = text.replace(
            '    assert (\n        config.model_type not in _CLASS_MAPPING\n    ), f"Model type {config.model_type} is already registered"\n\n    _CLASS_MAPPING[config.model_type]',
            '    if config.model_type in _CLASS_MAPPING:\n        return  # already registered\n    _CLASS_MAPPING[config.model_type]',
        )

    auto_config_path.write_text(text)
    print("  auto_config.py patched (trust_remote_code, aliases, lenient registration)")


def patch_models_init(models_dir: pathlib.Path, model_pkgs: list):
    """Update models/__init__.py to import new model packages."""
    init_path = models_dir / "__init__.py"
    text = init_path.read_text()
    changed = False
    for pkg in model_pkgs:
        marker = f"from petals.models.{pkg} import *"
        if marker not in text:
            text += f"\ntry:\n    from petals.models.{pkg} import *\nexcept ImportError:\n    pass\n"
            changed = True
    if changed:
        init_path.write_text(text)
        print("  models/__init__.py updated")
    else:
        print("  models/__init__.py already up to date")


def patch_convert_block(utils_dir: pathlib.Path):
    """Patch convert_block.py to handle models without num_heads attribute (e.g. Qwen3)."""
    cb_path = utils_dir / "convert_block.py"
    if not cb_path.exists():
        print("  convert_block.py not found; skipping")
        return

    text = cb_path.read_text()
    old = "total_heads += submodule.num_heads"
    new = (
        "total_heads += getattr(submodule, 'num_heads', None) "
        "or getattr(submodule, 'num_attention_heads', model_config.num_attention_heads)"
    )
    if old in text:
        cb_path.write_text(text.replace(old, new, 1))
        print("  convert_block.py patched (num_heads fallback)")
    else:
        print("  convert_block.py already patched or has different format")


def patch_hivemind_p2pd(site_pkgs: pathlib.Path):
    """Fix Unix socket multiaddr path encoding bug in hivemind p2p_daemon.py.

    Root cause: Python's Multiaddr('/unix/tmp/foo.sock').to_bytes() encodes the path
    WITHOUT the leading '/', producing bytes for 'tmp/foo.sock' (relative path).
    Go's manet.Dial then tries to connect to this relative path, fails silently, and
    the stream handler never fires — causing all inference requests to hang.

    Fix: construct multiaddr bytes manually so the path includes the leading '/'.
    """
    p2pd_path = site_pkgs / "hivemind" / "p2p" / "p2p_daemon.py"
    if not p2pd_path.exists():
        print("  hivemind p2p_daemon.py not found; skipping")
        return

    text = p2pd_path.read_text()

    # Already patched?
    if "_make_abs_unix_maddr" in text:
        print("  hivemind p2p_daemon.py already patched (Unix socket path fix)")
        return

    # Insert the helper function before the P2PContext dataclass decorator
    helper = (
        "\n\ndef _make_abs_unix_maddr(socket_path: str):\n"
        '    """Create a Unix socket multiaddr with absolute path correctly encoded in bytes.\n\n'
        "    The standard Multiaddr('/unix/path') strips the leading '/' from the path bytes,\n"
        "    causing Go's manet.Dial to try to connect to a relative path (fails silently).\n"
        "    This function encodes the path WITH the leading '/' so Go can find the socket.\n"
        '    """\n'
        "    p_unix_varint = bytes([0x90, 0x03])  # P_UNIX = 400\n"
        "    path_bytes = socket_path.encode()\n"
        "    length = len(path_bytes)\n"
        "    if length < 0x80:\n"
        "        len_bytes = bytes([length])\n"
        "    else:\n"
        "        result = []\n"
        "        while length > 0x7f:\n"
        "            result.append((length & 0x7f) | 0x80)\n"
        "            length >>= 7\n"
        "        result.append(length)\n"
        "        len_bytes = bytes(result)\n"
        "    from multiaddr import Multiaddr as _Multiaddr\n"
        "    return _Multiaddr(p_unix_varint + len_bytes + path_bytes)\n"
    )

    insert_marker = "@dataclass(frozen=True)"
    if insert_marker not in text:
        print("  hivemind p2p_daemon.py: expected marker not found; skipping")
        return

    text = text.replace(insert_marker, helper + "\n\n" + insert_marker, 1)

    # Replace both occurrences of _client_listen_maddr using the old (broken) Multiaddr call.
    # Pattern 1: in P2P.create()
    old1 = 'self._client_listen_maddr = Multiaddr(cls._UNIX_SOCKET_PREFIX + f"p2pclient-{socket_uid}.sock")'
    new1 = 'self._client_listen_maddr = _make_abs_unix_maddr(f"/tmp/hivemind-p2pclient-{socket_uid}.sock")'
    if old1 in text:
        text = text.replace(old1, new1, 1)
    else:
        print("  WARNING: first _client_listen_maddr pattern not found; skipping that replacement")

    # Pattern 2: in P2P.replicate()
    old2 = 'self._client_listen_maddr = Multiaddr(cls._UNIX_SOCKET_PREFIX + f"p2pclient-{socket_uid}.sock")'
    new2 = 'self._client_listen_maddr = _make_abs_unix_maddr(f"/tmp/hivemind-p2pclient-{socket_uid}.sock")'
    # (same string; replace any remaining instance)
    if old2 in text:
        text = text.replace(old2, new2, 1)

    p2pd_path.write_text(text)
    print("  hivemind p2p_daemon.py patched (Unix socket absolute path fix)")


def patch_hivemind_grad_scaler(site_pkgs: pathlib.Path):
    """Patch hivemind grad_scaler.py for torch>=2.5 compatibility."""
    gs_path = site_pkgs / "hivemind" / "optim" / "grad_scaler.py"
    if not gs_path.exists():
        print("  hivemind grad_scaler.py not found; skipping")
        return

    text = gs_path.read_text()
    old_import = "from torch.cuda.amp.grad_scaler import OptState, _refresh_per_optimizer_state"
    new_import = (
        "try:\n"
        "    from torch.cuda.amp.grad_scaler import OptState, _refresh_per_optimizer_state\n"
        "except ImportError:\n"
        "    import enum\n"
        "    class OptState(enum.Enum):\n"
        "        READY = 0; UNSCALED = 1; STEPPED = 2\n"
        "    def _refresh_per_optimizer_state():\n"
        "        return {'stage': OptState.READY, 'found_inf_per_device': {}}"
    )
    if old_import in text:
        gs_path.write_text(text.replace(old_import, new_import, 1))
        print("  hivemind grad_scaler.py patched (torch>=2.5 compat)")
    elif "except ImportError" in text and "OptState" in text:
        print("  hivemind grad_scaler.py already patched")
    else:
        print("  hivemind grad_scaler.py: unexpected format, skipping")


def main():
    print("Patching petals with frontier model support...")
    patch_auto_config(utils_dir)
    patch_convert_block(utils_dir)
    patch_hivemind_grad_scaler(site_pkgs)
    patch_hivemind_p2pd(site_pkgs)

    for spec in MODELS:
        patch_model(models_dir, spec)

    # Kimi K2 is special — not in standard transformers, needs trust_remote_code.
    # We skip the overlay for now as it requires the model repo's custom code.
    # The auto_config alias (kimi_k25 -> kimi_k2) is already patched above.

    patch_models_init(models_dir, [m["pkg"] for m in MODELS])
    print("Done.")


if __name__ == "__main__":
    main()
