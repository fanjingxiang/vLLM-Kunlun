"""vllm kunlun init"""

import builtins
import importlib
import logging
import os
import sys

from vllm.logger import init_logger as init_vllm_logger

OLD_IMPORT_HOOK = builtins.__import__


def _apply_mamba_spec_patch(mamba_abstract_module) -> None:
    """Patch MambaBase.get_kv_cache_spec to allow qwen3_5_moe with MTP.

    Upstream vLLM only whitelists "qwen3_next" for mamba + speculative decoding.
    Imports are kept local to avoid triggering a circular import of vllm.config
    when this runs during plugin registration.
    """
    from vllm.v1.kv_cache_interface import MambaSpec

    _ALLOWED = {"qwen3_next", "qwen3_5_moe"}

    def _patched_get_kv_cache_spec(self, vllm_config):
        if (
            vllm_config.speculative_config is not None
            and vllm_config.model_config.hf_config.model_type not in _ALLOWED
        ):
            raise NotImplementedError(
                "Mamba with speculative decoding is not supported yet."
            )
        cache_config = vllm_config.cache_config
        return MambaSpec(
            shapes=self.get_state_shape(),
            dtypes=self.get_state_dtype(),
            block_size=cache_config.mamba_block_size,
            page_size_padded=cache_config.mamba_page_size_padded,
            mamba_type=self.mamba_type,
            mamba_cache_mode=cache_config.mamba_cache_mode,
            num_speculative_blocks=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config is not None
                else 0
            ),
        )

    mamba_abstract_module.MambaBase.get_kv_cache_spec = _patched_get_kv_cache_spec


def _configure_kunlun_logger() -> logging.Logger:
    """Reuse vLLM's handler for the vllm_kunlun logger tree."""
    vllm_logger = init_vllm_logger("vllm")
    kunlun_logger = logging.getLogger("vllm_kunlun")

    if not kunlun_logger.handlers:
        for handler in vllm_logger.handlers:
            kunlun_logger.addHandler(handler)

    kunlun_logger.setLevel(vllm_logger.getEffectiveLevel())
    kunlun_logger.propagate = False
    return kunlun_logger


def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):
    try:
        module_mappings = {
            "vllm.compilation.wrapper": "vllm_kunlun.compilation.wrapper",
            "vllm.model_executor.model_loader.bitsandbytes_loader": "vllm_kunlun.models.model_loader.bitsandbytes_loader",
            "vllm.v1.sample.ops.topk_topp_sampler": "vllm_kunlun.v1.sample.ops.topk_topp_sampler",
            "vllm.v1.sample.rejection_sampler": "vllm_kunlun.v1.sample.rejection_sampler",
            "vllm.attention.ops.merge_attn_states": "vllm_kunlun.ops.attention.merge_attn_states",
            "vllm.v1.attention.backends.gdn_attn": "vllm_kunlun.v1.attention.backends.gdn_attn",
            "vllm.model_executor.models.config": "vllm_kunlun.models.config",
        }

        # Install mapping into sys.modules on first import. Do NOT early-return
        # here: returning sys.modules[leaf] bypasses Python's normal __import__
        # semantics (which, for `import a.b.c`, should return the top-level
        # package `a`, not the leaf). Let OLD_IMPORT_HOOK do the right thing.
        if module_name in module_mappings and module_name not in sys.modules:
            target_module = module_mappings[module_name]
            mapped = importlib.import_module(target_module)
            sys.modules[module_name] = mapped
            sys.modules[target_module] = mapped
    except Exception:
        pass

    module = OLD_IMPORT_HOOK(
        module_name, globals=globals, locals=locals, fromlist=fromlist, level=level
    )

    # Lazy patch: only apply once vllm has finished importing the upstream
    # rejection_sampler module. Running apply() during register() causes a
    # circular import (vllm.config is still partially initialized at that
    # point). Guard against recursion since apply() itself imports the module.
    if (
        module_name == "vllm.v1.sample.rejection_sampler"
        and not getattr(_custom_import, "_patch_in_progress", False)
    ):
        _custom_import._patch_in_progress = True
        try:
            from .v1.sample import rejection_sampler_patch

            rejection_sampler_patch.apply()
        except Exception:
            logging.getLogger("vllm_kunlun").exception(
                "[KunlunPlugin] lazy rejection_sampler_patch.apply() failed"
            )
        finally:
            _custom_import._patch_in_progress = False

    # Lazy patch for bind_kv_cache. Same reason as rejection_sampler above:
    # importing vllm.v1.worker.utils during register() triggers a circular
    # import via vllm.config. Instead, wait until the module actually finishes
    # loading and then swap in the Kunlun-aware implementation on both the
    # source module and any consumer module that already bound the upstream
    # symbol into its namespace.
    if (
        module_name == "vllm.v1.worker.utils"
        and not getattr(_custom_import, "_bind_kv_cache_patched", False)
    ):
        _custom_import._bind_kv_cache_patched = True
        try:
            from vllm_kunlun.v1.worker.bind_kv_cache_patch import (
                bind_kv_cache as _kunlun_bind_kv_cache,
            )

            module.bind_kv_cache = _kunlun_bind_kv_cache
            for _mod_name in ("vllm.v1.worker.gpu_model_runner",):
                _mod = sys.modules.get(_mod_name)
                if _mod is not None and hasattr(_mod, "bind_kv_cache"):
                    _mod.bind_kv_cache = _kunlun_bind_kv_cache
            logging.getLogger("vllm_kunlun").info(
                "[KunlunPlugin] lazy-patched bind_kv_cache for OOT platform"
            )
        except Exception:
            _custom_import._bind_kv_cache_patched = False
            logging.getLogger("vllm_kunlun").exception(
                "[KunlunPlugin] lazy bind_kv_cache patch failed"
            )

    # Lazy patch for MambaBase.get_kv_cache_spec. Same circular-import reason
    # as bind_kv_cache above: importing vllm.model_executor.layers.mamba.abstract
    # during register() triggers `from vllm.config import VllmConfig` while
    # vllm.config is still initializing.
    if (
        module_name == "vllm.model_executor.layers.mamba.abstract"
        and not getattr(_custom_import, "_mamba_spec_patched", False)
    ):
        _custom_import._mamba_spec_patched = True
        try:
            _apply_mamba_spec_patch(module)
            logging.getLogger("vllm_kunlun").info(
                "[KunlunPlugin] lazy-patched MambaBase.get_kv_cache_spec"
            )
        except Exception:
            _custom_import._mamba_spec_patched = False
            logging.getLogger("vllm_kunlun").exception(
                "[KunlunPlugin] lazy MambaBase.get_kv_cache_spec patch failed"
            )

    return module


def import_hook():
    """Apply import hook for VLLM Kunlun"""
    builtins.__import__ = _custom_import


def register():
    """Register the Kunlun platform"""

    logger = _configure_kunlun_logger()
    logger.info("[KunlunPlugin] register() pid=%s", os.getpid())

    # --- load native extension to register torch.ops._C.weak_ref_tensor ---
    try:
        from . import _kunlun  # noqa: F401

        logger.info("[KunlunPlugin] _kunlun native extension loaded")
    except ImportError as e:
        logger.warning("[KunlunPlugin] Failed to load _kunlun: %s", e)

    # --- import wrapper & patch utils ---
    try:
        from .schema import direct_register_custom_op  # noqa: F401
        from .schema import patch_annotations_for_schema  # noqa: F401

        logger.info("[KunlunPlugin] vllm_utils_wrapper loaded and patched")
    except Exception:
        logger.exception("[KunlunPlugin] wrapper import/patch failed")
        raise

    # TODO @xyDong0223 Fix Hear, import failed in v15.1
    # --- optional GLM5 config patch ---
    # if "vllm.transformers_utils.config" in sys.modules:
    #     from .transformer_utils.config import _XPU_CONFIG_REGISTRY
    #     sys.modules["vllm.transformers_utils.config"]._CONFIG_REGISTRY = _XPU_CONFIG_REGISTRY
    #     logger.info("[KunlunPlugin] patched transformers_utils.config")

    # --- patch ModelConfig ---
    # try:
    #     import vllm.config.model as model_module
    #     from .config.model import is_deepseek_mla
    #     model_module.ModelConfig.is_deepseek_mla = property(is_deepseek_mla)
    #     logger.info("[KunlunPlugin] patched ModelConfig.is_deepseek_mla")
    # except Exception:
    #     logger.exception("[KunlunPlugin] ModelConfig patch failed")
    #     raise

    # --- patch MambaBase.get_kv_cache_spec to allow qwen3_5_moe with
    # speculative decoding (MTP). Upstream vLLM only whitelists "qwen3_next".
    # The actual patch happens lazily inside _custom_import (see below) when
    # vllm.model_executor.layers.mamba.abstract finishes loading. Importing it
    # here triggers a circular import via vllm.config. If it was already loaded
    # before the plugin registered, apply the patch now.
    if "vllm.model_executor.layers.mamba.abstract" in sys.modules and not getattr(
        _custom_import, "_mamba_spec_patched", False
    ):
        _custom_import._mamba_spec_patched = True
        try:
            _apply_mamba_spec_patch(
                sys.modules["vllm.model_executor.layers.mamba.abstract"]
            )
            logger.info(
                "[KunlunPlugin] patched MambaBase.get_kv_cache_spec to allow qwen3_5_moe"
            )
        except Exception:
            _custom_import._mamba_spec_patched = False
            logger.exception("[KunlunPlugin] MambaBase.get_kv_cache_spec patch failed")

    # --- import hook ---
    try:
        import_hook()
        logger.info("[KunlunPlugin] import_hook() ok")
    except Exception:
        logger.exception("[KunlunPlugin] import_hook() failed")
        raise

    # --- patch bind_kv_cache ---
    # The actual monkey-patch happens lazily inside _custom_import when
    # vllm.v1.worker.utils finishes importing. Doing it here would trigger a
    # circular import of vllm.config (plugin register() runs while vllm.config
    # is still initializing). If the module was already imported before the
    # plugin loaded, apply the patch immediately.
    if "vllm.v1.worker.utils" in sys.modules and not getattr(
        _custom_import, "_bind_kv_cache_patched", False
    ):
        _custom_import._bind_kv_cache_patched = True
        try:
            _vllm_worker_utils = sys.modules["vllm.v1.worker.utils"]
            from vllm_kunlun.v1.worker.bind_kv_cache_patch import (
                bind_kv_cache as _kunlun_bind_kv_cache,
            )

            _vllm_worker_utils.bind_kv_cache = _kunlun_bind_kv_cache
            for _mod_name in ("vllm.v1.worker.gpu_model_runner",):
                _mod = sys.modules.get(_mod_name)
                if _mod is not None and hasattr(_mod, "bind_kv_cache"):
                    _mod.bind_kv_cache = _kunlun_bind_kv_cache
            logger.info(
                "[KunlunPlugin] eagerly patched bind_kv_cache (already loaded)"
            )
        except Exception:
            _custom_import._bind_kv_cache_patched = False
            logger.exception("[KunlunPlugin] eager bind_kv_cache patch failed")

    # NOTE: rejection_sampler_patch is now applied lazily from _custom_import
    # when vllm.v1.sample.rejection_sampler finishes importing. Applying it
    # here would trigger a circular import of vllm.config because plugin
    # register() runs while vllm.config.__init__ is still executing.

    # --- register reasoning parser override (lazy, to avoid circular import) ---
    try:
        from vllm.reasoning import ReasoningParserManager

        # Override the lazy registration path with our custom parser.
        # This happens before vllm's default lazy registration (which is
        # triggered when vllm.reasoning module is imported), so our path
        # takes precedence.
        # Custom parser for Qwen3.5 support
        ReasoningParserManager.register_lazy_module(
            name="qwen3",
            module_path="vllm_kunlun.reasoning.qwen3_reasoning_parser",
            class_name="Qwen3ReasoningParser",
        )
        logger.info("[KunlunPlugin] registered Qwen3ReasoningParser override (lazy)")
    except Exception:
        logger.exception("[KunlunPlugin] Qwen3ReasoningParser registration failed")
        # Non-fatal: continue without the override

    logger.info("[KunlunPlugin] register() done")
    return "vllm_kunlun.platforms.kunlun.KunlunPlatform"


def register_model():
    """Register models for training and inference"""
    from .models import register_model as _reg

    _reg()
