"""Aggregate the JMCAI MiniMax H3 node modules.

Each source module is loaded independently so that a compatibility failure in
one optional node does not prevent the remaining H3 nodes from registering.
"""

import importlib
import traceback


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_MODULE_NAMES = (
    "h3_dynamic_cfg_scheduler",
    "h3_sigma_refiner",
    "H3DistanceAttentionPatcher",
    "minimax_i2v_tail",
    "h3_prompt_relay",
    "h3_tiled_sampler",
    "jmcai_h3_cond_noise_aug",
)

_PACKAGE = __package__ or __name__.rpartition(".")[0]

for _module_name in _MODULE_NAMES:
    try:
        _module = importlib.import_module(
            f".minimax_h3.{_module_name}",
            package=_PACKAGE,
        )
        NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(
            getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {})
        )
    except Exception as exc:
        print(
            f"\033[31m[JMCAI Error]\033[0m 无法加载 MiniMax H3 模块 "
            f"{_module_name}: {exc}"
        )
        traceback.print_exc()


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
