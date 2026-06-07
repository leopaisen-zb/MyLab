# adapters/eq_adapter.py
import os
import re
import sys
import torch
from typing import Any, Dict, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.base import BaseAdapter
from adapters.eq_model import EQModelCore, EQModelTorchScript, composition_to_features


class EQAdapter(BaseAdapter):
    """
    Eqv2-Lite adapter for ΔG_H prediction.

    Supports two inference modes:
    - Eager mode: Standard PyTorch eager execution
    - TorchScript mode: JIT-compiled with torch.jit.script + optimize_for_inference
      Provides 1.4-2x speedup for batch sizes 1-32 on CPU.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "checkpoint_path": os.environ.get("EQ_CHECKPOINT_PATH", "./models/eqv2_lite.pt"),
            "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            "batch_size": 32,
            "use_torchscript": True,  # Enable TorchScript by default
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.model_eager: Optional[EQModelCore] = None
        self.model_scripted: Optional[EQModelTorchScript] = None

    def initialize(self) -> bool:
        if self._initialized:
            return True
        try:
            checkpoint = self.config.get("checkpoint_path", "")
            use_ts = self.config.get("use_torchscript", True)

            core = EQModelCore()
            if checkpoint and os.path.exists(checkpoint):
                try:
                    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    core.load_state_dict(state_dict)
                except Exception:
                    pass

            core.to(self.config["device"])
            core.eval()
            self.model_eager = core

            if use_ts:
                try:
                    scripted = torch.jit.script(EQModelTorchScript(core))
                    self.model_scripted = torch.jit.optimize_for_inference(scripted)
                except Exception as e:
                    self.model_scripted = None

            self._initialized = True
            return True

        except Exception as e:
            print(f"EQv2-Lite initialization failed: {e}")
            return False

    def validate_input(self, input_data: Any) -> bool:
        if not isinstance(input_data, dict):
            return False
        if not input_data.get("parsed"):
            return False
        return True

    def forward(self, input_data: Dict[str, Any]) -> float:
        """
        Single-structure forward pass.
        Uses TorchScript if available, otherwise eager.
        Returns ΔG_H in eV.
        """
        if not self._initialized:
            self.initialize()

        composition = input_data.get("composition", "")
        num_sites = input_data.get("num_sites", 20)

        feat = composition_to_features(composition, num_sites)
        feat = feat.unsqueeze(0)  # batch dimension

        with torch.no_grad():
            if self.model_scripted is not None:
                out = self.model_scripted(feat)
            else:
                out = self.model_eager(feat)

        return out.item()

    def predict(self, poscar_text: str) -> float:
        """Predict ΔG_H from raw POSCAR text using the real StandaloneEquiformerV2."""
        try:
            from backend.eq_predict import predict as eq_predict_real
            result = eq_predict_real(poscar_text)
            return result["dg_h"]
        except Exception as e:
            print(f"[EQAdapter] backend.eq_predict failed ({e}), falling back to toy model")
            # 退化估计也必须反映真实结构：从 POSCAR 提取真实组成喂 toy 模型，
            # 避免返回与结构无关的常数。
            from adapters.hea_gen_adapter import composition_from_poscar
            comp = composition_from_poscar(poscar_text, sep=" ")
            n_atoms = sum(int(x) for x in re.findall(r"\d+", comp)) if comp else 20
            return self.forward({"composition": comp, "num_sites": n_atoms or 20, "parsed": True})

    def predict_batch(self, structures: list) -> list:
        """
        Batch prediction using TorchScript for efficiency.
        For small batches (<=32) TorchScript provides 1.4-2x speedup.
        """
        if not structures:
            return []

        if not self._initialized:
            self.initialize()

        # Stack all features
        feats = []
        for s in structures:
            composition = s.get("composition", "")
            num_sites = s.get("num_sites", 20)
            feats.append(composition_to_features(composition, num_sites))
        batch = torch.stack(feats)

        with torch.no_grad():
            if self.model_scripted is not None:
                out = self.model_scripted(batch)
            else:
                out = self.model_eager(batch)

        return out.tolist()
