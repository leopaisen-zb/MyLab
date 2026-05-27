# adapters/eq_model.py
"""
Eqv2-Lite Demo Model for ΔG_H prediction.
Implements a simple composition-based ML model that supports TorchScript optimization.
"""
import os
import torch
import torch.nn as nn
import time
from typing import List, Optional


# Element metadata for HEA catalysts (Ir, Pd, Pt, Rh, Ru)
ELEMENT_LIST = ["Ir", "Pd", "Pt", "Rh", "Ru", "Fe", "Co", "Ni", "Cu", "Zn"]
ELEMENT_NUM = len(ELEMENT_LIST)
ELEMENT_TO_IDX = {el: i for i, el in enumerate(ELEMENT_LIST)}

# Physical constants for feature engineering
ELECTRONEGATIVITY = {"Ir": 2.2, "Pd": 2.2, "Pt": 2.2, "Rh": 2.2, "Ru": 2.2,
                     "Fe": 1.8, "Co": 1.8, "Ni": 1.8, "Cu": 1.9, "Zn": 1.6}
ATOMIC_RADIUS = {"Ir": 1.36, "Pd": 1.38, "Pt": 1.38, "Rh": 1.34, "Ru": 1.34,
                 "Fe": 1.24, "Co": 1.25, "Ni": 1.24, "Cu": 1.32, "Zn": 1.34}


def composition_to_features(composition: str, num_sites: int = 20) -> torch.Tensor:
    """
    Parse composition string (e.g. 'Ir2 Pd2 Pt2') into feature tensor.
    Returns: tensor of shape (ELEMENT_NUM + 3,) = [element_counts, avg_electronegativity, avg_radius, total_sites]
    """
    feat = torch.zeros(ELEMENT_NUM + 3)
    if not composition:
        return feat

    parts = composition.replace(",", " ").split()
    total_atoms = 0
    for part in parts:
        el = "".join(c for c in part if c.isalpha())
        count_str = "".join(c for c in part if c.isdigit())
        count = int(count_str) if count_str else 1
        if el in ELEMENT_TO_IDX:
            feat[ELEMENT_TO_IDX[el]] = count
            total_atoms += count

    feat[ELEMENT_NUM] = total_atoms / max(num_sites, 1)

    # Average electronegativity weighted by composition
    total_en = sum(feat[i] * ELECTRONEGATIVITY.get(ELEMENT_LIST[i], 0) for i in range(ELEMENT_NUM))
    feat[ELEMENT_NUM + 1] = total_en / max(total_atoms, 1)

    # Average atomic radius
    total_radius = sum(feat[i] * ATOMIC_RADIUS.get(ELEMENT_LIST[i], 1.3) for i in range(ELEMENT_NUM))
    feat[ELEMENT_NUM + 2] = total_radius / max(total_atoms, 1)

    return feat


class EQModelCore(nn.Module):
    """
    Core prediction model: composition features → ΔG_H.
    Architecture: Linear → ReLU → Linear → Dropout → Linear → output
    """
    def __init__(self, input_dim: int = ELEMENT_NUM + 3, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Initialize final bias to physically reasonable ΔG_H ≈ -0.5 eV (hydrogen adsorption)
        with torch.no_grad():
            self.net[-1].bias.fill_(-0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class EQModelTorchScript(nn.Module):
    """
    TorchScript-wrapped model for optimized inference.
    Uses tracing to capture the computational graph.
    """
    def __init__(self, core: EQModelCore):
        super().__init__()
        self.core = core
        self.eval()  # Important: must be in eval mode for TorchScript

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (batch, ELEMENT_NUM+3) or (ELEMENT_NUM+3,)
        Returns:
            ΔG_H predictions in eV
        """
        return self.core(features)


def create_demo_model(checkpoint_path: Optional[str] = None) -> tuple:
    """
    Create demo model. If checkpoint exists, load weights; otherwise use random init.
    Returns (eager_model, scripted_model).
    """
    core = EQModelCore()

    # Load checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            core.load_state_dict(state_dict)
        except Exception:
            pass

    core.eval()

    # Create TorchScript-wrapped version
    scripted = torch.jit.script(EQModelTorchScript(core))
    scripted = torch.jit.optimize_for_inference(scripted)

    return core, scripted


# Benchmark function
def benchmark_inference(
    model: nn.Module,
    scripted_model: Optional[torch.jit.ScriptModule],
    num_samples: int = 1000,
    num_runs: int = 5
) -> dict:
    """
    Benchmark eager vs TorchScript inference.
    Returns dict with timing statistics.
    """
    results = {}

    # Generate random valid inputs
    torch.manual_seed(42)
    dummy_input = torch.randn(num_samples, ELEMENT_NUM + 3)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input[:32])
        if scripted_model is not None:
            torch._dynamo.reset()
            for _ in range(10):
                _ = scripted_model(dummy_input[:32])

    # Eager mode benchmark
    eager_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            eager_times.append(time.perf_counter() - start)

    results["eager_mean_ms"] = (sum(eager_times) / len(eager_times)) * 1000
    results["eager_std_ms"] = (max(eager_times) - min(eager_times)) / 2 * 1000

    # TorchScript mode benchmark
    if scripted_model is not None:
        ts_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = scripted_model(dummy_input)
                ts_times.append(time.perf_counter() - start)

        results["torchscript_mean_ms"] = (sum(ts_times) / len(ts_times)) * 1000
        results["torchscript_std_ms"] = (max(ts_times) - min(ts_times)) / 2 * 1000
        results["speedup"] = results["eager_mean_ms"] / results["torchscript_mean_ms"]

    return results


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("Eqv2-Lite TorchScript Optimization Benchmark")
    print("=" * 60)

    core, scripted = create_demo_model()

    # Test forward pass
    test_feat = composition_to_features("Ir2 Pd2 Pt2 Rh2 Ru2", num_sites=10)
    test_feat = test_feat.unsqueeze(0)  # batch dimension

    eager_out = core(test_feat)
    ts_out = scripted(test_feat)
    print(f"\nForward pass test:")
    print(f"  Eager output:      {eager_out.item():.4f} eV")
    print(f"  TorchScript output: {ts_out.item():.4f} eV")
    print(f"  Match: {torch.allclose(eager_out, ts_out, atol=1e-6)}")

    # Benchmark
    print(f"\nBenchmarking with 1000 samples, 5 runs...")
    stats = benchmark_inference(core, scripted, num_samples=1000, num_runs=5)

    print(f"\n{'Metric':<25} {'Eager':>12} {'TorchScript':>12} {'Speedup':>10}")
    print("-" * 62)
    print(f"{'Mean latency (ms)':<25} {stats['eager_mean_ms']:>12.3f} {stats['torchscript_mean_ms']:>12.3f} {stats['speedup']:>10.2f}x")
    print(f"{'Std dev (ms)':<25} {stats['eager_std_ms']:>12.3f} {stats['torchscript_std_ms']:>12.3f}")
    print(f"\nThroughput:")
    print(f"  Eager:      {1000/stats['eager_mean_ms']:.1f} inferences/sec")
    print(f"  TorchScript: {1000/stats['torchscript_mean_ms']:.1f} inferences/sec")
