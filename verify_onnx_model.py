"""Verify the local ONNX LSTM model and scaler package without starting Streamlit."""
from pathlib import Path
import pickle
import sys

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "trained_models"


def first_file(pattern: str) -> Path:
    files = sorted(MODEL_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} file found in {MODEL_DIR}")
    return files[0]


def main() -> int:
    model_path = first_file("*.onnx")
    package_path = first_file("*.pkl")

    print("Mekong Salinity ONNX model check")
    print("=" * 42)
    print("Python:", sys.executable)
    print("ONNX model:", model_path.name)
    print("Scaler package:", package_path.name)

    with package_path.open("rb") as handle:
        package = pickle.load(handle)
    missing = [key for key in ("scaler_dyn", "scaler_stat") if key not in package]
    if missing:
        raise KeyError("Missing .pkl keys: " + ", ".join(missing))

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print("\nInputs:")
    for node in inputs:
        print(f" - {node.name}: {node.shape} ({node.type})")
    print("Outputs:")
    for node in outputs:
        print(f" - {node.name}: {node.shape} ({node.type})")

    dynamic = next((node for node in inputs if len(node.shape) == 3), None)
    static = next((node for node in inputs if len(node.shape) == 2), None)
    if dynamic is None or static is None:
        raise ValueError("Expected dynamic rank-3 and static rank-2 ONNX inputs.")

    # This only checks runtime wiring. Real inference in the app uses scaled data.
    dynamic_zeros = np.zeros((1, 6, 10), dtype=np.float32)
    static_zeros = np.zeros((1, 10), dtype=np.float32)
    prediction = session.run(None, {dynamic.name: dynamic_zeros, static.name: static_zeros})[0]
    print("\nTest prediction shape:", np.asarray(prediction).shape)
    print("SUCCESS: ONNX Runtime can load and execute the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
