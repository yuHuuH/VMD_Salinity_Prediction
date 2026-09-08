"""Console setup check for the CSV-only ONNX salinity demo."""
from pathlib import Path

from salinity_pipeline import canonical_history, discover_inventory, read_csv_flexible, valid_year_table

ROOT = Path(__file__).resolve().parent


def main() -> int:
    inventory = discover_inventory(ROOT)
    print("\nMekong Salinity CSV-Only Demo — setup check")
    print("=" * 52)
    print(f"Project folder: {ROOT}")
    print(f"ONNX model(s): {len(inventory.onnx_models)}")
    print(f"Scaler package(s): {len(inventory.pkl_packages)}")
    print(f"Boundary file(s): {len(inventory.boundaries)}")
    print(f"Prepared Final CSV(s): {len(inventory.history_csvs)}")

    missing = []
    if not inventory.onnx_models:
        missing.append(".onnx model in trained_models/")
    if not inventory.pkl_packages:
        missing.append(".pkl scaler package in trained_models/")
    if not inventory.boundaries:
        missing.append("GADM boundary JSON in boundary/")
    if not inventory.history_csvs:
        missing.append("VMD_Salinity_Dataset_IDW_500_Final.csv in data/")
    if missing:
        print("\nMissing required inputs:")
        for item in missing:
            print(f" - {item}")
        return 1

    try:
        history = canonical_history(read_csv_flexible(inventory.history_csvs[0]))
        status = valid_year_table(history)
    except Exception as exc:
        print(f"\nCould not calculate selectable years: {exc}")
        return 1

    print("\nCSV year availability:")
    print(status[["target_year", "status", "csv_rows", "eligible_rows"]].to_string(index=False))
    valid = status.loc[status["status"] == "Valid", "target_year"].tolist()
    print(f"\nSelectable years: {valid if valid else 'none'}")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
