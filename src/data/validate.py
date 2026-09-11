"""Great Expectations validation for the PaySim data contract."""

from __future__ import annotations

import sys
from pathlib import Path

import great_expectations as gx
import pandas as pd
import yaml

CONTRACT_PATH = Path(__file__).parents[2] / "data_contracts" / "paysim_contract.yml"


def validate_paysim(frame: pd.DataFrame, contract_path: str | Path = CONTRACT_PATH):
    """Run the contract expectations and return the Great Expectations result."""
    contract = yaml.safe_load(Path(contract_path).read_text(encoding="utf-8"))
    validator = gx.from_pandas(frame)
    validator.expect_table_columns_to_match_ordered_list(list(contract["columns"]))
    for name, rules in contract["columns"].items():
        if not rules["nullable"]:
            validator.expect_column_values_to_not_be_null(name)
        if "allowed" in rules:
            validator.expect_column_values_to_be_in_set(name, rules["allowed"])
        if "min" in rules:
            validator.expect_column_values_to_be_between(name, min_value=rules["min"])
    return validator.validate()


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m src.data.validate PATH_TO_CSV_OR_PARQUET")
        return 2
    path = Path(sys.argv[1])
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    result = validate_paysim(frame)
    print(result)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())