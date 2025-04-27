from pathlib import Path

# Path building
src_path = Path(__file__).resolve().parent
DATA_DIR = src_path.parent / 'data'
OUTPUTS_DIR = src_path.parent / 'outputs'

# Ensure the outputs directory exists
OUTPUTS_DIR.mkdir(exist_ok=True)

print(f"[CONFIG] DATA_DIR set to: {DATA_DIR}")
print(f"[CONFIG] OUTPUTS_DIR set to: {OUTPUTS_DIR}")


