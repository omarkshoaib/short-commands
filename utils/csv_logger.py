import os
import csv
from typing import Dict, Any, List

import config


def _read_existing_header(path: str) -> List[str]:
	try:
		with open(path, "r", newline="", encoding="utf-8") as f:
			reader = csv.reader(f)
			for row in reader:
				return row
	except FileNotFoundError:
		return []
	except Exception:
		# If unreadable, force rotation
		return []
	return []


def _sanitize(value: Any) -> str:
	text = "" if value is None else str(value)
	# Replace newlines/tabs that may break row shape
	return text.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()


def append_result_row(row: Dict[str, Any]) -> None:
	os.makedirs(os.path.dirname(config.RESULTS_CSV), exist_ok=True)

	# Ensure schema: rotate legacy file if header mismatches current schema
	existing_header = _read_existing_header(config.RESULTS_CSV)
	if existing_header and existing_header != config.CSV_COLUMNS:
		base = os.path.splitext(config.RESULTS_CSV)[0]
		bak_path = f"{base}_legacy.csv"
		try:
			os.replace(config.RESULTS_CSV, bak_path)
		except Exception:
			# If replace fails, attempt rename
			try:
				os.rename(config.RESULTS_CSV, bak_path)
			except Exception:
				pass

	# Normalize and sanitize values for the configured columns
	normalized = {key: _sanitize(row.get(key, "")) for key in config.CSV_COLUMNS}
	values = [normalized[key] for key in config.CSV_COLUMNS]

	# Write header if file missing
	file_exists = os.path.exists(config.RESULTS_CSV)
	mode = "a" if file_exists else "w"
	with open(config.RESULTS_CSV, mode, newline="", encoding="utf-8") as f:
		writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
		if not file_exists:
			writer.writerow(config.CSV_COLUMNS)
		writer.writerow(values)

