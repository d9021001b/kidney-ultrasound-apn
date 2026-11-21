import csv
import json
import os
import re

csv_path = os.path.join("NCKUH", "NCKUH.csv")
json_path = "nckuh_metadata.json"

def parse_filename(fname):
    # Example: 04680079_20070809_1_12.66_11.6_LL.png
    # or      04680079_20070809_2_16.54_11.3685501_LL (L).png
    # Find _1_ or _2_ (gender), then two floats after
    m = re.search(r"_(1|2)_([0-9.]+)_([0-9.]+)_", fname)
    if not m:
        # Try alternative: sometimes the last float may not be followed by _
        m = re.search(r"_(1|2)_([0-9.]+)_([0-9.]+)", fname)
    if not m:
        return None, None, None
    gender = "male" if m.group(1) == "1" else "female"
    long_axis = float(m.group(2))
    short_axis = float(m.group(3))
    return gender, long_axis, short_axis

data = []
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) < 2:
            continue
        fname = row[0]
        label_str = row[1]
        # Skip header or non-integer label rows
        try:
            label = int(label_str)
        except ValueError:
            continue
        gender, long_axis, short_axis = parse_filename(fname)
        entry = {
            "filename": fname,
            "label": label,
            "gender": gender,
            "long_axis": long_axis,
            "short_axis": short_axis
        }
        data.append(entry)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Extracted metadata saved to {json_path}")
