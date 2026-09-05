"""Rebuild the report's annotator map from the two source workbooks (stdlib only)."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from fm_adaptation.neurite_annotations import image_id


def read_rows(path):
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as book:
        strings = (["".join(node.itertext()) for node in
                    ET.fromstring(book.read("xl/sharedStrings.xml")).findall("s:si", ns)]
                   if "xl/sharedStrings.xml" in book.namelist() else [])
        for row in ET.fromstring(book.read("xl/worksheets/sheet1.xml")).findall(".//s:row", ns):
            values = {}
            for cell in row.findall("s:c", ns):
                value = cell.find("s:v", ns)
                value = value.text if value is not None else "".join(cell.itertext())
                if cell.get("t") == "s":
                    value = strings[int(value)]
                values[''.join(c for c in cell.get("r") if c.isalpha())] = value
            yield values


def convert(path, paired):
    cases = {}
    for row in list(read_rows(path))[1:]:
        if not row.get("D") or not row.get("F"):
            continue
        stem = image_id(row["D"])
        people = [(row["F"], row.get("J" if paired else "H"))]
        if paired and row.get("G"):
            people.append((row["G"], row.get("K")))
        for index, (person, roi) in enumerate(people, 1):
            if roi:
                key = stem + (f"_rater{index}" if len(people) > 1 else "")
                if key in cases:
                    raise ValueError(f"Duplicate annotation {key} in {path}")
                cases[key] = person.strip()
    return {"workbook": str(path), "sheet": "Sheet1", "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yvonne", type=Path, required=True)
    parser.add_argument("--yvonne-b2", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("src/fm_adaptation/assets/neurite_annotators.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps({"Yvonne": convert(args.yvonne, True),
                                      "Yvonne_b2": convert(args.yvonne_b2, False)}, indent=2) + "\n")
