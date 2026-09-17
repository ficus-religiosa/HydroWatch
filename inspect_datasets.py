"""
Inspect the three datasets before defining the unified class taxonomy.

Usage examples:

    python inspect_datasets.py --uppd PATH
    python inspect_datasets.py --trashcan PATH
    python inspect_datasets.py --seaclear PATH --seaclear-json PATH

The script reports image counts and raw class names/IDs.
It intentionally does NOT create a final class mapping.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inspect_uppd(root):
    root = Path(root)
    image_count = sum(
        1
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
    )

    ids = set()

    for p in root.rglob("*.txt"):
        for line in p.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines():
            parts = line.split()
            if parts:
                try:
                    ids.add(int(float(parts[0])))
                except ValueError:
                    pass

    print("\nUPPD")
    print("  Images:", image_count)
    print("  Raw YOLO class IDs:", sorted(ids))


def inspect_trashcan(root):
    root = Path(root)
    xmls = list(root.rglob("*.xml"))
    names = set()

    for p in xmls:
        try:
            root_xml = ET.parse(p).getroot()
        except ET.ParseError:
            continue

        for obj in root_xml.findall("object"):
            name = obj.findtext("name")
            if name:
                names.add(name)

    print("\nTrashCan")
    print("  XML annotations:", len(xmls))
    print("  Raw class names:", sorted(names))


def inspect_seaclear(root, annotation_file=None):
    root = Path(root)

    if annotation_file:
        p = Path(annotation_file)
    else:
        jsons = sorted(root.rglob("*.json"))
        preferred = [
            x for x in jsons
            if "annot" in x.name.lower()
            or "instances" in x.name.lower()
        ]
        p = preferred[0] if preferred else jsons[0]

    data = json.loads(
        p.read_text(
            encoding="utf-8"
        )
    )

    categories = {
        int(x["id"]): x["name"]
        for x in data.get("categories", [])
    }

    print("\nSeaClear")
    print("  Images:", len(data.get("images", [])))
    print("  Annotations:", len(data.get("annotations", [])))
    print("  Categories:", len(categories))

    for cid, name in sorted(
        categories.items()
    ):
        print(
            f"    {cid}: {name}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--uppd")
    parser.add_argument("--trashcan")
    parser.add_argument("--seaclear")
    parser.add_argument("--seaclear-json")

    args = parser.parse_args()

    if not any([
        args.uppd,
        args.trashcan,
        args.seaclear
    ]):
        parser.error(
            "Provide at least one dataset path."
        )

    if args.uppd:
        inspect_uppd(args.uppd)

    if args.trashcan:
        inspect_trashcan(args.trashcan)

    if args.seaclear:
        inspect_seaclear(
            args.seaclear,
            args.seaclear_json
        )


if __name__ == "__main__":
    main()
