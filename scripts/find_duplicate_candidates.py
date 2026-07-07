"""Flag FLOGA event pairs that look like duplicate views of one fire.

Signature (learned from Evia 2021, events 61/62): same year, same pre AND
post Sentinel-2 datatakes (sensing timestamp), different tiles. Such pairs
can only coexist where adjacent tiles overlap — i.e. one fire, two views.

Reads data/floga/<year>_label_stats.csv produced by scan_labels.py.
Candidates need label-overlap confirmation (scripts/check_duplicate_event.py)
before being treated as proven duplicates.
"""

import csv
from pathlib import Path


def scene_datatake(scene: str) -> str:
    return scene.split("_")[2] if scene else ""


def scene_tile(scene: str) -> str:
    return scene.split("_")[5] if scene else ""


def main() -> None:
    split = {}
    with open("data/floga/data_split.csv") as fh:
        for row in csv.DictReader(fh):
            split[(row["year"], row["event_id"])] = row["set"]

    candidates = []
    for path in sorted(Path("data/floga").glob("*_label_stats.csv")):
        year = path.stem.split("_")[0]
        rows = list(csv.DictReader(open(path)))
        if rows and "pre_scene" not in rows[0]:
            print(f"skipping {path.name}: no scene columns (rescan needed)")
            continue
        for i, a in enumerate(rows):
            for b in rows[i + 1 :]:
                if (
                    scene_datatake(a["pre_scene"]) == scene_datatake(b["pre_scene"])
                    and scene_datatake(a["post_scene"]) == scene_datatake(b["post_scene"])
                    and scene_tile(a["pre_scene"]) != scene_tile(b["pre_scene"])
                    and int(a["burned_ha"]) > 0
                    and int(b["burned_ha"]) > 0
                ):
                    sa = split.get((year, a["event_id"]), "?")
                    sb = split.get((year, b["event_id"]), "?")
                    candidates.append((year, a, b, sa, sb))

    print(f"{len(candidates)} candidate duplicate pairs\n")
    for year, a, b, sa, sb in candidates:
        leak = "  <-- SPLIT LEAK" if sa != sb and "?" not in (sa, sb) else ""
        print(
            f"{year}: event {a['event_id']} ({scene_tile(a['pre_scene'])}, "
            f"{int(a['burned_ha']):,} ha, {sa}) | event {b['event_id']} "
            f"({scene_tile(b['pre_scene'])}, {int(b['burned_ha']):,} ha, {sb})"
            f"  dates {a['pre_date']}->{a['post_date']}{leak}"
        )


if __name__ == "__main__":
    main()
