
from pathlib import Path
from collections import Counter

PROJECT = Path(__file__).resolve().parent.parent

image_dir = PROJECT / "dataset" / "images" / "test"
label_dir = PROJECT / "dataset" / "labels" / "test"

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    p for p in image_dir.iterdir()
    if p.suffix.lower() in image_extensions
]

labels = list(label_dir.glob("*.txt"))

image_stems = {p.stem.lower() for p in images}
label_stems = {p.stem.lower() for p in labels}

missing = image_stems - label_stems
extra = label_stems - image_stems

empty = 0
valid = 0
invalid = 0
classes = Counter()

for label in labels:
    content = label.read_text(encoding="utf-8").strip()

    if not content:
        empty += 1
        continue

    file_valid = True

    for line in content.splitlines():
        parts = line.split()

        try:
            if len(parts) != 5:
                raise ValueError()

            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            if not 0 <= class_id < 10:
                raise ValueError()

            if not all(0 <= x <= 1 for x in coords):
                raise ValueError()

            classes[class_id] += 1

        except ValueError:
            file_valid = False
            break

    if file_valid:
        valid += 1
    else:
        invalid += 1

print("\n========== TEST DATASET DIAGNOSTIC ==========")
print("Test images:", len(images))
print("Test label files:", len(labels))
print("Missing label files:", len(missing))
print("Extra label files:", len(extra))
print("Empty label files:", empty)
print("Valid label files:", valid)
print("Invalid label files:", invalid)

print("\nClass distribution:")
for class_id, count in sorted(classes.items()):
    print(f"Class {class_id}: {count} objects")

print("\nFirst 10 missing labels:")
for name in sorted(missing)[:10]:
    print(name + ".txt")

print("\n=============================================")
