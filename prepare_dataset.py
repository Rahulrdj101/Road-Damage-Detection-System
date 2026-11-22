
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
import random
from collections import Counter

# Original dataset
SOURCE = Path(
    r"C:\Users\Lenovo\.cache\kagglehub\datasets\musfequa\india-road-damage\versions\1\India"
)

# Project dataset
DEST = Path(r"E:\Road Damage Detection System\dataset")

# Preserve all observed classes
CLASSES = [
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
]

CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}

random.seed(42)

def convert_annotation(xml_path, label_path):
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing image dimensions: {xml_path.name}")

    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {xml_path.name}")

    lines = []

    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()

        if name not in CLASS_MAP:
            raise ValueError(f"Unknown class {name!r} in {xml_path.name}")

        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bounding box: {xml_path.name}")

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        # Clip coordinates to image boundaries
        xmin = max(0, min(xmin, width))
        xmax = max(0, min(xmax, width))
        ymin = max(0, min(ymin, height))
        ymax = max(0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid bounding box: {xml_path.name}")

        # YOLO normalized center coordinates and dimensions
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        class_id = CLASS_MAP[name]

        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    label_path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def main():
    train_images = SOURCE / "train" / "images"
    xml_folder = SOURCE / "train" / "annotations" / "xmls"
    test_images = SOURCE / "test" / "images"

    annotated = []
    empty_images = []
    class_counts = Counter()
    total_objects = 0

    print("Checking and converting annotations...")

    for xml_path in sorted(xml_folder.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        image_name = root.findtext("filename")

        if not image_name:
            raise ValueError(f"Missing filename in {xml_path.name}")

        image_path = train_images / image_name

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        object_count = len(root.findall("object"))

        if object_count:
            annotated.append((image_path, xml_path))
        else:
            empty_images.append(image_path)

    random.shuffle(annotated)

    split_index = int(len(annotated) * 0.8)
    train_items = annotated[:split_index]
    val_items = annotated[split_index:]

    def process_split(items, split):
        nonlocal total_objects

        for image_path, xml_path in items:
            label_name = image_path.stem + ".txt"

            output_image = DEST / "images" / split / image_path.name
            output_label = DEST / "labels" / split / label_name

            shutil.copy2(image_path, output_image)
            count = convert_annotation(xml_path, output_label)
            total_objects += count

            root = ET.parse(xml_path).getroot()
            for obj in root.findall("object"):
                class_counts[obj.findtext("name", "UNKNOWN")] += 1

    process_split(train_items, "train")
    process_split(val_items, "val")

    # Include empty-annotation images in training as negative examples
    for image_path in empty_images:
        shutil.copy2(
            image_path,
            DEST / "images" / "train" / image_path.name
        )
        (DEST / "labels" / "train" /
         (image_path.stem + ".txt")).write_text("", encoding="utf-8")

    # Copy test images without inventing labels
    test_count = 0
    for image_path in test_images.glob("*.jpg"):
        shutil.copy2(
            image_path,
            DEST / "images" / "test" / image_path.name
        )
        test_count += 1

    print("\nDATASET PREPARATION COMPLETE")
    print("Annotated training images:", len(train_items))
    print("Validation images:", len(val_items))
    print("Empty training images:", len(empty_images))
    print("Test images:", test_count)
    print("Converted damage objects:", total_objects)
    print("Class counts:", dict(class_counts))
    print("Class mapping:", CLASS_MAP)


if __name__ == "__main__":
    main()
