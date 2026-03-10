import json
from pathlib import Path


def convert_filtered(coco_json_path: str, images_folder: str, output_path: str) -> None:
    """
    Reads a COCO-format JSON and filters annotations to only the images
    that exist in images_folder, then saves a simplified ground_truth.json
    in the format expected by the pipeline:
        {"image_name.jpg": [[x, y, w, h], ...], ...}
    """
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Build set of image filenames that were actually copied
    copied_images = {p.name for p in Path(images_folder).glob("*.*")
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    print(f"Images found in folder : {len(copied_images)}")

    # Map image_id -> filename, but only for copied images
    image_id_to_name: dict[int, str] = {}
    for img in coco["images"]:
        if img["file_name"] in copied_images:
            image_id_to_name[img["id"]] = img["file_name"]

    print(f"Images matched in COCO  : {len(image_id_to_name)}")

    result: dict[str, list] = {}

    for ann in coco["annotations"]:
        image_id = ann["image_id"]

        if image_id not in image_id_to_name:
            continue

        image_name = image_id_to_name[image_id]
        # COCO bbox is already [x, y, width, height] — cast to int
        bbox = [int(b) for b in ann["bbox"]]

        result.setdefault(image_name, []).append(bbox)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    total_boxes = sum(len(v) for v in result.values())
    print(f"Annotations written    : {total_boxes} boxes across {len(result)} images")
    print(f"ground_truth.json saved to: {output_path}")


if __name__ == "__main__":
    # Resolve paths relative to this script's location
    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parents[1]          # image-processing/

    coco_json   = Path("/Users/asadkhan/Downloads/archive/annotations/instances_default.json")
    images_dir  = repo_root / "data" / "final-project"
    output_file = images_dir / "ground_truth.json"

    convert_filtered(str(coco_json), str(images_dir), str(output_file))

