# scripts/recognize_text.py

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_model(model_name, device):
    print(f"Loading recognition model: {model_name}")

    processor = TrOCRProcessor.from_pretrained(model_name)

    model = VisionEncoderDecoderModel.from_pretrained(
        model_name
    ).to(device)

    model.eval()

    return processor, model


def recognize_image(image_path, processor, model, device):
    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text


def main():
    parser = argparse.ArgumentParser(
        description="Recognize text in images using TrOCR."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Image file or directory containing image crops."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for recognition results."
    )

    parser.add_argument(
        "--model",
        default="microsoft/trocr-base-handwritten"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted(
            path for path in input_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not image_paths:
        raise RuntimeError("No supported images found.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    processor, model = load_model(
        args.model,
        device
    )

    results = []

    for image_path in image_paths:
        try:
            recognized_text = recognize_image(
                image_path,
                processor,
                model,
                device
            )

            results.append({
                "image": str(image_path),
                "text": recognized_text,
                "recognizer": "trocr"
            })

            print(
                f"{image_path.name}: {recognized_text}"
            )

        except Exception as error:
            print(f"Error processing {image_path}: {error}")

    results_path = output_dir / "transcriptions.json"

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"\nProcessed {len(results)} images.")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()