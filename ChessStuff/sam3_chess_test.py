#!/usr/bin/env python3
"""
Hugging Face + SAM Chess Piece Detector (Auto-download version)
"""

import cv2
import numpy as np
import os
import torch
import urllib.request
from transformers import pipeline
from segment_anything import sam_model_registry, SamPredictor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# WARNING: Update these if you reorganize folders
IMAGE_PATH = os.path.join(SCRIPT_DIR, "piece_templates", "hdr_loaded_with_lighting.png")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "piece_templates", "chess_overlay.png")

# Text prompt
TEXT_PROMPT = "white pawn, white rook, white knight, white bishop, white queen, white king, black pawn, black rook, black knight, black bishop, black queen, black king"

# SAM checkpoint (will be downloaded automatically)
SAM_CHECKPOINT = os.path.join(SCRIPT_DIR, "sam_vit_b_01ec64.pth")
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def download_sam_if_needed():
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"Downloading SAM model (first time only, ~375 MB)...")
        urllib.request.urlretrieve(SAM_URL, SAM_CHECKPOINT)
        print("Download complete!")


def main():
    print("=== Hugging Face + SAM Chess Detection ===")
    print(f"Image: {IMAGE_PATH}")

    download_sam_if_needed()

    # Load zero-shot detector
    print("Loading zero-shot detector (downloads ~1.5GB on first run)...")
    detector = pipeline(
        "zero-shot-object-detection",
        model="IDEA-Research/grounding-dino-tiny",
        device=0 if torch.cuda.is_available() else -1
    )

    # Load SAM
    print("Loading SAM...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(device)
    predictor = SamPredictor(sam)

    # Detect
    print("Detecting pieces...")
    results = detector(IMAGE_PATH, candidate_labels=TEXT_PROMPT.split(", "), threshold=0.2)
    print(f"Found {len(results)} pieces")

    # Create masks with SAM
    image = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    overlay = image.copy()

    for item in results:
        box = item["box"]
        label = item["label"]
        score = item["score"]

        input_box = np.array([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        mask = masks[0].astype(np.uint8) * 255
        color = (255, 180, 80) if "white" in label else (40, 40, 220)

        colored = np.zeros_like(image)
        colored[:] = color
        colored = cv2.bitwise_and(colored, colored, mask=mask)

        alpha = 0.45
        overlay = cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        cv2.putText(overlay, f"{label} {score:.2f}", (box["xmin"], box["ymin"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, overlay)
    print(f"\n✅ Saved to: {OUTPUT_PATH}")

    cv2.namedWindow("Chess Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Chess Detection", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()