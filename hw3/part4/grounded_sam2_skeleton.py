import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
from PIL import Image
import groundingdino.datasets.transforms as T
from tqdm import tqdm
import argparse

"""
Hyper parameters
"""
TEXT_PROMPT = ("There is tree with lions sitting on its branches.")
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
OUTPUT_DIR = Path("outputs")
DUMP_JSON_RESULTS = True
DEFAULT_VALUE = 0

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_canny_edge_map(image_rgb: np.ndarray, threshold1: int, threshold2: int) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=threshold1, threshold2=threshold2)
    return edges


def grounding_tensor_from_rgb(image_rgb: np.ndarray) -> torch.Tensor:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pil = Image.fromarray(image_rgb)
    image_transformed, _ = transform(image_pil, None)
    return image_transformed

def initialize_model(device):
    # build SAM2 image predictor
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=device
    )

    return sam2_predictor, grounding_model

def process_image(
    img_path: str,
    sam2_predictor,
    grounding_model,
    device: str,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    canny_threshold1: int,
    canny_threshold2: int,
):
    base_name = Path(img_path).stem
    output_edge_file = OUTPUT_DIR / f"{base_name}_canny_edges.png"
    output_mask_file = OUTPUT_DIR / f"{base_name}_grounded_sam2_annotated.jpg"
    output_json_file = OUTPUT_DIR / f"{base_name}_grounded_sam2_results.json"
    combined_mask_file = OUTPUT_DIR / f"{base_name}_mask.png"

    # Skip processing if the mask file already exists
    if output_mask_file.exists() and output_json_file.exists() and output_edge_file.exists() and combined_mask_file.exists():
        print(f"Skipping {img_path}, results already exist.")
        return

    # Load original image
    image_source, _ = load_image(img_path)  # image_source is RGB np.ndarray

    # 1) Canny edge map
    edges = compute_canny_edge_map(
        image_rgb=image_source,
        threshold1=canny_threshold1,
        threshold2=canny_threshold2,
    )
    cv2.imwrite(str(output_edge_file), edges)

    # 2) Use edge map (as 3-channel RGB) as input to Grounded-SAM
    edge_rgb = np.stack([edges, edges, edges], axis=-1).astype(np.uint8)
    edge_tensor = grounding_tensor_from_rgb(edge_rgb)
    sam2_predictor.set_image(edge_rgb)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=edge_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    # Process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes.to(device) * torch.tensor([w, h, w, h], device=device)
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    if input_boxes.size == 0:
        print(f"No boxes found for prompt: {text_prompt}")
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(str(combined_mask_file), combined_mask)
        # still save a simple visualization (original image) to keep outputs consistent
        img = cv2.imread(str(img_path))
        cv2.imwrite(str(output_mask_file), img)
        if DUMP_JSON_RESULTS:
            results = {
                "image_path": str(img_path),
                "edge_map_path": str(output_edge_file),
                "edge_input": {
                    "type": "canny",
                    "threshold1": int(canny_threshold1),
                    "threshold2": int(canny_threshold2),
                },
                "text_prompt": text_prompt,
                "annotations": [],
                "box_format": "xyxy",
                "img_width": w,
                "img_height": h,
            }
            with open(output_json_file, "w") as f:
                json.dump(results, f, indent=4)
        return

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.cpu().numpy().tolist()
    class_names = labels

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]
    
    print("labels:", labels)

    # Visualize image with supervision API
    img = cv2.imread(str(img_path))

    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(str(output_mask_file), annotated_frame)

    # Save a combined binary mask for convenience
    if masks.size > 0:
        combined_mask = np.any(masks.astype(bool), axis=0).astype(np.uint8) * 255
    else:
        combined_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.imwrite(str(combined_mask_file), combined_mask)

    if DUMP_JSON_RESULTS:
        def single_mask_to_rle(mask):
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes = input_boxes.tolist()
        scores = scores.tolist()
        results = {
            "image_path": str(img_path),
            "edge_map_path": str(output_edge_file),
            "edge_input": {
                "type": "canny",
                "threshold1": int(canny_threshold1),
                "threshold2": int(canny_threshold2),
            },
            "text_prompt": text_prompt,
            "annotations" : [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }

        with open(output_json_file, "w") as f:
            json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Grounded SAM2 Image Processing")
    parser.add_argument("--image", type=str, required=True, help="Path to an input image")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (if CUDA available)")
    parser.add_argument("--text-prompt", type=str, default=TEXT_PROMPT, help="Text prompt for GroundingDINO")
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD, help="Box threshold")
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD, help="Text threshold")
    parser.add_argument("--canny1", type=int, default=100, help="Canny threshold1")
    parser.add_argument("--canny2", type=int, default=200, help="Canny threshold2")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    sam2_predictor, grounding_model = initialize_model(device)

    process_image(
        args.image,
        sam2_predictor,
        grounding_model,
        device,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        canny_threshold1=args.canny1,
        canny_threshold2=args.canny2,
    )

if __name__ == "__main__":
    main()
