"""QuPAINT inference on one micrograph or a folder of micrographs.

    python inference.py --checkpoint uark-cviu/QuPAINT-4B \
                        --image examples/sample-01.jpg \
                        --prompt "Identify monolayer candidates and provide bounding boxes [x,y,w,h]"

Annotated images and the raw response are written to --save_dir.
"""

import argparse
import glob
import json
import os

import torch
from PIL import Image
from transformers import AutoTokenizer

from qupaint import QuPAINTModel, draw_boxes, load_image, parse_boxes
from qupaint.preprocess import DEFAULT_CANVAS, DEFAULT_MAX_TILES

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

DEFAULT_PROMPT = "Identify all flakes and provide bounding boxes [x,y,w,h] for each"


def load_model(checkpoint: str, device: str = "cuda", use_flash_attn: bool = False):
    model = QuPAINTModel.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn,
    ).eval()
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    return model, tokenizer


def infer_one(
    model,
    tokenizer,
    image_path: str,
    prompt: str = DEFAULT_PROMPT,
    save_dir: str = "./outputs",
    max_new_tokens: int = 4096,
    do_sample: bool = False,
    max_tiles: int = DEFAULT_MAX_TILES,
):
    """Run QuPAINT on one image; returns (response, boxes_in_pixels, annotated_path)."""
    os.makedirs(save_dir, exist_ok=True)

    pixel_values = load_image(image_path, max_num=max_tiles)
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    response = model.chat(
        tokenizer, pixel_values, f"<image>\n{prompt}.", generation_config
    )

    canvas = Image.open(image_path).convert("RGB").resize(DEFAULT_CANVAS)
    width, height = canvas.size
    boxes = parse_boxes(response, height=height, width=width)

    annotated_path = os.path.join(save_dir, os.path.basename(image_path))
    draw_boxes(canvas, boxes).save(annotated_path)

    stem = os.path.splitext(annotated_path)[0]
    with open(f"{stem}.json", "w") as fh:
        json.dump(
            {"image": image_path, "prompt": prompt, "response": response, "boxes_px": boxes},
            fh,
            indent=2,
        )
    return response, boxes, annotated_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="uark-cviu/QuPAINT-4B",
                        help="Hugging Face repo id or local checkpoint directory")
    parser.add_argument("--image", required=True, help="image file or directory")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--save_dir", default="./outputs")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_tiles", type=int, default=DEFAULT_MAX_TILES,
                        help="cap on 448px tiles; lower it to trade accuracy for speed")
    parser.add_argument("--do_sample", action="store_true",
                        help="sample instead of greedy decoding (greedy is reproducible)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flash_attn", action="store_true",
                        help="use FlashAttention in the vision encoder (needs flash-attn)")
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint, args.device, args.flash_attn)

    if os.path.isdir(args.image):
        paths = sorted(
            p for p in glob.glob(os.path.join(args.image, "*"))
            if p.lower().endswith(IMAGE_EXTS)
        )
        if not paths:
            raise SystemExit(f"No images found in {args.image}")
    else:
        paths = [args.image]

    for path in paths:
        response, boxes, annotated = infer_one(
            model, tokenizer, path, args.prompt, args.save_dir,
            args.max_new_tokens, args.do_sample, args.max_tiles,
        )
        print(f"\n=== {path} ===\n{response}\n--> {len(boxes)} box(es); saved {annotated}")


if __name__ == "__main__":
    main()
