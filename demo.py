"""Gradio demo for QuPAINT.

    python demo.py --checkpoint uark-cviu/QuPAINT-4B

Opens on http://127.0.0.1:7860. Add --share for a temporary public link.
"""

import argparse

import gradio as gr
import torch
from PIL import Image

from qupaint import draw_boxes, load_image, parse_boxes
from qupaint.preprocess import DEFAULT_CANVAS, DEFAULT_MAX_TILES
from inference import DEFAULT_PROMPT, load_model

EXAMPLE_PROMPTS = [
    "Identify all flakes and provide bounding boxes [x,y,w,h] for each",
    "Identify monolayer candidates and provide bounding boxes [x,y,w,h]",
    "How many material flakes are there in the image?",
    "Describe the optical contrast of the flakes relative to the substrate",
]


def build_ui(model, tokenizer, max_tiles, max_new_tokens):
    def run(image: Image.Image, prompt: str, do_sample: bool):
        if image is None:
            return "Upload a micrograph first.", None
        pixel_values = load_image(image, max_num=max_tiles).to(torch.bfloat16).to(model.device)
        response = model.chat(
            tokenizer,
            pixel_values,
            f"<image>\n{prompt}.",
            {"max_new_tokens": max_new_tokens, "do_sample": do_sample},
        )
        canvas = image.convert("RGB").resize(DEFAULT_CANVAS)
        boxes = parse_boxes(response, height=canvas.size[1], width=canvas.size[0])
        return response, draw_boxes(canvas, boxes)

    with gr.Blocks(title="QuPAINT") as demo:
        gr.Markdown(
            "# QuPAINT — Physics-Aware Instruction Tuning for Quantum Material Discovery\n"
            "Upload an optical micrograph of 2D material flakes and ask a question. "
            "Boxes in the `<CONCLUSION>` span are drawn on the image."
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Micrograph")
                prompt_input = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=2)
                gr.Examples(EXAMPLE_PROMPTS, inputs=prompt_input, label="Example prompts")
                sample_toggle = gr.Checkbox(
                    label="Sample (unchecked = greedy, reproducible)", value=False
                )
                run_button = gr.Button("Run inference", variant="primary")
            with gr.Column():
                response_output = gr.Textbox(
                    label="Model response", lines=14, max_lines=28, show_copy_button=True
                )
                image_output = gr.Image(type="pil", label="Annotated micrograph")

        run_button.click(
            run,
            inputs=[image_input, prompt_input, sample_toggle],
            outputs=[response_output, image_output],
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="uark-cviu/QuPAINT-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--max_tiles", type=int, default=DEFAULT_MAX_TILES)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--flash_attn", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint, args.device, args.flash_attn)
    demo = build_ui(model, tokenizer, args.max_tiles, args.max_new_tokens)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
