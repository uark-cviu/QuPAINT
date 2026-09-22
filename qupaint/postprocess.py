"""Parsing and visualisation of QuPAINT responses.

QuPAINT answers in free text and closes with a `<CONCLUSION>` span that holds the
flakes it commits to, each as a `<box>x, y, w, h</box>` quadruple in **percent**
of image width/height (top-left origin).
"""

import re
from typing import List, Sequence, Tuple

CONCLUSION_RE = re.compile(r"<\s*/?\s*CONCLUSION>(.*?)<\s*/?\s*CONCLUSION>", re.DOTALL)
BOX_RE = re.compile(r"<box>(.*?)</box>", re.DOTALL)

Box = Tuple[float, float, float, float]


def parse_conclusion_boxes(response: str) -> List[Box]:
    """Return the `<box>` quadruples inside the `<CONCLUSION>` span, in percent."""
    match = CONCLUSION_RE.search(response)
    if not match:
        return []
    boxes = []
    for raw in BOX_RE.findall(match.group(1)):
        try:
            coords = [float(v.strip()) for v in raw.split(",")]
        except ValueError:
            continue
        if len(coords) == 4:
            boxes.append(tuple(coords))
    return boxes


def denorm_box(box: Box, height: int, width: int) -> Tuple[int, int, int, int]:
    """Percent box (x, y, w, h) -> pixel box (x, y, w, h) for a `height` x `width` image."""
    x, y, w, h = box
    return (
        int(x / 100 * width),
        int(y / 100 * height),
        int(w / 100 * width),
        int(h / 100 * height),
    )


def parse_boxes(response: str, height: int, width: int) -> List[Tuple[int, int, int, int]]:
    """Parse a response straight into pixel boxes for an image of the given size."""
    return [denorm_box(box, height, width) for box in parse_conclusion_boxes(response)]


def draw_boxes(
    image,
    boxes: Sequence[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
):
    """Draw pixel boxes onto a PIL image and return a new annotated image."""
    from PIL import ImageDraw

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for x, y, w, h in boxes:
        draw.rectangle([x, y, x + w, y + h], outline=color, width=thickness)
    return annotated
