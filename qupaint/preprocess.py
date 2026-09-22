"""Image preprocessing for QuPAINT.

The released checkpoint expects the same pipeline used at tuning time:

1. the micrograph is resized to a fixed 1792x1344 canvas,
2. that canvas is tiled into 448x448 patches by aspect-ratio matching
   (InternVL-style dynamic tiling, up to `max_num` tiles),
3. a downscaled thumbnail of the whole image is appended as a final tile,
4. every tile is normalised with ImageNet statistics.

Because boxes are predicted in *percent* of image size, the fixed canvas does
not distort results -- but changing `DEFAULT_CANVAS` moves the model away from
the resolution it was tuned at, so keep it unless you know why you are changing it.
"""

from typing import List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_CANVAS: Tuple[int, int] = (1792, 1344)
DEFAULT_TILE_SIZE: int = 448
DEFAULT_MAX_TILES: int = 12


def build_transform(input_size: int = DEFAULT_TILE_SIZE) -> T.Compose:
    """Tile transform: RGB, bicubic resize to `input_size`, ImageNet normalisation."""
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = DEFAULT_MAX_TILES,
    image_size: int = DEFAULT_TILE_SIZE,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    """Split `image` into `image_size` tiles using the closest allowed aspect ratio."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    cols = target_width // image_size
    processed_images = []
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(
    image_file,
    input_size: int = DEFAULT_TILE_SIZE,
    max_num: int = DEFAULT_MAX_TILES,
    canvas: Tuple[int, int] = DEFAULT_CANVAS,
) -> torch.Tensor:
    """Load a micrograph and return the stacked tile tensor of shape (tiles, 3, S, S)."""
    image = image_file
    if not isinstance(image, Image.Image):
        image = Image.open(image_file)
    image = image.convert("RGB").resize(canvas)

    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    return torch.stack([transform(tile) for tile in tiles])
