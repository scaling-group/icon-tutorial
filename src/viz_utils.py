
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import wandb
import matplotlib



def fig_to_wandb(fig, cfg=None):
    """
    Converts a Matplotlib figure or a PIL image to a wandb.Image object
    Parameters:
    - fig: figure to be converted.
    Returns:
    - wandb.Image object
    """
    if cfg is None:
        cfg = {}

    if type(fig) == matplotlib.figure.Figure:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        wandb_image = wandb.Image(image, **cfg)
        buf.close()
    else:  # already a PIL image
        wandb_image = wandb.Image(fig, **cfg)
    return wandb_image



def merge_images(figs_2d, spacing=0):
    """
    Converts a 2D list of Matplotlib figures to a single PIL image arranged in a grid.
    Parameters:
    - figs_2d: 2D list of Matplotlib figures to be merged and converted.
    - spacing: Space between images in pixels.
    Returns:
    - Merged and converted PIL image.
    """
    # Store the merged images of each row
    row_images = []
    max_row_height = 0
    total_width = 0

    # Process each row
    for figs_row in figs_2d:
        imgs = []
        bufs = []
        for fig in figs_row:
            buf = io.BytesIO()
            bufs.append(buf)
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
            imgs.append(img)

        # Determine the total size for this row
        row_width = sum(img.width for img in imgs) + spacing * (len(imgs) - 1)
        row_height = max(img.height for img in imgs)
        max_row_height = max(max_row_height, row_height)
        total_width = max(total_width, row_width)

        # Create row image and paste figures
        row_image = Image.new("RGB", (row_width, row_height))
        x_offset = 0
        for img in imgs:
            row_image.paste(img, (x_offset, 0))
            x_offset += img.width + spacing

        row_images.append(row_image)

        # Close all the buffers
        for buf in bufs:
            buf.close()

    # Determine total size for the final merged image
    total_height = sum(img.height for img in row_images) + spacing * (len(row_images) - 1)

    # Create final merged image and paste row images
    merged_img = Image.new("RGB", (total_width, total_height))
    y_offset = 0
    for img in row_images:
        merged_img.paste(img, (0, y_offset))
        y_offset += img.height + spacing

    return merged_img