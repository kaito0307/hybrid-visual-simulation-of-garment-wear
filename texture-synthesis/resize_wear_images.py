import os
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description="Resize the 10 wear images (1.jpg to 10.jpg) of a fabric.")
parser.add_argument('input_dir', type=str, help="folder of a fabric in the wear image dataset (e.g., ../wear-image-dataset/chino)")
parser.add_argument('-o', '--output_dir', type=str, default=None,
                    help="output folder (default: data/cloth_texture/<fabric>/<width>x<height>)")
parser.add_argument('--size', type=int, nargs=2, default=[100, 100], metavar=('WIDTH', 'HEIGHT'),
                    help="output size in pixels (default: 100 100)")


def resize_image(input_path, output_path, target_width_px, target_height_px):
    """Resize a single image and save it."""
    with Image.open(input_path) as img:
        # Resize to the given size, ignoring the aspect ratio
        resized_img = img.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)

        # Save
        resized_img.save(output_path, quality=95)


def main(args):
    width, height = args.size

    output_dir = args.output_dir
    if output_dir is None:
        fabric = os.path.basename(os.path.normpath(args.input_dir))
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "data", "cloth_texture", fabric, f"{width}x{height}")

    # Create the output folder if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, 11):
        input_path = os.path.join(args.input_dir, f"{i}.jpg")
        output_path = os.path.join(output_dir, f"{i}.jpg")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")

        resize_image(input_path, output_path, width, height)
        print(f"Saved: {input_path} -> {output_path}")


if __name__ == "__main__":
    main(parser.parse_args())
