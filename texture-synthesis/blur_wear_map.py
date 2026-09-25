import os
import argparse
import cv2

parser = argparse.ArgumentParser(description="Apply Gaussian blur to a wear map to smooth the boundaries between wear levels.")
parser.add_argument('input', type=str, help="path to the input wear map")
parser.add_argument('-o', '--output', type=str, default=None,
                    help="path to the output wear map (default: <input>_blurred.png)")
parser.add_argument('--sigma', type=float, default=10.0, help="standard deviation of the Gaussian kernel")


def main(args):
    # Load the image
    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    # Apply Gaussian blur (with ksize = (0, 0), the kernel size is computed from sigma)
    blurred_img = cv2.GaussianBlur(img, (0, 0), sigmaX=args.sigma)

    # Save
    output_path = args.output
    if output_path is None:
        root, _ = os.path.splitext(args.input)
        output_path = f"{root}_blurred.png"
    cv2.imwrite(output_path, blurred_img)
    print(f"Saved blurred wear map to: {output_path}")


if __name__ == "__main__":
    main(parser.parse_args())
