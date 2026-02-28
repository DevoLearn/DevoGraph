import cv2
import numpy as np
from devolearn import cell_nucleus_segmentor


class WatershedSegmenter:
    """
    Segments microscopy images using devolearn's cell_nucleus_segmentor
    neural network for preprocessing, followed by watershed instance
    segmentation.

    The devolearn model produces a probability map highlighting cell
    nuclei. This is then thresholded and passed through the watershed
    algorithm to produce an instance-labeled mask where each cell gets
    a unique integer label.
    """

    def __init__(self, threshold=10):
        self.threshold = threshold
        # Initialize the devolearn neural network segmentor.
        self.segmentor = cell_nucleus_segmentor()

    def _preprocess(self, image_path):
        """
        Preprocesses a raw microscopy feature image using devolearn's
        cell_nucleus_segmentor neural network to produce a clean
        segmentation prediction (probability map).

        The output is resized to match the original image dimensions
        (devolearn internally resizes to 350x250).
        """
        # Read original to get its dimensions
        original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if original is None:
            print(f"Error: Could not read image at {image_path}")
            return None
        orig_h, orig_w = original.shape[:2]

        seg_pred = self.segmentor.predict(image_path=image_path)

        # probability map to image
        seg_image = (255 * seg_pred).astype(np.uint8)

        # Fix size mismatch: devolearn resizes to (350,250) internally,
        # resize back to match the original input dimensions.
        if seg_image.shape[:2] != (orig_h, orig_w):
            seg_image = cv2.resize(seg_image, (orig_w, orig_h),
                                   interpolation=cv2.INTER_LINEAR)

        return seg_image

    def segment(self, image_path, output_path=None):
        """
        Segments an image using devolearn neural network preprocessing
        followed by thresholding and watershed instance segmentation.

        Args:
            image_path: Path to the input microscopy image.
            output_path: If provided, saves the instance-labeled mask
                         as a 16-bit PNG to this path.

        Returns:
            np.ndarray (int32): Instance-labeled mask (0 = background,
            1..N = individual cell labels), or None on error.
        """
        # Preprocess with devolearn's neural network
        preprocessed = self._preprocess(image_path)
        if preprocessed is None:
            print(f"Error: Could not preprocess image at {image_path}")
            return None

        # Binary thresholding on the preprocessed (clean) image
        _, binary_image = cv2.threshold(
            preprocessed, self.threshold, 255, cv2.THRESH_BINARY
        )

        # Watershed (instance-labeled) segmentation
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
        dt_max = float(dist_transform.max())
        if dt_max <= 0.0:
            sure_fg = np.zeros_like(opening)
        else:
            _, sure_fg = cv2.threshold(dist_transform, 0.2 * dt_max, 255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        num_cc, markers = cv2.connectedComponents(sure_fg)
        if num_cc <= 1:
            num_cc, label_mask = cv2.connectedComponents(opening)
            label_mask = label_mask.astype(np.int32)
        else:
            markers = markers + 1
            markers[unknown == 255] = 0

            grad = cv2.morphologyEx(preprocessed, cv2.MORPH_GRADIENT, kernel)
            grad_bgr = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(grad_bgr, markers)

            label_mask = markers.astype(np.int32)

        # Convert watershed output to a clean label image:
        # - boundaries are -1 -> 0
        # - background is 1 -> 0
        label_mask[label_mask < 0] = 0
        label_mask[label_mask == 1] = 0
        label_mask = (label_mask - 1).astype(np.int32)
        label_mask[label_mask < 0] = 0

        # Save mask if output path is specified
        if output_path is not None:
            # Save the actual 16-bit raw mask for downstream processing
            label_mask_to_save = label_mask.astype(np.uint16)
            ok = cv2.imwrite(output_path, label_mask_to_save)
            if not ok:
                print(f"Error: Could not write mask to {output_path}")
                return None
            print(f"Saved raw instance mask to     {output_path}")

            # Save visible, color-mapped version for human inspection
            if label_mask.max() > 0:
                norm_mask = cv2.normalize(
                    label_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                # Apply a colormap (JET) to give different cells distinct colors
                vis_mask = cv2.applyColorMap(norm_mask, cv2.COLORMAP_JET)
                vis_mask[label_mask == 0] = 0
            else:
                # Empty mask
                vis_mask = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
            
            vis_path = output_path.replace(".png", "_vis.png")
            cv2.imwrite(vis_path, vis_mask)
            print(f"Saved visible color mask to    {vis_path}")

        return label_mask


if __name__ == "__main__":
    import argparse
    import os
    import glob

    parser = argparse.ArgumentParser(
        description="Segment microscopy images using devolearn + watershed"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input image path or directory of images"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, required=True,
        help="Directory to save output mask images"
    )
    parser.add_argument(
        "--threshold", "-t", type=int, default=10,
        help="Binary threshold value (default: 10)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    segmenter = WatershedSegmenter(threshold=args.threshold)

    # Handle single image or directory
    if os.path.isdir(args.input):
        image_files = sorted(
            glob.glob(os.path.join(args.input, "*.png")) +
            glob.glob(os.path.join(args.input, "*.tif")) +
            glob.glob(os.path.join(args.input, "*.tiff")) +
            glob.glob(os.path.join(args.input, "*.jpg"))
        )
        print(f"Found {len(image_files)} images in {args.input}")
    else:
        image_files = [args.input]

    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output_dir, f"{basename}_mask.png")
        print(f"\nProcessing: {os.path.basename(img_path)}")
        mask = segmenter.segment(img_path, out_path)
        if mask is not None:
            num_cells = len(set(mask.flat)) - (1 if 0 in set(mask.flat) else 0)
            print(f"  Cells found: {num_cells}")

    print(f"\nDone. Masks saved to: {args.output_dir}")
