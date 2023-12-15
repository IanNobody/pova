import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import os
from tqdm import tqdm
import argparse
from enum import Enum


class DiffMethod(Enum):
    diff = "diff"
    background_reduction = "background-reduction"

    def __str__(self):
        return self.value


def parseargs():
    parser = argparse.ArgumentParser('Try to remove backgroud of the images .')
    parser.add_argument('-i', '--input-path', required=True, help='Path to input images.')
    parser.add_argument('-o', '--output-path', required=True, help='Where to save the output images.')
    parser.add_argument('-m', '--diff-method', type=DiffMethod, choices=list(DiffMethod), default=DiffMethod.background_reduction,
                        help='Choose diff method. diff - use rgb diff between images, background-reduction - use advanced background reduction.')
    parser.add_argument('-r', '--resize-factor', type=float,
                        help='Resize factor for image aligment. Smaller factor will be faster but less accurate.',
                        default=1)
    parser.add_argument('-v', '--verbose', action='store_true', help='Show images.')
    parser.add_argument('-p', '--reference-file-prefix', default=".ref", help='Reference file prefix.')
    args = parser.parse_args()
    return args




def diff(im1, im2):
    im1 = Image.fromarray(im1)
    im2 = Image.fromarray(im2)

    if im1.mode != im2.mode:
        raise ValueError(
            (
                "Differing color modes:\n  im1: {}\n  im2: {}\n"
                "Ensure image color modes are the same."
            ).format(im1.mode, im2.mode)
        )

    im2 = im2.resize((im1.width, im1.height))

    # Generate diff image in memory.
    diff_img = ImageChops.difference(im1, im2)

    return np.array(diff_img)


def background_reduction(target_image, background):
    gray1 = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(target_image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    inverted_mask = cv2.bitwise_not(mask)

    background = cv2.subtract(target_image, mask)
    foreground = cv2.subtract(target_image, inverted_mask)

    return cv2.addWeighted(foreground, 1, background, 0.25, 0)


def find_matches(img1, img2, resize_factor):
    img1_rs = cv2.resize(img1, (0, 0), fx=resize_factor, fy=resize_factor)
    img2_rs = cv2.resize(img2, (0, 0), fx=resize_factor, fy=resize_factor)

    sift_detector = cv2.SIFT_create()
    kp1, des1 = sift_detector.detectAndCompute(img1_rs, None)
    kp2, des2 = sift_detector.detectAndCompute(img2_rs, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Filter out poor matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    matches = good_matches

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    return points1, points2, img1_rs.shape[0], img1_rs.shape[1]


def equalize_hist_RGB(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def diff_alignment(img1_file: str, img2_file: str, method: DiffMethod = DiffMethod.background_reduction,  resize_factor: float = 0.5, verbose: bool = False):
    img1 = cv2.imread(img1_file, cv2.IMREAD_COLOR)  # referenceImage
    img2 = cv2.imread(img2_file, cv2.IMREAD_COLOR)  # sensedImage

    img1_blur = cv2.blur(img1, (3, 3))
    img2_blur = cv2.blur(img2, (3, 3))

    points1, points2, low_height, low_width = find_matches(img1_blur, img2_blur, resize_factor)

    if len(points1) < 10:
        # Try full size
        resize_factor = 1
        points1, points2, low_height, low_width = find_matches(img1_blur, img2_blur, resize_factor)

    if len(points1) < 4:
        print("found only 4 matches for img: ", img1_file, " ref: ", img2_file)
        return None

    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Get low-res and high-res sizes
    height, width, _ = img1.shape
    low_size = np.float32([[0, 0], [0, low_height], [low_width, low_height], [low_width, 0]])
    high_size = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

    # Compute scaling transformations
    scale_up = cv2.getPerspectiveTransform(low_size, high_size)
    scale_down = cv2.getPerspectiveTransform(high_size, low_size)

    #  Combine the transformations.
    h_and_scale_up = np.matmul(scale_up, H)
    scale_down_h_scale_up = np.matmul(h_and_scale_up, scale_down)

    # Warp image 1 to align with image 2
    img1Reg = cv2.warpPerspective(
        img1.astype(float),
        scale_down_h_scale_up,
        (img2.shape[1], img2.shape[0]),
        borderValue=-1)

    # Mask reference image to have sam shape as aligned image
    _, mask = cv2.threshold(img1Reg[:, :, 0], -1, 255, cv2.THRESH_BINARY)
    img2_masked = cv2.bitwise_and(img2, img2, mask=mask.astype(np.uint8))
    img1Reg = img1Reg.astype(np.uint8)
    img1Reg = cv2.bitwise_and(img1Reg, img1Reg, mask=mask.astype(np.uint8))

    # img1Reg = equalize_hist_RGB(img1Reg)
    # img2_masked = equalize_hist_RGB(img2_masked)

    if method == DiffMethod.diff:
        diff_img = diff(img1Reg, img2_masked)
    elif method == DiffMethod.background_reduction:
        diff_img = background_reduction(img2_masked, img1Reg)


    if verbose:
        f, ax = plt.subplots(3, 2, figsize=(15, 15))
        ax[0, 0].imshow(img1)
        ax[0, 0].set_title("Input")

        ax[0, 1].imshow(img2)
        ax[0, 1].set_title("Reference")

        ax[1, 0].imshow(img1Reg)
        ax[1, 0].set_title("Aligned Img")

        ax[1, 1].imshow(img2_masked)
        ax[1, 1].set_title("Masked Reference")

        ax[2, 0].imshow(diff_img)
        ax[2, 0].set_title("diff aligned")

        diff_orig, diff_orig_ratio = diff(Image.fromarray(img1), Image.fromarray(img2))
        ax[2, 1].imshow(diff_orig)
        ax[2, 1].set_title("diff without alignment")
        plt.show()

    return diff_img


def get_reference_file(dir, reference_file_prefix):
    for root, dirs, files in os.walk(dir):
        for filename in files:
            if reference_file_prefix in filename:
                return os.path.join(root, filename)
    return None


def mkdirs_if_not_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':

    args = parseargs()

    for root, dirs, files in tqdm(list(os.walk(args.input_path))):
        ref_file = get_reference_file(root, args.reference_file_prefix)

        if ref_file is not None:
            diff_img_dir = os.path.join(args.output_path, os.path.basename(root))
            mkdirs_if_not_exists(diff_img_dir)

            for filename in files:
                abs_filename = os.path.join(root, filename)

                if abs_filename != ref_file:
                    diff_img = diff_alignment(ref_file, abs_filename, args.diff_method, args.resize_factor, args.verbose)
                    if diff_img is not None:
                        cv2.imwrite(os.path.join(diff_img_dir, filename), diff_img)
