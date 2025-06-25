import cv2
import os
import random

IMAGE_DIR = "test_dataset/images"
LABEL_DIR = "test_dataset/labels"
OUTPUT_IMAGE_DIR = "cropped_dataset/images"
OUTPUT_LABEL_DIR = "cropped_dataset/labels"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)


def load_ball_bboxes(label_path):
    ball_bboxes = []
    with open(label_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id == 1:
                x, y, w, h = map(float, parts[1:])
                ball_bboxes.append((x, y, w, h))
    return ball_bboxes


def get_random_point_in_bbox(x, y, w, h):
    x_min = x - w / 2
    x_max = x + w / 2
    y_min = y - h / 2
    y_max = y + h / 2
    rand_x = random.uniform(x_min, x_max)
    rand_y = random.uniform(y_min, y_max)
    return rand_x, rand_y


def compute_crop_size(y_norm, base_crop=200):
    factor = 0.5 + y_norm
    return int(base_crop * factor)


def crop_image(image, center_x, center_y, crop_size):
    h, w, _ = image.shape
    cx = int(center_x * w)
    cy = int(center_y * h)
    half_crop = crop_size // 2

    x1 = max(0, cx - half_crop)
    y1 = max(0, cy - half_crop)
    x2 = min(w, cx + half_crop)
    y2 = min(h, cy + half_crop)

    cropped = image[y1:y2, x1:x2]

    crop_info = {
        'x1': x1, 'y1': y1,
        'width': x2 - x1,
        'height': y2 - y1
    }

    return cropped, crop_info


def save_new_label(crop_info, point_x, point_y, output_label_path, orig_w, orig_h):
    crop_w, crop_h = crop_info['width'], crop_info['height']
    abs_x = point_x * orig_w
    abs_y = point_y * orig_h

    rel_x = (abs_x - crop_info['x1']) / crop_w
    rel_y = (abs_y - crop_info['y1']) / crop_h

    box_size = 20
    new_w = box_size / crop_w
    new_h = box_size / crop_h

    with open(output_label_path, "w") as f:
        f.write(f"1 {rel_x:.6f} {rel_y:.6f} {new_w:.6f} {new_h:.6f}\n")


def process_image_label_pair(image_path, label_path, output_img_path, output_lbl_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading {image_path}")
        return

    orig_h, orig_w = image.shape[:2]
    ball_bboxes = load_ball_bboxes(label_path)

    if not ball_bboxes:
        print(f"No ball label found in {label_path}")
        return

    x, y, w, h = ball_bboxes[0]
    rand_x, rand_y = get_random_point_in_bbox(x, y, w, h)
    crop_size = compute_crop_size(rand_y)

    cropped_img, crop_info = crop_image(image, rand_x, rand_y, crop_size)
    cv2.imwrite(output_img_path, cropped_img)
    save_new_label(crop_info, rand_x, rand_y, output_lbl_path, orig_w, orig_h)


def main():
    for filename in os.listdir(IMAGE_DIR):
        lower_name = filename.lower()
        if not lower_name.endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(IMAGE_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        label_filename = base_name + ".txt"
        label_path = os.path.join(LABEL_DIR, label_filename)

        image_path = os.path.normpath(image_path)
        label_path = os.path.normpath(label_path)

        if not os.path.exists(label_path):
            print(f"No label for {filename}")
            continue

        output_img_path = os.path.join(OUTPUT_IMAGE_DIR,
                                       filename.replace(".jpg", "_crop.jpg").replace(".png", "_crop.png"))
        output_lbl_path = os.path.join(OUTPUT_LABEL_DIR,
                                       filename.replace(".jpg", "_crop.txt").replace(".png", "_crop.txt"))

        process_image_label_pair(image_path, label_path, output_img_path, output_lbl_path)
        print(f"Processed: {filename}")


if __name__ == "__main__":
    main()
