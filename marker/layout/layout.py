from collections import Counter
from typing import List

import supervision as sv
from surya.layout import batch_layout_detection
from surya.schema import LayoutBox, LayoutResult
from ultralytics import YOLO

import cv2
import supervision as sv
import numpy as np
import os

from marker.schema.bbox import rescale_bbox
from marker.schema.block import bbox_from_lines
from marker.schema.page import Page
from marker.settings import settings


def get_batch_size():
    if settings.LAYOUT_BATCH_SIZE is not None:
        return settings.LAYOUT_BATCH_SIZE
    elif settings.TORCH_DEVICE_MODEL == "cuda":
        return 6
    return 6

def save_fullscreen_images(images, output):
    output_dir = os.path.join(output, "fullscreen_images")
    os.makedirs(output_dir, exist_ok=True)  # Создаёт папку, если её нет

    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f"page_{i + 1}_fullscreen.png")
        image.save(output_path)
        print(f"Fullscreen image saved to: {output_path}")

def annotate_yolo_images(images, output):
    yolo_model = YOLO("yolov11x_best.pt")
    output_dir = os.path.join(output, "yolo_images")
    os.makedirs(output_dir, exist_ok=True)  # Создаёт папку, если её нет

    for idx, image in enumerate(images):
        # Convert image to OpenCV format if needed
        if isinstance(image, str):
            image_cv = cv2.imread(image)  # Если это путь, загружаем напрямую
        else:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL.Image to OpenCV image

        results = yolo_model(image_cv, conf=0.2, iou=0.8)[0]
        detections = sv.Detections.from_ultralytics(results)

        class_colors = [
            sv.Color(255, 0, 0),    # Red for "Caption"
            sv.Color(0, 255, 0),    # Green for "Footnote"
            sv.Color(0, 0, 255),    # Blue for "Formula"
            sv.Color(255, 255, 0),  # Yellow for "List-item"
            sv.Color(255, 0, 255),  # Magenta for "Page-footer"
            sv.Color(0, 255, 255),  # Cyan for "Page-header"
            sv.Color(128, 0, 128),  # Purple for "Picture"
            sv.Color(128, 128, 0),  # Olive for "Section-header"
            sv.Color(128, 128, 128),# Gray for "Table"
            sv.Color(0, 128, 128),  # Teal for "Text"
            sv.Color(128, 0, 0)     # Maroon for "Title"
        ]

        box_annotator = sv.BoxAnnotator(color=sv.ColorPalette(class_colors), thickness=3)
        annotated_image = box_annotator.annotate(scene=image_cv, detections=detections)

        label_annotator = sv.LabelAnnotator(color=sv.ColorPalette(class_colors), text_color=sv.Color(255, 255, 255))
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        output_path = os.path.join(output_dir, f"annotated_image_{idx + 1}.png")
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to: {output_path}")
        
def surya_layout(images: list, pages: List[Page], layout_model, batch_multiplier=1, image_flag=0, output = './'):
    text_detection_results = [p.text_lines for p in pages]

    processor = layout_model.processor
    layout_results = batch_layout_detection(images, layout_model, processor, detection_results=text_detection_results, batch_size=int(get_batch_size() * batch_multiplier))
    for page, layout_result in zip(pages, layout_results):
        page.layout = layout_result

    if image_flag == 1:
        # Fullscreen image saving logic
        save_fullscreen_images(images, output)
    elif image_flag == 2:
        # YOLO model-based image annotation and saving
        annotate_yolo_images(images, output)
    elif image_flag == 3:
        # Fullscreen and YOLO image saving logic
        save_fullscreen_images(images, output)
        annotate_yolo_images(images, output)   


def annotate_block_types(pages: List[Page]):
    for page in pages:
        max_intersections = {}
        for i, block in enumerate(page.blocks):
            for j, layout_block in enumerate(page.layout.bboxes):
                layout_bbox = layout_block.bbox
                layout_bbox = rescale_bbox(
                    page.layout.image_bbox, page.bbox, layout_bbox
                )
                intersection_pct = block.intersection_pct(layout_bbox)
                if i not in max_intersections:
                    max_intersections[i] = (intersection_pct, j)
                elif intersection_pct > max_intersections[i][0]:
                    max_intersections[i] = (intersection_pct, j)

        for i, block in enumerate(page.blocks):
            block = page.blocks[i]
            block_type = None
            if i in max_intersections and max_intersections[i][0] > 0.0:
                j = max_intersections[i][1]
                block_type = page.layout.bboxes[j].label
            block.block_type = block_type

        # Smarter block layout assignment - first assign same as closest block
        # Next, fall back to text
        for i, block in enumerate(page.blocks):
            if block.block_type is not None:
                continue
            min_dist = None
            min_dist_idx = None
            for j, block2 in enumerate(page.blocks):
                if j == i or block2.block_type is None:
                    continue
                dist = block.distance(block2.bbox)
                if min_dist_idx is None or dist < min_dist:
                    min_dist = dist
                    min_dist_idx = j
                for line in block2.lines:
                    dist = block.distance(line.bbox)
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_idx = j

            if min_dist_idx is not None:
                block.block_type = page.blocks[min_dist_idx].block_type

        for i, block in enumerate(page.blocks):
            if block.block_type is None:
                block.block_type = settings.DEFAULT_BLOCK_TYPE

        def get_layout_label(block_labels: List[str]):
            counter = Counter(block_labels)
            return counter.most_common(1)[0][0]

        def generate_block(block, block_labels):
            block.bbox = bbox_from_lines(block.lines)
            block.block_type = get_layout_label(block_labels)
            return block

        # Merge blocks together, preserving pdf order
        curr_layout_idx = None
        curr_layout_block = None
        curr_block_labels = []
        new_blocks = []
        for i in range(len(page.blocks)):
            if i not in max_intersections or max_intersections[i][0] == 0:
                if curr_layout_block is not None:
                    new_blocks.append(
                        generate_block(curr_layout_block, curr_block_labels)
                    )
                curr_layout_block = None
                curr_layout_idx = None
                curr_block_labels = []
                new_blocks.append(page.blocks[i])
            elif max_intersections[i][1] != curr_layout_idx:
                if curr_layout_block is not None:
                    new_blocks.append(
                        generate_block(curr_layout_block, curr_block_labels)
                    )
                curr_layout_block = page.blocks[i].copy()
                curr_layout_idx = max_intersections[i][1]
                curr_block_labels = [page.blocks[i].block_type]
            else:
                curr_layout_block.lines.extend(page.blocks[i].lines)
                curr_block_labels.append(page.blocks[i].block_type)

        if curr_layout_block is not None:
            new_blocks.append(generate_block(curr_layout_block, curr_block_labels))

        page.blocks = new_blocks
