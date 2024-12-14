from collections import Counter
from typing import List

import supervision as sv
from surya.layout import batch_layout_detection
from surya.schema import LayoutBox, LayoutResult
from ultralytics import YOLO

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


def surya_layout(images: list, pages: List[Page], layout_model, batch_multiplier=1):
    text_detection_results = [p.text_lines for p in pages]
    processor = layout_model.processor
    layout_results = batch_layout_detection(
        images,
        layout_model,
        processor,
        detection_results=text_detection_results,
        batch_size=int(get_batch_size() * batch_multiplier),
    )

    assert len(pages) == len(
        layout_results
    ), "Mismatched number of pages and layout results"

    yolo_model = YOLO("yolov11x_best.pt")
    yolo_detections = []

    for image in images:
        result = yolo_model(image)[0]
        detections = sv.Detections.from_ultralytics(result)
        yolo_detections.append(detections)

    for page, layout_result, yolo_detection in zip(
        pages, layout_results, yolo_detections
    ):
        updated_bboxes = []
        for layout_box in layout_result.bboxes:
            if layout_box.label == "Figure":
                for yolo_box, yolo_label in zip(
                    yolo_detection.xyxy, yolo_detection.data["class_name"]
                ):
                    if yolo_label == "Picture":
                        new_layout_box = LayoutBox(
                            polygon=[
                                [yolo_box[0], yolo_box[1]],
                                [yolo_box[2], yolo_box[1]],
                                [yolo_box[2], yolo_box[3]],
                                [yolo_box[0], yolo_box[3]],
                            ],
                            confidence=layout_box.confidence,
                            label=layout_box.label,
                            bbox=yolo_box,
                        )
                        updated_bboxes.append(new_layout_box)
                        break
            else:
                updated_bboxes.append(layout_box)

        layout_result_updated = LayoutResult(
            bboxes=updated_bboxes,
            segmentation_map=layout_result.segmentation_map,
            heatmaps=None,
            image_bbox=layout_result.image_bbox,
        )
        page.layout = layout_result_updated


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
