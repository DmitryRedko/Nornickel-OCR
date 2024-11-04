import cv2
import numpy as np
from PIL import Image as PILImage

from pypdfium2 import PdfDocument
from tabled.inference.detection import merge_tables

from surya.input.pdflines import get_page_text_lines

from marker.pdf.images import render_image
from marker.schema.bbox import rescale_bbox
from marker.schema.page import Page
from typing import List
from marker.schema.block import Line, Span, Block
from marker.settings import settings


def get_table_boxes(pages: List[Page], doc: PdfDocument, fname):
    table_imgs = []
    table_counts = []
    table_bboxes = []
    img_sizes = []
    pnums = []

    for page_idx, page in enumerate(pages):
        # The bbox for the entire table
        bbox = [b.bbox for b in page.layout.bboxes if b.label == "Table"]
        highres_img = render_image(doc[page_idx], dpi=settings.SURYA_TABLE_DPI)

        page_table_imgs = []
        page_bboxes = []

        # Merge tables that are next to each other
        bbox = merge_tables(bbox)
        bbox = list(filter(lambda b: b[3] - b[1] > 10 and b[2] - b[0] > 10, bbox))

        if len(bbox) == 0:
            table_counts.append(0)
            img_sizes.append(None)
            pnums.append(page.pnum)
            continue

        # Number of tables per page
        table_counts.append(len(bbox))
        img_sizes.append(highres_img.size)
        pnums.append(page.pnum)

        for bb in bbox:
            highres_bb = rescale_bbox(page.layout.image_bbox, [0, 0, highres_img.size[0], highres_img.size[1]], bb)
            page_table_imgs.append(highres_img.crop(highres_bb))
            page_bboxes.append(highres_bb)

        table_imgs.extend(page_table_imgs)
        table_bboxes.extend(page_bboxes)

    # The page number in doc and in the original document are not the same
    # Doc has had pages removed from the start to align to start_page
    # This corrects for that
    doc_idxs = [pnum for pnum, tc in zip(pnums, table_counts) if tc > 0]
    table_idxs = [i for i, tc in enumerate(table_counts) if tc > 0]
    sel_text_lines = get_page_text_lines(
        fname,
        doc_idxs,
        [hr for i, hr in enumerate(img_sizes) if i in table_idxs],
        # Add flatten pdf here
    )
    text_lines = []
    out_img_sizes = []
    for i in range(len(table_counts)):
        if i in table_idxs:
            page_ocred = pages[i].ocr_method is not None
            if page_ocred:
                # This will force re-detection of cells if the page was ocred (the text lines are not accurate)
                text_lines.extend([None] * table_counts[i])
            else:
                text_lines.extend([sel_text_lines.pop(0)] * table_counts[i])
            out_img_sizes.extend([img_sizes[i]] * table_counts[i])

    assert len(table_imgs) == len(table_bboxes) == len(text_lines) == len(out_img_sizes)
    assert sum(table_counts) == len(table_imgs)

    return table_imgs, table_bboxes, table_counts, text_lines, out_img_sizes

def detect_and_remove_tables(images: np.array, pages, doc, fname) -> PILImage:
        """
        Обнаруживает таблицы на изображении и маскирует их.
        Возвращает изображение с замаскированными таблицами.
        """

        table_imgs, table_boxes, table_counts, table_text_lines, img_sizes = get_table_boxes(pages, doc, fname)

        res_images = []
        table_count = 0
        for page_idx, image in enumerate(images):

            page_table_count = table_counts[page_idx]
            if page_table_count == 0:
                continue

            table_insert_points = {}

            blocks_to_remove = set()
            pnum = pages[page_idx].pnum
            highres_size = img_sizes[table_count]
            page_table_boxes = table_boxes[table_count:table_count + page_table_count]

            for table_idx, table_box in enumerate(page_table_boxes):
                lowres_table_box = rescale_bbox([0, 0, highres_size[0], highres_size[1]], pages[page_idx].bbox, table_box)

                for block_idx, block in enumerate(pages[page_idx].blocks):
                    intersect_pct = block.intersection_pct(lowres_table_box)
                    if intersect_pct > settings.TABLE_INTERSECTION_THRESH and block.block_type == "Table":
                        if table_idx not in table_insert_points:
                            table_insert_points[table_idx] = max(0, block_idx - len(blocks_to_remove)) # Where to insert the new table
                        blocks_to_remove.add(block_idx)

            new_page_blocks = []
            for block_idx, block in enumerate(pages[page_idx].blocks):
                if block_idx in blocks_to_remove:
                    continue
                new_page_blocks.append(block)

            for table_idx, table_box in enumerate(page_table_boxes):
                if table_idx not in table_insert_points:
                    table_count += 1
                    continue
                table_block = Block(
                    bbox=table_box,
                    block_type="Table",
                    pnum=pnum,
                    lines=[Line(
                        bbox=table_box,
                        spans=[Span(
                            bbox=table_box,
                            span_id=f"{table_idx}_table",
                            font="Table",
                            font_size=0,
                            font_weight=0,
                            block_type="Table",
                            text=''
                        )]
                    )]
                )
                insert_point = table_insert_points[table_idx]
                insert_point = min(insert_point, len(new_page_blocks))
                new_page_blocks.insert(insert_point, table_block)

            pages[page_idx].blocks = new_page_blocks

            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 9
            )

            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

            table_mask = cv2.add(horizontal_lines, vertical_lines)

            contours, _ = cv2.findContours(
                table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            mask = np.ones_like(gray) * 255

            min_table_area = 1000
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_table_area:
                    cv2.drawContours(mask, [contour], -1, 0, -1)

            result = cv2.bitwise_and(gray, gray, mask=mask)

            res_images+=[PILImage.fromarray(result)]
        
        return res_images
        