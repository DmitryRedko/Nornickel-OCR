from marker.images.save import get_image_filename
from marker.pdf.images import render_bbox_image
from marker.schema.bbox import rescale_bbox
from marker.schema.block import find_insert_block, Span, Line
from marker.settings import settings


def find_image_blocks(page, min_image_height, min_image_width):
    image_blocks = []
    page_width, page_height = page.bbox[2] - page.bbox[0], page.bbox[3] - page.bbox[1]
    if min_image_width<0 or min_image_width>1 or min_image_height<0 or min_image_height>1:
        raise ValueError("min_image_width and min_image_height should be in range(0,1)")
    
    min_width = (1-min_image_width) * page_width
    min_height = (1-min_image_height) * page_height
    
    image_regions = [l.bbox for l in page.layout.bboxes if l.label in ["Figure", "Picture"]]
    image_regions = [rescale_bbox(page.layout.image_bbox, page.bbox, b) for b in image_regions]
    
    # Filter image regions based on size
    image_regions = [region for region in image_regions if (region[2] - region[0] >= min_width and region[3] - region[1] >= min_height)]
    
    insert_points = {}
    for region_idx, region in enumerate(image_regions):
        for block_idx, block in enumerate(page.blocks):
            for line_idx, line in enumerate(block.lines):
                if line.intersection_pct(region) > settings.BBOX_INTERSECTION_THRESH:
                    line.spans = []  # We will remove this line from the block
                    
                    if region_idx not in insert_points:
                        insert_points[region_idx] = (block_idx, line_idx)
    
    # Account for images with no detected lines
    for region_idx, region in enumerate(image_regions):
        if region_idx in insert_points:
            continue
        
        insert_points[region_idx] = (find_insert_block(page.blocks, region), 0)
    
    for region_idx, image_region in enumerate(image_regions):
        image_insert = insert_points[region_idx]
        image_blocks.append([image_insert[0], image_insert[1], image_region])
    
    return image_blocks

def extract_page_images(page_obj, page, min_image_height, min_image_width):
    page.images = []
    image_blocks = find_image_blocks(page, min_image_height, min_image_width)

    for image_idx, (block_idx, line_idx, bbox) in enumerate(image_blocks):
        if block_idx >= len(page.blocks):
            block_idx = len(page.blocks) - 1
        if block_idx < 0:
            continue

        block = page.blocks[block_idx]
        image = render_bbox_image(page_obj, page, bbox)
        image_filename = get_image_filename(page, image_idx)
        image_markdown = f"\n\n![{image_filename}]({image_filename})\n\n"
        image_span = Span(
            bbox=bbox,
            text=image_markdown,
            font="Image",
            rotation=0,
            font_weight=0,
            font_size=0,
            image=True,
            span_id=f"image_{image_idx}"
        )

        # Sometimes, the block has zero lines
        if len(block.lines) > line_idx:
            block.lines[line_idx].spans.append(image_span)
        else:
            line = Line(
                bbox=bbox,
                spans=[image_span]
            )
            block.lines.append(line)
        page.images.append(image)

def extract_images(doc, pages, min_image_height, min_image_width):
    for page_idx, page in enumerate(pages):
        page_obj = doc[page_idx]
        extract_page_images(page_obj, page, min_image_height, min_image_width)
