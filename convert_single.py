import time

import pypdfium2 # Needs to be at the top to avoid warnings
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS

import argparse
from marker.convert import convert_single_pdf
from marker.logger import configure_logging
from marker.models import load_all_models

from marker.output import save_markdown

configure_logging()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="PDF file to parse")
    parser.add_argument("output", help="Output base folder path")
    parser.add_argument("--max_pages", type=int, default=None, help="Maximum number of pages to parse")
    parser.add_argument("--start_page", type=int, default=None, help="Page to start processing at")
    parser.add_argument("--langs", type=str, help="Optional languages to use for OCR, comma separated", default=None)
    parser.add_argument("--batch_multiplier", type=int, default=2, help="How much to increase batch sizes")
    parser.add_argument("--save_table", type=int, default=1, help="Inv value: (0 - don't save; 1-md only; 2-img only; 3-img+md)")
    parser.add_argument("--min_image_width", type=int, default=0.3, help="Minimal ratio of image in the comparison doc width")
    parser.add_argument("--min_image_height", type=int, default=0.2, help="Minimal ratio of image in the comparison doc height")

    args = parser.parse_args()

    langs = args.langs.split(",") if args.langs else None

    fname = args.filename
    base_fname = os.path.basename(fname).rsplit(".", 1)[0]

    os.makedirs(os.path.join(args.output, base_fname), exist_ok=True)
    os.makedirs(os.path.join(os.path.join(args.output, base_fname), 'tables'), exist_ok=True)

    model_lst = load_all_models()
    start = time.time()
    
    tables_path = os.path.join(os.path.join(args.output, base_fname), 'tables')
    print(tables_path)
    print(langs)
    full_text, images, out_meta = convert_single_pdf(
                                            fname, 
                                            model_lst, 
                                            max_pages=args.max_pages, 
                                            langs=langs, 
                                            batch_multiplier=args.batch_multiplier, 
                                            start_page=args.start_page,
                                            save_table = args.save_table,
                                            min_image_width = args.min_image_width,
                                            min_image_height = args.min_image_height,
                                            tables_path = tables_path
                                        )

    fname = os.path.basename(fname)
    subfolder_path = save_markdown(args.output, fname, full_text, images, out_meta)

    print(f"Saved markdown to the {subfolder_path} folder")
    print(f"Total time: {time.time() - start}")


if __name__ == "__main__":
    main()
