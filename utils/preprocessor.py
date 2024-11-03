from text.docx_predprocessor import DocProcessor
from text.pdf_predprocessor import PDFProcessor
from tables.tables_predprocessor import PDFTableExtractor
import os
import json
import PyPDF2

class Preprocessor:
    def __init__(self, output_path):
        self.output_path = output_path
        self.doc_parser = DocProcessor()
        self.pdf_processor = PDFProcessor(output_path=output_path)
        self.pdf_table_extractor = PDFTableExtractor()

    def process_file(self, file_path):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        output_file = self.output_path+'/output.json'

        extension = os.path.splitext(file_path)[1].lower()

        if extension in ['.doc', '.docx']:
            return self.doc_parser.parse(file_path)
        elif extension == '.pdf':
            page_count = self.get_pdf_page_count(file_path)
            flag_table_continous = 0

            # Initialize the JSON file
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write('{')

            for page in range(0, page_count):  
                pdf_parser = PDFProcessor(output_path=self.output_path)
                text = pdf_parser.process_pdf_page(file_path, page + 1, page_count)

                table_extractor = PDFTableExtractor()
                tables_list = table_extractor.extract_tables_from_pdf(file_path, page + 1, page_count, combine_tables=True)

                page_info = {
                    "file_path": file_path,
                    "page_number": page + 1,
                    "text": text,
                    "tables": tables_list
                }

                for i in range(len(tables_list)):
                    if flag_table_continous == 1:
                        flag_table_continous = 0                    
                        continue

                    for j in range(len(tables_list[i]['page_coords'])):
                        image_path = self.output_path+f'/tables/page_{page + 1}_table_{i + 1}_{j + 1}.png'
                        table_extractor.extract_table_image(file_path, page + 1 + j, tables_list[i]['page_coords'][j]).save(image_path)

                    if len(tables_list[i]['page_coords']) > 1:
                        flag_table_continous = 1
                
                # Append page info to the JSON file
                with open(output_file, 'a', encoding='utf-8') as file:
                    if page == 0:
                        file.write(f'"{page + 1}": {json.dumps(page_info, ensure_ascii=False, indent=4)}')
                    else:
                        file.write(f', "{page + 1}": {json.dumps(page_info, ensure_ascii=False, indent=4)}')

            # Close the JSON object
            with open(output_file, 'a', encoding='utf-8') as file:
                file.write('}')

            print(f'All data saved in file: {output_file}')

        else:
            raise ValueError(f"Unsupported file extension: {extension}")

    def get_pdf_page_count(self, file_path):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return len(reader.pages)
