from utils.preprocessor import Preprocessor
import os

def process_docs_folder(folder_path):
    # Проверяем и создаем папку main_result, если её нет
    main_result_path = 'main_result'
    if not os.path.exists(main_result_path):
        os.mkdir(main_result_path)
    
    # Проходимся по каждому файлу в папке DOCS
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        
        if os.path.isfile(file_path):
            print(file_name)
            doc_folder = os.path.join(main_result_path, os.path.splitext(file_name)[0])
            os.makedirs(doc_folder, exist_ok=True)
            
           
            os.makedirs(os.path.join(doc_folder, 'tables'), exist_ok=True)
            os.makedirs(os.path.join(doc_folder, 'images'), exist_ok=True)

            preprocessor = Preprocessor(doc_folder)
            preprocessor.process_file(file_path)

if __name__ == "__main__":
    docs_folder = 'DOCS'  
    process_docs_folder(docs_folder)
