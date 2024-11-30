import os
import json
import shutil
import time



def search_keys_in_json(json_data, search_strings):

    if isinstance(json_data, dict):
        for key in json_data:
            if any(search_str in key for search_str in search_strings):
                return True

            if search_keys_in_json(json_data[key], search_strings):
                return True
    elif isinstance(json_data, list):
        for item in json_data:
            if search_keys_in_json(item, search_strings):
                return True
    return False



def process_json_files(source_folder, target_folder, search_strings):

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)


    for filename in os.listdir(source_folder):

        if filename.endswith('.json'):
            file_path = os.path.join(source_folder, filename)
            retries = 3
            for attempt in range(retries):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        json_data = json.load(file)


                        if search_keys_in_json(json_data, search_strings):

                            shutil.copy(file_path, os.path.join(target_folder, filename))
                            print(f"Copied file: {filename}")
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        print(f"Error processing file {filename}: {e}. Retrying...")
                        time.sleep(1)
                    else:
                        print(f"Failed to process file {filename} after {retries} attempts: {e}")

if __name__ == "__main__":
    source_folder = "<YOUR_PATH>"
    target_folder = "<YOUR_PATH>" # Replace with the path to your target folder
    search_strings = ['Предисловие', 'Предисловие']  # Replace with the list of strings to search for in keys

    process_json_files(source_folder, target_folder, search_strings)
