
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Dict
import json


def create_splits(docs: Dict[str, Dict], embed_fun) -> List:
    text_splitter = SemanticChunker(embed_fun, breakpoint_threshold_type="standard_deviation")
    all_chunks = []

    for key in docs.keys():
        if "text" in docs[key]:
            # Generate initial chunks for each text in the JSON
            chunks = text_splitter.create_documents([docs[key]["text"]])
            all_chunks.extend(chunks)

    return all_chunks


def add_overlap(chunks: List, overlap: int = 10) -> List:
    overlapped_chunks = []

    for idx, chunk in enumerate(chunks):
        # Get current chunk content
        current_content = chunk.page_content

        # Add overlap from the end of the previous chunk if it exists
        if idx > 0:
            prev_content = chunks[idx - 1].page_content
            current_content = prev_content[-overlap:] + " " + current_content

        # Add overlap from the beginning of the next chunk if it exists
        if idx < len(chunks) - 1:
            next_content = chunks[idx + 1].page_content
            current_content = current_content + " " + next_content[:overlap]

        # Append the modified chunk with overlaps as a new Document
        overlapped_chunks.append(Document(page_content=current_content))

    return overlapped_chunks


def remove_overlap(chunks: List, overlap: int = 10) -> List:
    clean_chunks = []

    for idx, chunk in enumerate(chunks):
        content = chunk.page_content

        # Remove overlap at the beginning if it exists (from the previous chunk)
        if idx > 0:
            content = content[overlap:]  # Remove the first 'overlap' characters

        # Remove overlap at the end if it exists (from the next chunk)
        if idx < len(chunks) - 1:
            content = content[:-overlap]  # Remove the last 'overlap' characters
        content = content.strip()
        # Append cleaned chunk
        clean_chunks.append(Document(page_content=content))

    return clean_chunks

if __name__ == "__main__":
    # Initialize embedding function
    embed_fun = HuggingFaceEmbeddings(model_name="deepvk/USER-bge-m3")

    # Load JSON data
    with open("output212.json", "r") as f:
        docs = json.load(f)

    # Create initial chunks without overlap
    initial_chunks = create_splits(docs, embed_fun)

    # Add overlap between chunks
    final_chunks = add_overlap(initial_chunks, overlap=10)
    cleaned_chunks = remove_overlap(final_chunks, overlap=10)
    # Display results to verify overlap content
    # for idx, doc in enumerate(cleaned_chunks):
    #     print(f"Chunk {idx + 1}: {doc.page_content[:100]}...")  # Preview each chunk with overlap

    # Write the overlapped chunks to a file
    with open("chunks_std_212_nooverlaps.txt", "w+") as f:
        for doc in cleaned_chunks:
            f.write(doc.page_content + "\n\n")