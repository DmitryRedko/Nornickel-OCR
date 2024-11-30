import re


def split_text_by_paragraphs(text,pattern=r'(?<!\S)(\d{1,2}(?:\.\d{1,2}){1,3})(?!\S)' ):
    """
    text: словарь из json
    pattern: регулярка
    """
    matches = list(re.finditer(pattern, text))

    if not matches:
        return [text.strip()]

    split_points = [0]
    split_points.extend(match.start() for match in matches)
    split_points.append(len(text))

    chunks = [
        text[split_points[i]:split_points[i + 1]].strip()
        for i in range(len(split_points) - 1)
    ]


    return [chunk for chunk in chunks if chunk]



def process_document(doc):
    chunks = []
    for key, values in doc.items():
        for value in values:
            # Split each value based on the paragraph pattern
            paragraphs = split_text_by_paragraphs(value)
            chunks.extend(paragraphs)
    return chunks
