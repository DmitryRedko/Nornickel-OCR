import re
import json


# Step 1: Parse JSON Document
def parse_json_document(document):
    """
    Parses a structured JSON document into a list of sections with content.
    """
    sections = []
    for section, paragraphs in document.items():
        full_text = " ".join(paragraphs)
        sections.append((section, full_text))
    return sections


# Step 2: Generate BIO Tags
def generate_bio_tags_from_sections(sections, subtitle_regex):
    """
    Converts sections into BIO-tagged data for subtitles.

    Parameters:
        sections (list of tuples): List of (section_title, text).
        subtitle_regex (str): Regex pattern to identify subtitles.

    Returns:
        list of tuples: Each tuple contains a word and its BIO tag.
    """
    bio_data = []
    for section, text in sections:
        matches = list(re.finditer(subtitle_regex, text))
        current_idx = 0

        for match in matches:
            start, end = match.span()


            pre_text = text[current_idx:start].strip()
            if pre_text:
                bio_data.extend([(word, "O") for word in pre_text.split()])

            subtitle = match.group(0)
            bio_data.append((subtitle, "B-SUBTITLE"))

            current_idx = end

        remaining_text = text[current_idx:].strip()
        if remaining_text:
            bio_data.extend([(word, "O") for word in remaining_text.split()])

    return bio_data