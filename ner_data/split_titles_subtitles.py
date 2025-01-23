import re


def split_document_to_subsections(document, regex):
    """
    Сплитить джсоны
    """

    def split_to_subsections(text, pattern):

        matches = list(re.finditer(pattern, text))
        if not matches:
            return text.strip()

        subsections = {}
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            subtitle = match.group(0)

            subsection_text = text[start + len(subtitle):end].strip()
            subsections[subtitle] = subsection_text

        return subsections

    structured_result = {}
    for title, values in document.items():
        full_text = " ".join(values)
        subsections = split_to_subsections(full_text, regex)
        if isinstance(subsections, dict):
            structured_result[title] = subsections
        else:
            structured_result[title] = subsections

    return structured_result


if __name__ == "__main__":
    document = {
    'Введение': [
        'Настоящий свод правил разработан в развитие положений [1]. Требования к путям эвакуации и эвакуационным выходам, '
        'изложенные в нормативных документах по пожарной безопасности, разработанных для зданий определенного класса функциональной '
        'пожарной опасности, для подтверждения их соответствия положениям [1] следует выполнять наряду с требованиями настоящего свода '
        'правил, с учетом особенностей их функционального назначения и специфики противопожарной защиты.'
    ],
    '1 Область применения': [
        '1.1 Настоящий свод правил устанавливает требования пожарной безопасности к эвакуационным путям, эвакуационным и аварийным '
        'выходам из помещений, зданий и сооружений (далее - здания), а также требования пожарной безопасности к эвакуационным путям '
        'для наружных технологических установок. Требования свода правил распространяются на объекты защиты при их проектировании, '
        'изменении функционального назначения, а также при проведении работ по реконструкции, капитальном ремонте и техническому '
        'перевооружению.',
        '1.2 Настоящий свод правил не распространяется на здания и сооружения специального назначения (для производства, хранения, '
        'переработки и уничтожения радиоактивных и взрывчатых веществ, материалов и средств взрывания, военного назначения, подземные '
        'сооружения метрополитенов, горные выработки), жилые здания высотой более 75 м и иные здания высотой более 50 м, а также на '
        'здания с числом подвальных этажей более одного, за исключением случая, когда в указанных этажах размещаются части здания, '
        'требования к которым изложены в настоящем своде правил.',
        '1.3 При изменении функционального назначения существующих зданий или отдельных помещений в них, а также при изменении '
        'объемно-планировочных и конструктивных решений должны применяться требования настоящего свода правил в соответствии с новым '
        'назначением этих зданий или помещений.'
    ]
}
    # Define a regex to match subtitles like 1.1, 1.2
    regex_pattern = r'(?<!\S)(\d{1,2}(?:\.\d{1,2}){1,3})(?!\S)'

    # Split the document into structured subsections
    structured_result = split_document_to_subsections(document, regex_pattern)

    # Print the structured result
    for title, content in structured_result.items():
        print(f"Title: {title}")
        if isinstance(content, dict):
            for subtitle, text in content.items():
                print(f"  Subtitle: {subtitle}")
                print(f"  Text: {text}")
        else:
            print(f"  Text: {content}")
