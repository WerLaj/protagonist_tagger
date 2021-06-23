import json
from tool.file_and_directory_management import read_file_polish, write_text_to_file, read_file_to_list_polish


def convert_news_data_to_correct_format(path_to_dir, file_name):
    annotated_news_text = read_file_polish(path_to_dir + file_name)
    annotations_count = annotated_news_text.count("<Entity name=")
    # [m.start() for m in re.finditer("<Entity name=", annotated_news_text)]
    text_without_annotations = ""
    last_analysed_index = 0
    entities = []
    full_names = set()
    for i in range(0, annotations_count):
        entity_index = annotated_news_text.find("<Entity name=")
        full_name_end_index = annotated_news_text.find(" type=")
        full_name = annotated_news_text[(entity_index + len("<Entity name=\"")):(full_name_end_index-1)]
        annotation_details_end_index = annotated_news_text.find("\">") + len("\">")
        annotation_closure_index = annotated_news_text.find("</Entity>")
        text_without_annotations = text_without_annotations + annotated_news_text[last_analysed_index:entity_index]
        text_without_annotations = text_without_annotations + annotated_news_text[annotation_details_end_index:annotation_closure_index]
        last_analysed_index = len(text_without_annotations)
        entity = [entity_index, entity_index + (annotation_closure_index - annotation_details_end_index), full_name]
        entities.append(entity)
        annotated_news_text = text_without_annotations + annotated_news_text[(annotation_closure_index + len("</Entity>")):]
        full_names.add(full_name)

    text_without_annotations = text_without_annotations + annotated_news_text[last_analysed_index:]

    # print(text_without_annotations)
    # for entity in entities:
    #     print(entity)

    dict = {}
    dict["content"] = text_without_annotations
    dict["entities"] = list(entities)

    return text_without_annotations, dict, list(full_names)


def prepare_and_save_to_files_news_data(dir_with_news, news_files_base_name, number_of_news):
    for i in range(1, number_of_news + 1):
        file_name = news_files_base_name + str(i)
        plain_text, annotated_news, full_names = convert_news_data_to_correct_format(dir_with_news + "original_data\\", file_name)
        write_text_to_file(dir_with_news + "plain_news\\" + file_name, plain_text)
        with open(dir_with_news + "gold_standard\\" + file_name + ".json", 'w+',  encoding="utf-8") as result:
            json.dump([annotated_news], result, ensure_ascii=False)
        write_text_to_file(dir_with_news + "full_names\\" + file_name, '\n'.join(full_names))


def get_all_tags_appearing_in_news(dir_with_files_listing_full_names, news_files_base_name, num_of_news):
    all_tags = set()
    for i in range(1, num_of_news + 1):
        tags_in_news = read_file_to_list_polish(dir_with_files_listing_full_names + "full_names\\" + news_files_base_name + str(i))
        all_tags.update(tags_in_news)

    all_tags = list(all_tags)

    write_text_to_file(dir_with_news + "all_full_names", '\n'.join(all_tags))

    return all_tags


if __name__ == "__main__":
    dir_with_news = "C:\\Users\\werka\\Desktop\\REFSA\\ISWC paper\\github\\protagonist_tagger\\data\\news\\"
    news_files_base_name = "doc"
    number_of_news = 100

    prepare_and_save_to_files_news_data(dir_with_news, news_files_base_name, number_of_news)

    get_all_tags_appearing_in_news(dir_with_news, news_files_base_name, number_of_news)



