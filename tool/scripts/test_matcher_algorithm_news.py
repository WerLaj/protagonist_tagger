from tabulate import tabulate

from tool.names_matcher import NamesMatcher
from tool.file_and_directory_management import read_file_to_list_polish, read_file_polish


def run_matcher(news_files_base_name, model_path, characters_list_path, num_of_news, texts_dir_path, results_dir_path, precision=75, tests_variant=True):
    names_matcher = NamesMatcher(precision, model_path)
    characters = read_file_to_list_polish(characters_list_path)

    for i in range(1, num_of_news + 1):
        text = read_file_polish(texts_dir_path + news_files_base_name + str(i))
        matches_table = names_matcher.match_names_for_text(characters,
                                                           [text],
                                                           results_dir_path,
                                                           news_files_base_name + str(i),
                                                           tests_variant,
                                                           save_ratios=True)

    print(tabulate(matches_table, tablefmt='orgtbl'))


if __name__ == "__main__":
    dir_with_news = "C:\\Users\\werka\\Desktop\\REFSA\\ISWC paper\\github\\protagonist_tagger\\data\\news\\"
    news_files_base_name = "doc"
    ner_model = "pl_core_news_sm"
    number_of_news = 100
    run_matcher(news_files_base_name,
                ner_model,
                dir_with_news + "all_full_names",
                number_of_news,
                dir_with_news + "plain_news\\",
                dir_with_news + "texts_annotated_by_tool\\")