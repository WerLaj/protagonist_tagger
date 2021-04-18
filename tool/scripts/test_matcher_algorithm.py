from tabulate import tabulate
import sys

from tool.names_matcher import NamesMatcher
from tool.file_and_directory_management import read_file_to_list, read_file, read_sentences_from_file


def get_complete_data_about_novel(title, characters_lists_dir_path, novels_texts_dir_path):
    characters = read_file_to_list(characters_lists_dir_path + title)
    novel_text = read_file(novels_texts_dir_path + title)
    return characters, novel_text


def get_test_data_for_novel(title, characters_lists_dir_path, testing_sets_dir_path):
    characters = read_file_to_list(characters_lists_dir_path + title)
    text = read_sentences_from_file(testing_sets_dir_path + title)
    return characters, text


def test_matcher(title, testing_string, precision, model_path, characters_lists_dir_path, novels_texts_dir_path):
    characters, _ = get_complete_data_about_novel(title, characters_lists_dir_path, novels_texts_dir_path)

    names_matcher = NamesMatcher(precision, model_path)
    matches_table = names_matcher.matcher_test(characters, testing_string, title, displacy_option=True)

    return matches_table


# titles_path - path to .txt file with titles of novels from which the sampled data are to be generated (titles should
#       not contain any special characters and spaces should be replaced with "_", for example "Pride_andPrejudice")
# precision - precision of approximate string matching; values in between [1,100] (recomended ~75)
# test_variant - if True then separate sentences in testing set are considered and annotated separately; if False whole
#       text given is annotated as a whole (without splitting to sentences)
# model_path - path to a fine-tuned nlp spacy model to be loaded and used during named entity recognition process; if
#       not the standard, pre-trained NER model is used
# characters_lists_dir_path - directory of files containing lists of characters from corresponding novels (names of
#       files should be the same as titles on the list from titles_path)
# texts_dir_path - directory of files containing texts from corresponding novels to be annotated (names of
#       files should be the same as titles on the list from titles_path)
def run_matcher(titles_path, model_path, characters_lists_dir_path, texts_dir_path, results_dir, precision=75, tests_variant=True):
    names_matcher = NamesMatcher(precision, model_path)
    titles = read_file_to_list(titles_path)
    for title in titles:
        if tests_variant:
            characters, text = get_test_data_for_novel(title, characters_lists_dir_path, texts_dir_path)
        else:
            characters, text = get_complete_data_about_novel(title, characters_lists_dir_path, texts_dir_path)
        matches_table = names_matcher.match_names_for_text(characters,
                                                           text,
                                                           results_dir,
                                                           title,
                                                           tests_variant,
                                                           save_ratios=True)

    print(tabulate(matches_table, tablefmt='orgtbl'))


def main(titles_path, model_path, characters_lists_dir_path, texts_dir_path, results_dir):
    run_matcher(titles_path, model_path, characters_lists_dir_path, texts_dir_path, results_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
