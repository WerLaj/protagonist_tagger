from file_and_directory_management import read_file_to_list
import os
from names_matcher import NamesMatcher
from tabulate import tabulate

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_matcher(title, testing_string, precision):
    names_matcher = NamesMatcher(precision)
    matches_table = names_matcher.matcher_test(title, testing_string, title, displacy_option=True)

    return matches_table


def run_matcher(precision):
    names_matcher = NamesMatcher(precision)
    titles = read_file_to_list(ROOT_DIR.replace("\\scripts", "") + "\\data\\additional_titles.txt")
    for title in titles:
        matches_table = names_matcher.match_names_for_text(title, title, descriptions_variant=False, tests_variant=True,
                                                           displacy_option=False, save_ratios=False, save_doc=False)

    print(tabulate(matches_table, tablefmt='orgtbl'))


def main():
    run_matcher(75)


if __name__ == "__main__":
    main()