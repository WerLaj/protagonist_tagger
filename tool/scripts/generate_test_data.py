import os

from data_generator import generate_sample_test_data, create_ner_person_lists
from file_and_directory_management import read_file_to_list

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    titles = read_file_to_list(ROOT_DIR.replace("\\scripts", "") + "\\data\\additional_titles.txt")

    generate_sample_test_data(titles, 150)
    create_ner_person_lists(titles)

    # displacy.serve(doc, style="ent")


if __name__ == "__main__":
    main()