import sys

from tool.data_generator import generate_sample_test_data
from tool.file_and_directory_management import read_file_to_list


# titles_path - path to .txt file with titles of novels from which the sampled data are to be generated (titles should
#       not contain any special characters and spaces should be replaced with "_", for example "Pride_andPrejudice")
# novels_texts_dir_path - path to directory with .txt files containing full texts of novels (names of files should be
#       the same as titles on the list from titles_path)
# number_of_sentences - umber of sentences containing at least one named entity recognized as PERSON by
#       standard NER model to be randomly extracted from each novel
# generated_data_dir - directory where generated data should be stored
def main(titles_path, novels_texts_dir_path, number_of_sentences, generated_data_dir):
    titles = read_file_to_list(titles_path)

    generate_sample_test_data(titles, number_of_sentences, novels_texts_dir_path, generated_data_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])

# eg. of execution from protagonist_tagger directory
#   python -m tool.scripts.generate_test_data C:\Users\werka\Desktop\testing\small_set.txt C:\Users\werka\Desktop\te
#       sting\small_set\ 20 C:\Users\werka\Desktop\testing\results\
