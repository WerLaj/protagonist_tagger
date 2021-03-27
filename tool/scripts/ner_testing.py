import sys
import re
import spacy
import os
import json
from spacy.tokens import Span
from tool.file_and_directory_management import read_file_to_list, read_sentences_from_file
from tool.data_generator import json_to_spacy_train_data, spacy_format_to_json


def generalize_tags(data):
    return re.sub(r"([0-9]+,\s[0-9]+,\s')[a-zA-Z\s\.àâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]+", r"\1PERSON", str(data))


# generalizing annotations - changing tags containing full names of literary characters to general tag PERSON
# titles - list of novels titles to be inlcluded in the generated data set (titles should not contain any special
#       characters and spaces should be replaced with "_", for example "Pride_andPrejudice")
# names_gold_standard_dir_path - path to directory with .txt files containing gold standard with annotations being full
#       names of literary characters (names of files should be the same as corresponding novels titles on the titles
#       list)
# generated_data_dir - directory where generated data should be stored
def generate_generalized_data(titles, names_gold_standard_dir_path, generated_data_dir):
    for title in titles:
        test_data = json_to_spacy_train_data(names_gold_standard_dir_path + title + ".json")
        generalized_test_data = generalize_tags(test_data)
        spacy_format_to_json(generated_data_dir + "generated_gold_standard\\", generalized_test_data, title)


def test_ner(data, model_dir=None):
    if model_dir is not None:
        nlp = spacy.load(model_dir)
    else:
        nlp = spacy.load("en_core_web_sm")
    result = []
    for sentence in data:
        doc = nlp(sentence)
        dict = {}
        entities = []
        for index, ent in enumerate(doc.ents):
            if ent.label_ == "PERSON":
                span = Span(doc, ent.start, ent.end, label="PERSON")
                doc.ents = [span if e == ent else e for e in doc.ents]
                entities.append([ent.start_char, ent.end_char, "PERSON"])

        dict["content"] = doc.text
        dict["entities"] = entities
        result.append(dict)

    return result


# titles_path - path to .txt file with titles of novels from which the sampled data are to be generated (titles should
#       not contain any special characters and spaces should be replaced with "_", for example "Pride_andPrejudice")
# names_gold_standard_dir_path - path to directory with .txt files containing gold standard with annotations being full
#       names of literary characters (names of files should be the same as corresponding novels titles on the titles
#       list)
# generated_data_dir - directory where generated data should be stored
# testing_data_dir_path - directory containing .txt files with sentences extrated from novels to be included in the
#       testing process
# ner_model_dir_path - path to directory containing fine-tune NER model to be tested; if None standard spacy NER
#       model is used
def main(titles_path, names_gold_standard_dir_path, testing_data_dir_path, generated_data_dir, ner_model_dir_path=None):
    titles = read_file_to_list(titles_path)
    generate_generalized_data(titles, names_gold_standard_dir_path, generated_data_dir)

    for title in titles:
        test_data = read_sentences_from_file(testing_data_dir_path + title)
        ner_result = test_ner(test_data, ner_model_dir_path)

        path = generated_data_dir + "ner_model_annotated\\" + title

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, 'w+') as result:
            json.dump(ner_result, result)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

# eg.
# python -m tool.scripts.ner_testing C:\Users\werka\Desktop\testing\small_set.txt C:\Users\werka\Desktop\testing\n
#   ames_gold_standard\ C:\Users\werka\Desktop\testing\test_small\ C:\Users\werka\Desktop\testing\my_results\ C:\Users\
#   werka\Desktop\testing\fine_tuned_ner_model
