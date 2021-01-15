import os
import re
import spacy
import json
from spacy.tokens import Span
from file_and_directory_management import read_file_to_list, read_sentences_from_file
from data_generator import json_to_spacy_train_data, spacy_format_to_json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def generalize_tags(data):
    return re.sub(r"([0-9]+,\s[0-9]+,\s')[a-zA-Z\s\.àâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]+", r"\1PERSON", str(data))


def generate_generalized_data(titles):
    for title in titles:
        test_data = json_to_spacy_train_data(ROOT_DIR + "\\data\\manually_annotated\\character_names_tags\\" + title + ".json")
        generalized_test_data = generalize_tags(test_data)
        spacy_format_to_json(generalized_test_data, title)


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


def main():
    titles = read_file_to_list(ROOT_DIR + "\\data\\titles.txt")
    # generate_generalized_data(titles)

    for title in titles:
        test_data = read_sentences_from_file(ROOT_DIR + "\\data\\testing_set\\" + title)
        ner_result = test_ner(test_data, ROOT_DIR + "\\results\\fine_tuning_ner\\model")
        with open(ROOT_DIR + "\\results\\fine_tuning_ner\\general_tag_PERSON\\" + title, 'w') as result:
            json.dump(ner_result, result)


if __name__ == "__main__":
    main()