import os
import random
import spacy
import json

from names_matcher import NamesMatcher, get_complete_data_about_novel, get_data_about_novel
from file_and_directory_management import write_text_to_file
from wiki_scanner import get_descriptions_of_characters, get_list_of_characters

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_lists_of_characters(titles):
    for title in titles:
        get_list_of_characters(title)


def generate_descriptions_of_characters(titles):
    for title in titles:
        get_descriptions_of_characters(title)


def generate_sample_test_data(titles, number_of_sentences):
    nlp = spacy.load("en_core_web_sm")

    for title in titles:
        _, novel_text = get_complete_data_about_novel(title)
        doc = nlp(novel_text)
        potential_sentences = []
        people = set()

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                people.add(ent.text)

        for sentence in doc.sents:
            if any(person in sentence.text for person in people):
                potential_sentences.append(sentence.text)

        selected_sentences = [sent for sent in random.sample(potential_sentences, k=number_of_sentences)]
        test_sample = "\n".join(selected_sentences)

        write_text_to_file(ROOT_DIR + "\\data\\testing_set\\" + title, test_sample)


def create_ner_person_lists(titles):
    nlp = spacy.load("en_core_web_sm")

    for title in titles:
        _, novel_text = get_complete_data_about_novel(title)
        doc = nlp(novel_text)
        people = set()
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                people.add(ent.text)

        people_list = "\n".join(people)

        write_text_to_file(ROOT_DIR + "\\data\\ner_person\\" + title, people_list)


def json_to_spacy_train_data(path):
    with open(path, encoding='utf-8') as train_data:
        train = json.load(train_data)

    train_data = []
    for data in train:
        ents = [tuple(entity) for entity in data['entities']]
        train_data.append((data['content'], {'entities': ents}))

    return train_data


def spacy_format_to_json(data, title):
    eval_data = list(eval(data))
    json_data = []

    for sentence in eval_data:
        dict = {"content": sentence[0], "entities": sentence[1]['entities']}
        json_data.append(dict)

    with open(ROOT_DIR + "\\data\\manually_annotated\\general_tag_PERSON\\" + title, 'w') as result:
        json.dump(json_data, result)


def data_from_json(path):
    with open(path, encoding='utf-8') as train_data:
        train = json.load(train_data)

    entities = []
    contents = []
    for data in train:
        entities.append(data['entities'])
        contents.append(data['content'])

    return entities, contents