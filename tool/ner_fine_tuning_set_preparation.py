import spacy
from spacy.tokens import Span
from names_matcher import get_complete_data_about_novel
from file_and_directory_management import read_file_to_list, write_text_to_file, read_sentences_from_file
import os
import random
import copy
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_not_recognized_entities(title):
    named_entities = read_file_to_list(ROOT_DIR + "\\data\\fine_tuning_ner\\not_recognized_entities\\not_recognized_ner\\" + title)
    return named_entities


def create_ner_training_set(title, named_entities, sentences_per_entity):
    nlp = spacy.load("en_core_web_sm")
    _, novel_text = get_complete_data_about_novel(title)
    doc = nlp(novel_text)
    potential_sentences = {}
    temporal_named_entities = copy.copy(named_entities)

    for entity in named_entities:
        potential_sentences[entity] = list()

    for sentence in doc.sents:
        if len(temporal_named_entities) == 0:
            break
        for entity in temporal_named_entities:
            if entity in sentence.text:
                current_list = potential_sentences[entity]
                current_list.append(sentence.text)
                potential_sentences[entity] = current_list
                if len(current_list) > sentences_per_entity:
                    temporal_named_entities.remove(entity)
                break

    selected_sentences = []
    for entity_potential_sentences in [potential_sentences[entity] for entity in named_entities]:
        count = min(sentences_per_entity, len(entity_potential_sentences))
        selected_sentences.extend([sent for sent in random.sample(entity_potential_sentences, k=count)])

    test_sample = "\n".join(selected_sentences)
    write_text_to_file(ROOT_DIR + "\\data\\fine_tuning_ner\\not_recognized_entities\\extracted_sentences\\" + title, test_sample)

    return selected_sentences


def get_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def overlap(intervals, interval):
    for inter in intervals:
        if get_overlap(inter, interval) > 0:
            return True
    return False


def annotate_training_set(data, names_entities):
    nlp = spacy.load("en_core_web_sm")
    label = "PERSON"
    result = []
    for sentence in data:
        doc = nlp(sentence)
        dict = {}
        entities = []
        intervals = []

        for entity in names_entities:
            if entity in sentence:
                start_index = sentence.index(entity)
                end_index = start_index + len(entity)
                if not overlap(intervals, [start_index, end_index]):
                    entities.append([start_index, end_index, label])
                    intervals.append([start_index, end_index])

        for index, ent in enumerate(doc.ents):
            span = Span(doc, ent.start, ent.end, label=ent.label_)
            doc.ents = [span if e == ent else e for e in doc.ents]

            if not overlap(intervals, [ent.start_char, ent.end_char]):
                entities.append([ent.start_char, ent.end_char, ent.label_])

        dict["content"] = doc.text
        dict["entities"] = list(entities)
        result.append(dict)

    return result


def main():
    titles = read_file_to_list(ROOT_DIR + "\\data\\titles.txt")
    for title in titles:
        named_entites = get_not_recognized_entities(title)
        # ner_training_set = create_ner_training_set(title, named_entites, 5)
        data = read_sentences_from_file(ROOT_DIR + "\\data\\fine_tuning_ner\\not_recognized_entities\\extracted_sentences\\" + title)
        training_set = annotate_training_set(data, named_entites)
        with open(ROOT_DIR + "\\data\\fine_tuning_ner\\not_recognized_entities\\training_set\\" + title, 'w') as result:
            json.dump(training_set, result)
        print("One done!")


if __name__ == "__main__":
    main()
