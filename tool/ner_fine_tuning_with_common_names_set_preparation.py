import spacy
from spacy.tokens import Span
from names_matcher import get_complete_data_about_novel
from data_generator import generate_sample_test_data
from file_and_directory_management import read_file_to_list, write_text_to_file, read_sentences_from_file
import os
import random
import copy
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_common_names():
    common_names = read_file_to_list(ROOT_DIR + "\\data\\common_names.txt")
    return common_names


def get_names_to_be_replaced(characters):
    names_to_be_replaced = [characters[0]]
    if " " in characters[0]:
        names_to_be_replaced.extend(characters[0].split(" "))
    return names_to_be_replaced


# extracting sentences from novels for fine-tuning ner by injecting common names
def extract_sentences_for_names_injection(titles, number_of_sentences):
    sentences_per_novel = number_of_sentences/len(titles)
    nlp = spacy.load("en_core_web_sm")

    for title in titles:
        characters, novel_text = get_complete_data_about_novel(title)
        names_to_be_replaced = get_names_to_be_replaced(characters)
        doc = nlp(novel_text)
        potential_sentences = []

        for sentence in doc.sents:
            if any(name in sentence.text for name in names_to_be_replaced):
                potential_sentences.append(sentence.text)

        selected_sentences = [sent for sent in random.sample(potential_sentences,
                                                             k=min(int(sentences_per_novel), len(potential_sentences)))]
        test_sample = "\n".join(selected_sentences)

        write_text_to_file(ROOT_DIR + "\\data\\fine_tuning_ner\\common_names\\extracted_sentences\\" + title,
                           test_sample)


def inject_common_names(common_names, sentences, names_to_be_replaced):
    nlp = spacy.load("en_core_web_sm")
    updated_sentences = []
    result = []

    for sentence in sentences:
        for name in names_to_be_replaced:
            if name in sentence:
                common_name = random.choice(common_names)
                sentence = sentence.replace(name, common_name)

        updated_sentences.append(sentence)
        doc = nlp(sentence)
        dict = {}
        entities = []

        for ent in doc.ents:
            span = Span(doc, ent.start, ent.end, label=ent.label_)
            doc.ents = [span if e == ent else e for e in doc.ents]
            entities.append([ent.start_char, ent.end_char, ent.label_])

        dict["content"] = doc.text
        dict["entities"] = list(entities)
        result.append(dict)

    return result, updated_sentences


def main():
    titles = read_file_to_list(ROOT_DIR + "\\data\\titles.txt")
    common_names = get_common_names()
    # number_sentences_to_extracted = 3 * len(common_names)
    # extract_sentences_for_names_injection(titles, number_sentences_to_extracted)
    sentences = []
    names_to_be_replaced = []
    for title in titles:
        data = read_sentences_from_file(
            ROOT_DIR + "\\data\\fine_tuning_ner\\common_names\\extracted_sentences\\" + title)
        sentences.extend(data)
        characters, _ = get_complete_data_about_novel(title)
        names_to_be_replaced.extend(get_names_to_be_replaced(characters))

    training_set, updated_sentences = inject_common_names(common_names, sentences, names_to_be_replaced)
    with open(ROOT_DIR + "\\data\\fine_tuning_ner\\common_names\\" + "training_set.json", 'w') as result:
        json.dump(training_set, result)
    print("One done!")


if __name__ == "__main__":
    main()