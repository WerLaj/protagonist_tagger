import json
from fuzzywuzzy import fuzz
import spacy
from spacy import displacy, gold
from spacy.tokens import Span
import os
import itertools

from tool.file_and_directory_management import read_file_to_list, write_list_to_file, read_file, write_text_to_file, read_sentences_from_file
from tool.gender_checker import get_name_gender, get_personal_titles, create_titles_and_gender_dictionary
from tool.diminutives_recognizer import get_names_from_diminutive


def prepare_list_for_ratios(characters):
    ratios = []
    row = ["RECOGNIZED NAMED ENTITY", "MATCH"]
    for char in characters:
        row.append(char)
    ratios.append(row)

    return ratios


class NamesMatcher:
    def __init__(self, partial_ratio_precision, model_path="en_core_web_sm"):
        self.personal_titles = get_personal_titles()
        self.titles_gender_dict = create_titles_and_gender_dictionary()
        self.nlp = spacy.load(model_path)
        self.partial_ratio_precision = partial_ratio_precision

    def recognize_person_entities(self, text, characters):
        matches_table = prepare_list_for_ratios(characters)
        train_data = []

        doc = self.nlp(text)
        dict = {}
        entities = []
        for index, ent in enumerate(doc.ents):
            if ent.label_ == "PERSON":
                personal_title = self.recognize_personal_title(doc, index)
                person = ent.text
                row = self.find_match_for_person(person, personal_title, characters)
                if row is not None:
                    matches_table.append(row)
                    if row[1] is not None:
                        label = "{" + row[1] + "}"
                        span = Span(doc, ent.start, ent.end, label=label)
                        doc.ents = [span if e == ent else e for e in doc.ents]
                        entities.append([ent.start_char, ent.end_char, row[1]])

        dict["content"] = doc.text
        dict["entities"] = entities
        train_data.append(dict)

        return matches_table, train_data, doc

    def match_names_for_text(self, characters, text, results_dir, filename=None, tests_variant=False, displacy_option=False, save_ratios=False, save_doc=False):
        if tests_variant:
            train_data = []
            matches_table = prepare_list_for_ratios(characters)
            for sentence in text:
                matches_table_row, data_for_sentence, _ = self.recognize_person_entities(sentence, characters)
                train_data.append(data_for_sentence[0])
                matches_table.extend(matches_table_row[1:])
        else:
            matches_table, train_data, doc = self.recognize_person_entities(text, characters)

        if filename is not None:
            if save_doc:
                json_data = gold.docs_to_json(doc)
                with open(results_dir + "\\docs\\" + filename, 'w') as result:
                    json.dump(json_data, result)

            if save_ratios:
                write_list_to_file(results_dir + "\\ratios\\" + filename, matches_table)

            if tests_variant:
                with open(results_dir + filename, 'w', encoding='utf8') as result:
                    json.dump(train_data, result, ensure_ascii=False)
            else:
                with open(results_dir + filename, 'w') as result:
                    json.dump(train_data, result)

        if displacy_option:
            displacy.serve(doc, style="ent")

    def matcher_test(self, characters, testing_string, results_dir, filename=None, displacy_option=False):
        matches_table, train_data, doc = self.recognize_person_entities(testing_string, characters)

        if filename is not None:
            write_list_to_file(results_dir + "\\ratios\\" + filename + "_test", matches_table)
            with open(results_dir + filename + "_test_spacy.json", 'w') as result:
                json.dump(train_data, result)

        if displacy_option:
            sentence_spans = list(doc.sents)
            displacy.serve(sentence_spans, style="ent")

    def find_match_for_person(self, person, personal_title, characters):
        row_ratios = []
        potential_matches = []
        if "Miss " in person:
            person = person.replace("Miss ", "")
            personal_title = "Miss"

        for index, char in enumerate(characters):
            # partial_ratio = fuzz.partial_ratio(((personal_title + " ") if personal_title is not None else "") + person, char)
            # ratio = fuzz.ratio(((personal_title + " ") if personal_title is not None else "") + person, char)
            partial_ratio = self.get_partial_ratio_for_all_permutations(person, char)
            ratio = fuzz.ratio(((personal_title.replace(".", "") + " ") if personal_title is not None else "") + person, char)
            ratio_no_title = fuzz.ratio(person,                                char)
            if ratio == 100 or ratio_no_title == 100:
                potential_matches = [[char, ratio]]
                row_ratios = row_ratios + ["---" for i in range(0,len(characters)-index)]
                break
            if partial_ratio >= self.partial_ratio_precision:
                row_ratios.append("---" + str(partial_ratio) + "---")
                potential_matches.append([char, partial_ratio])
            else:
                row_ratios.append(str(partial_ratio))

        potential_matches = sorted(potential_matches, key=lambda x: x[1], reverse=True)
        row = [((personal_title + " ") if personal_title is not None else "") + str(person)]
        ner_match = self.choose_best_match(person, personal_title, potential_matches, characters)
        if ner_match is None:
            return None

        row.append(ner_match)
        row.extend(row_ratios)

        return row

    def get_partial_ratio_for_all_permutations(self, potential_match, character_name):
        character_name_components = character_name.split()
        character_name_permutations = list(itertools.permutations(character_name_components))
        partial_ratios = []
        for permutation in character_name_permutations:
            partial_ratios.append(fuzz.partial_ratio(' '.join(permutation), potential_match))

        return max(partial_ratios)

    def choose_best_match(self, person, personal_title, potential_matches, characters):
        if len(potential_matches) > 1:
            ner_match = self.handle_multiple_potential_matches(person, personal_title, potential_matches)
        elif len(potential_matches) == 1:
            ner_match = potential_matches[0][0]
        else:
            ner_match = "PERSON"
            potential_names_from_diminutive = get_names_from_diminutive(person)
            if potential_names_from_diminutive is not None:
                for char in characters:
                    for name in potential_names_from_diminutive:
                        if name in char.lower().split():
                            return char

        return ner_match

    def handle_multiple_potential_matches(self, person, personal_title, potential_matches):
        ner_match = None
        if personal_title is not None:
            if personal_title == "the":
                return "the " + person
            else:
                title_gender = self.titles_gender_dict[personal_title][0]
                for match in potential_matches:
                    if get_name_gender(match[0]) == title_gender:
                        ner_match = match[0]
                        break

        else: # todo handle Bennet sisters, daughters, etc.
            ner_match = potential_matches[0][0]

        return ner_match

    # gives personal title for name at index in doc; if there is no title in front of the name None is returned
    def recognize_personal_title(self, doc, index):
        personal_title = None
        span = doc.ents[index]
        if span.start > 0:
            word_before_name = doc[span.start - 1]
            if word_before_name.text.replace(".", "") in self.personal_titles:
                personal_title = word_before_name.text.replace(".", "")
            if word_before_name.text.lower() == "the":
                personal_title = "the"

        return personal_title
