import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from data_generator import data_from_json
import tabulate
import os
from file_and_directory_management import read_file_to_list
import pickle

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_to_pickle(filename, data, stats_path):
    pickle_out = open(ROOT_DIR + stats_path + filename, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def load_from_pickle(filename, stats_path):
    file = open(ROOT_DIR + stats_path + filename, "rb")
    data = pickle.load(file)
    return data


def organize_entities(entities_gold, entities_matcher):
    gold = []
    matcher = []

    for index, entities in enumerate(entities_gold):
        gold_temp = list(entity[2] for entity in entities)
        matcher_temp = list(entity[2] for entity in entities_matcher[index])
        gold.extend(gold_temp)
        matcher.extend(matcher_temp)
        missing_ents = np.abs(len(gold_temp) - len(matcher_temp))
        filler = [''] * missing_ents

        if len(gold_temp) > len(matcher_temp):
            matcher.extend(filler)
        else:
            gold.extend(filler)

    return gold, matcher


# def calculate_metrics(gold, matcher, overall=False):
#     characters = list(dict.fromkeys(gold + matcher))
#     characters.remove('')
#     if not overall:
#         result = precision_recall_fscore_support(np.array(gold), np.array(matcher), labels=(characters))
#     result_micro = precision_recall_fscore_support(np.array(gold), np.array(matcher), beta=0.8, average='micro')
#     result_macro = precision_recall_fscore_support(np.array(gold), np.array(matcher), beta=0.8, average='macro')
#     result_weighted = precision_recall_fscore_support(np.array(gold), np.array(matcher), beta=0.8, average='weighted')
#
#     result_micro = [np.round(a, 2) for a in result_micro[0:3]]
#     result_macro = [np.round(a, 2) for a in result_macro[0:3]]
#     result_weighted = [np.round(a, 2) for a in result_weighted[0:3]]
#
#     metrics = list()
#     if overall:
#         headers = ["***"] + ["micro"] + ["macro"] + ["weighted"]
#     else:
#         headers = ["***"] + ["micro"] + ["macro"] + ["weighted"] + characters
#     metrics.append(headers)
#
#     if overall:
#         metrics.append(["precision"] + [result_micro[0]] + [result_macro[0]] + [result_weighted[0]])
#         metrics.append(["recall"] + [result_micro[1]] + [result_macro[1]] + [result_weighted[1]])
#         metrics.append(["fbeta"] + [result_micro[2]] + [result_macro[2]] + [result_weighted[2]])
#     else:
#         metrics.append(["precision"] + [result_micro[0]] + [result_macro[0]] + [result_weighted[0]] + list(result[0]))
#         metrics.append(["recall"] + [result_micro[1]] + [result_macro[1]] + [result_weighted[1]] + list(result[1]))
#         metrics.append(["fbeta"] + [result_micro[2]] + [result_macro[2]] + [result_weighted[2]] + list(result[2]))
#         metrics.append(["support"] + [" "] * 3 + list(result[3]))
#
#     return metrics


def calculate_metrics_ner(gold, matcher):
    characters = list(dict.fromkeys(gold + matcher))
    characters.remove('')

    result = precision_recall_fscore_support(np.array(gold), np.array(matcher), labels=(characters))

    support = result[3]
    result = [np.round(a, 2) for a in result[0:3]]
    result.append(support)

    return result


def calculate_metrics(gold, matcher):
    characters = list(dict.fromkeys(gold + matcher))
    characters.remove('')

    result = precision_recall_fscore_support(np.array(gold), np.array(matcher), labels=(characters), average='weighted')

    result = [np.round(a, 2) for a in result[0:3]]

    return result


def create_and_save_stats(title, gold_standard_path, result_path, stats_path, ner=False):
    entities_gold, _ = data_from_json(ROOT_DIR + gold_standard_path + title + ".json")
    # entities_gold, _ = data_from_json(ROOT_DIR + gold_standard_path + title)
    entities_matcher, _ = data_from_json(ROOT_DIR + result_path + title)

    gold, matcher = organize_entities(entities_gold, entities_matcher)

    if ner is True:
        metrics = calculate_metrics_ner(gold, matcher)
    else:
        metrics = calculate_metrics(gold, matcher)

    save_to_pickle(title, metrics, stats_path)


def create_overall_stats(titles, gold_standard_path, result_path, stats_path, ner=False):
    gold_overall = []
    matcher_overall = []

    for title in titles:
        entities_gold, _ = data_from_json(ROOT_DIR + gold_standard_path + title + ".json")
        # entities_gold, _ = data_from_json(ROOT_DIR + gold_standard_path + title)
        entities_matcher, _ = data_from_json(ROOT_DIR + result_path + title)

        gold, matcher = organize_entities(entities_gold, entities_matcher)
        gold_overall.extend(gold)
        matcher_overall.extend(matcher)

    if ner is True:
        metrics = calculate_metrics_ner(gold_overall, matcher_overall)
    else:
        # metrics = calculate_metrics(gold_overall, matcher_overall, overall=True)
        metrics = calculate_metrics(gold_overall, matcher_overall)
    save_to_pickle("overall_metrics", metrics, stats_path)
    return metrics


def metrics_per_novel(gold_standard_path, result_path, stats_path):
    titles = read_file_to_list(ROOT_DIR + "\\data\\titles.txt")
    for title in titles:
        create_and_save_stats(title, gold_standard_path, result_path, stats_path)

    for title in titles:
        metrics = load_from_pickle(title, stats_path)
        print("*****************")
        print(title)
        print(tabulate.tabulate(metrics[1:], headers=metrics[0], tablefmt='orgtbl'))
        print("*****************")


def overall_metrics(gold_standard_path, result_path, stats_path):
    titles = read_file_to_list(ROOT_DIR + "\\data\\titles.txt")
    metrics = create_overall_stats(titles, gold_standard_path, result_path, stats_path)
    print(tabulate.tabulate(metrics[1:], headers=metrics[0], tablefmt='orgtbl'))


def characters_tags_metrics(gold_standard_path, result_path, stats_path):
    titles = read_file_to_list(ROOT_DIR + "\\data\\additional_titles.txt")
    for title in titles:
        create_and_save_stats(title, gold_standard_path, result_path, stats_path, ner=False)

    metrics_table = []
    headers = ["Novel title", "precision", "recall", "F-measure"]

    for title in titles:
        metrics = load_from_pickle(title, stats_path)
        metrics_title = [title].__add__([m for m in metrics])
        metrics_table.append(metrics_title)

    metrics = create_overall_stats(titles, gold_standard_path, result_path, stats_path, ner=False)
    metrics_table.append(["*** overall results ***"].__add__([m for m in metrics]))
    print(tabulate.tabulate(metrics_table, headers=headers, tablefmt='latex'))


def ner_metrics(gold_standard_path, result_path, stats_path):
    titles = read_file_to_list(ROOT_DIR + "\\data\\titles.txt")
    for title in titles:
        create_and_save_stats(title, gold_standard_path, result_path, stats_path, ner=True)

    metrics_table = []
    headers = ["Novel title", "precision", "recall", "F-measure", "support"]

    for title in titles:
        metrics = load_from_pickle(title, stats_path)
        metrics_title = [title].__add__([m[0] for m in metrics])
        metrics_table.append(metrics_title)

    titles = read_file_to_list(ROOT_DIR + "\\data\\titles.txt")
    metrics = create_overall_stats(titles, gold_standard_path, result_path, stats_path, ner=True)
    metrics_table.append(["*** overall results ***"].__add__([m[0] for m in metrics]))
    print(tabulate.tabulate(metrics_table, headers=headers, tablefmt='latex'))
