import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tool.data_generator import data_from_json
import tabulate
from tool.file_and_directory_management import read_file_to_list
import pickle


def save_to_pickle(filename, data, stats_path):
    pickle_out = open(stats_path + filename, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def load_from_pickle(filename, stats_path):
    file = open(stats_path + filename, "rb")
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
    # entities_gold, _ = data_from_json(gold_standard_path + title + ".json")
    entities_gold, _ = data_from_json(gold_standard_path + title)
    entities_matcher, _ = data_from_json(result_path + title)

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
        # entities_gold, _ = data_from_json(gold_standard_path + title + ".json")
        entities_gold, _ = data_from_json(gold_standard_path + title)
        entities_matcher, _ = data_from_json(result_path + title)

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


def metrics_per_novel(titles_path, gold_standard_path, result_path, stats_path):
    titles = read_file_to_list(titles_path)
    for title in titles:
        create_and_save_stats(title, gold_standard_path, result_path, stats_path)

    for title in titles:
        metrics = load_from_pickle(title, stats_path)
        print("*****************")
        print(title)
        print(tabulate.tabulate(metrics[1:], headers=metrics[0], tablefmt='orgtbl'))
        print("*****************")


def overall_metrics(titles_path, gold_standard_path, result_path, stats_path):
    titles = read_file_to_list(titles_path)
    metrics = create_overall_stats(titles, gold_standard_path, result_path, stats_path)
    print(tabulate.tabulate(metrics[1:], headers=metrics[0], tablefmt='orgtbl'))


def characters_tags_metrics(titles_path, gold_standard_path, result_path, stats_path):
    titles = read_file_to_list(titles_path)
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


def ner_metrics(titles_path, gold_standard_path, result_path, stats_path):
    titles = read_file_to_list(titles_path)
    for title in titles:
        create_and_save_stats(title, gold_standard_path, result_path, stats_path, ner=True)

    metrics_table = []
    headers = ["Novel title", "precision", "recall", "F-measure", "support"]

    for title in titles:
        metrics = load_from_pickle(title, stats_path)
        metrics_title = [title].__add__([m[0] for m in metrics])
        metrics_table.append(metrics_title)

    metrics = create_overall_stats(titles, gold_standard_path, result_path, stats_path, ner=True)
    metrics_table.append(["*** overall results ***"].__add__([m[0] for m in metrics]))
    print(tabulate.tabulate(metrics_table, headers=headers, tablefmt='latex'))
