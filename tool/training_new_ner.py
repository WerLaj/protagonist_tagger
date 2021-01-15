from __future__ import unicode_literals, print_function
import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from data_generator import json_to_spacy_train_data
from file_and_directory_management import read_file_to_list
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def prepare_training_data():
    titles = read_file_to_list(ROOT_DIR + "\\data\\titles.txt")
    train_data = []

    for title in titles:
        data_slice = json_to_spacy_train_data(ROOT_DIR + "\\data\\fine_tuning_ner\\not_recognized_entities\\training_set\\" + title)
        train_data.extend(data_slice)

    data_second_set = json_to_spacy_train_data(ROOT_DIR + "\\data\\fine_tuning_ner\\common_names\\" + "training_set.json")
    train_data.extend(data_second_set)

    return train_data


def fine_tune_ner(output_dir):
    train_data = prepare_training_data()
    n_iter = 100
    nlp = spacy.load("en_core_web_sm")

    # if "ner" not in nlp.pipe_names:
    #     ner = nlp.create_pipe("ner")
    #     nlp.add_pipe(ner, last=True)
    # else:
    #     ner = nlp.get_pipe("ner")
    # for _, annotations in train_data:
    #     for ent in annotations.get("entities"):
    #         ner.add_label(ent[2])

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        warnings.filterwarnings("once", category=UserWarning, module='spacy')
        for itn in range(n_iter):
            random.shuffle(train_data)

            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    for text, _ in train_data:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    if output_dir is not None:
        nlp.to_disk(output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in train_data:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


def main():
    # fine_tune_ner(ROOT_DIR + "\\results\\fine_tuning_ner\\not_recognized_entities\\model_common_names")
    fine_tune_ner(ROOT_DIR + "\\results\\fine_tuning_ner\\common_names\\model")


if __name__ == "__main__":
    main()


# import random
# import warnings
# from pathlib import Path
# import spacy
# from spacy.util import minibatch, compounding
# import os
# import json
# from spacy.scorer import Scorer
# from spacy.gold import GoldParse
# from data_generator import json_to_spacy_train_data
#
# # https://stackoverflow.com/questions/50644777/understanding-spacys-scorer-output
# # https://github.com/explosion/spaCy/issues/5318
# # https://stackoverflow.com/questions/44827930/evaluation-in-a-spacy-ner-model
#
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#
#
# def train_model_with_new_labels(nlp, train_data, n_iter=30):
#     if "ner" not in nlp.pipe_names:
#         ner = nlp.create_pipe("ner")
#         nlp.add_pipe(ner)
#     else:   # otherwise, get it, so we can add labels to it
#         ner = nlp.get_pipe("ner")
#
#     for data in train_data:
#         for ent in data[1]['entities']:
#             ner.add_label(ent[2])   # add new entity label to entity recognizer
#
#     optimizer = nlp.begin_training()
#     move_names = list(ner.move_names)
#     # get names of other pipes to disable them during training
#     pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
#     other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
#     # only train NER
#     with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
#         # show warnings for misaligned entity spans once
#         warnings.filterwarnings("once", category=UserWarning, module='spacy')
#
#         sizes = compounding(1.0, 4.0, 1.001)
#         # batch up the examples using spaCy's minibatch
#         for itn in range(n_iter):
#             random.shuffle(train_data)
#             batches = minibatch(train_data, size=sizes)
#             losses = {}
#             for batch in batches:
#                 texts, annotations = zip(*batch)
#                 nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
#             print("Losses", losses)
#
#     return nlp
#
#
# def save_trained_model(model, output_dir, new_model_name, move_names):
#     output_dir = Path(output_dir)
#     if not output_dir.exists():
#         output_dir.mkdir()
#     model.meta["name"] = new_model_name  # rename model
#     model.to_disk(output_dir)
#     print("Saved model to", output_dir)
#
#     print("Loading from", output_dir)   # test the saved model
#     nlp2 = spacy.load(output_dir)
#     # Check the classes have loaded back consistently
#     assert nlp2.get_pipe("ner").move_names == move_names
#
#
# def test_trained_model(model, test_data):
#     scorer = Scorer()
#     for text, annot in test_data:
#         # doc_gold_text = model.make_doc(text)
#         # gold = GoldParse(doc_gold_text, entities=annot)
#         pred_value = model(text)
#         # scorer.score(pred_value, gold)
#
#         print("Entities in '%s'" % text)
#         for ent in pred_value.ents:
#             print(ent.label_, ent.text)
#
#     # return scorer.scores
#
#
# def main():
#     title = "Pride_and_Prejudice"
#     nlp = spacy.load("en_core_web_sm")
#
#     train_data = json_to_spacy_train_data(ROOT_DIR + "\\results\\testing_set\\" + title)
#     model = train_model_with_new_labels(nlp, train_data, 3)
#     test_data = json_to_spacy_train_data(ROOT_DIR + "\\results\\testing_set\\test_pride")
#     test_trained_model(model, test_data)
#     # scores = test_trained_model(model, test_data)
#     #
#     # print(scores)
#
#
# if __name__ == "__main__":
#     main()
