import sys

from tool.ner_metrics import characters_tags_metrics, ner_metrics


# titles_path - path to .txt file with titles of novels which should be included in the analysis (titles should
#       not contain any special characters and spaces should be replaced with "_", for example "Pride_andPrejudice")
# gold_standard_dir_path - path to directory containing gold standard for the testing set (names of files should be
#       the same as titles on the list)
# testing_set_dir_path - path to directory containing testing sets (selected sentences) extracted from novels, each in
#       file with the same name as corresponding title of the novel
# stats_dir - directory where the computed metrics should be stored
# protagonist_tagger - if true metrics are calculated for sets annotated with names of literary characters (computing
#       metrics for protagonistTagger performance); if false metrics are calculated for general tag person (computing
#       metrics for NER model performance)
def main(titles_path, gold_standard_dir_path, testing_set_dir_path, stats_dir, protagonist_tagger=True):
    if protagonist_tagger:
        characters_tags_metrics(titles_path, gold_standard_dir_path, testing_set_dir_path, stats_dir)
    else:
        ner_metrics(titles_path, gold_standard_dir_path, testing_set_dir_path, stats_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

# eg.
# python -m tool.scripts.compute_metrics C:\Users\werka\Desktop\testing\small_set.txt C:\Users\werka\Desktop\testi
#   ng\test_small_names_gold_standard\ C:\Users\werka\Desktop\testing\annotated_test_small_names\ C:\Users\werka\Des
#   ktop\testing\results\ True
