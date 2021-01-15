from ner_metrics import overall_metrics, metrics_per_novel, ner_metrics, characters_tags_metrics


def main():
    # metrics_per_novel("\\data\\manually_annotated\\character_names_tags\\",
    #                   "\\results\\testing_set\\",
    #                   "\\results\\stats\\")
    # overall_metrics("\\data\\manually_annotated\\character_names_tags\\",
    #                 "\\results\\testing_set\\",
    #                 "\\results\\stats\\")
    # characters_tags_metrics("\\data\\manually_annotated\\character_names_tags\\",
    #                 "\\results\\testing_set\\",
    #                 "\\results\\stats\\")
    characters_tags_metrics("\\data\\additional_testing_set\\manually_annotated\\character_names_tags\\",
                    "\\results\\testing_set_2\\",
                    "\\results\\stats_2\\")
    # ner_metrics("\\data\\manually_annotated\\general_tag_PERSON\\",
    #             "\\results\\pretrained_ner\\general_tag_PERSON\\",
    #             "\\results\\pretrained_ner\\stats\\")
    # ner_metrics("\\data\\manually_annotated\\general_tag_PERSON\\",
    #             "\\results\\fine_tuning_ner\\not_recognized_entities\\general_tag_PERSON\\",
    #             "\\results\\fine_tuning_ner\\not_recognized_entities\\stats\\")
    # ner_metrics("\\data\\additional_testing_set\\manually_annotated\\general_tag_PERSON\\",
    #             "\\results\\additional_testing_set\\fine_tuned_ner\\common_names\\general_tag_PERSON\\",
    #             "\\results\\additional_testing_set\\fine_tuned_ner\\common_names\\stats\\")
    # ner_metrics("\\data\\additional_testing_set\\manually_annotated\\general_tag_PERSON\\",
    #             "\\results\\additional_testing_set\\pretrained_ner\\general_tag_PERSON\\",
    #             "\\results\\additional_testing_set\\pretrained_ner\\stats\\")


if __name__ == "__main__":
    main()
