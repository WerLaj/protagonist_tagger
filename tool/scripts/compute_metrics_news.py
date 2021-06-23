from tool.ner_metrics import characters_tags_metrics_news


if __name__ == "__main__":
    dir_with_news = "C:\\Users\\werka\\Desktop\\REFSA\\ISWC paper\\github\\protagonist_tagger\\data\\news\\"
    news_files_base_name = "doc"
    number_of_news = 100
    characters_tags_metrics_news(news_files_base_name, number_of_news, dir_with_news + "gold_standard\\", dir_with_news + "texts_annotated_by_tool\\", dir_with_news + "stats\\")
