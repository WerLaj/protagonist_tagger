# Protagonists' Catcher in Novels -- A Dataset and A Method
Semantic annotation of long texts, such as novels, remains an open challenge in the field of Semantic Web (SW) and Natural Language Processing (NLP). Recognizing and identifying literary characters in full-text novels is the first step for a more detailed literary text analysis. This research investigates the problem of ontology population, i.e. recognizing people (especially main characters) in novels and annotating them. We prepared a tool -- **protagonistTagger** -- for this annotation and a dataset to test it. 

Our process of identifying literary characters in a text, implemented in **protagonistTagger**, comprises two stages: (1) named entity recognition (NER) of persons, (2) matching method of each recognized person with the literary character's full name associated with it, based on **approximate text matching**. 

The performance of **protagonistTagger** in thirteen full-text novels shows that the tool achieved both precision and recall above 80\%. The test datasets comprise 1300 sentences from classic novels of different genres that a novel reader had annotated. 

Exemplary annotations (written in between --*assigned_tag*--) performed by **protagonistTagger**:
>"Her disappointment in **Charlotte --Charlotte Lucas--** made her turn with fonder regard to her sister, of whose rectitude and delicacy she was sure her opinion could never be shaken, and for whose happiness she grew daily more anxious, as **Bingley --Charles Bingley--** had now been gone a week and nothing more was heard of his return. **Jane --Jane Bennet--** had sent **Caroline --Caroline Bingley--** an early answer to her letter and was counting the days till she might reasonably hope to hear again. The promised letter of thanks from **Mr. Collins --Mr William Collins--** arrived on Tuesday, addressed to their father, and written with all the solemnity of gratitude which a twelvemonth’s abode in the family might have prompted."  

## General Project Workflow
The process of creating the corpus of annotated novels and the **protagonistTagger** tool comprises several stages:
- Gathering an initial corpus with plain novels' texts without annotations. 
- Creating a list of full names of all protagonists for each novel in the initial corpus. These names are the predefined tags that will be used in further steps for annotations.
- Recognizing named entity of category **person** in the texts of the novels in the initial corpus. Training NER model from scratch for this specific problem is not reasonable due to the amount of time and computing power it requires. It is possible to use some pre-trained NER model and fine-tune it using a sample of manually annotated data. The evaluation of the NER mechanism is done on a testing set extracted from the full texts of novels. The task is quite complex and may include several iterations.
- Each named entity of category **person**  recognized by NER in the previous step is a potential candidate to be annotated with one of our tags predefined in step 3. At this point, an algorithm (let us call it **matching algorithm** for reference), based on approximate string matching, is used to choose from the list of predefined tags the one that matches most accurately the recognized named entity. 
- The annotations done by the **matching algorithm** are accessed according to their accuracy and correctness.

The **protagonistTagger** (fine-tuned NER + matching algorithm) is used to annotate more novels in order to create the corpus of annotated novels. 

![alt text here](project_workflow.png)

## What can you find here
This repository comprises three main parts:
1. corpus of thirteen novels annotated with full names of protagonists (see *protagonist_tagger/annotated_corpus/*)
2. data set containing: 
    + the following  information about each novel:
		+ full plain text of novel (see *protagonist_tagger/data/complete_literary_texts/*)
		+ list of literary characters (see *protagonist_tagger/data/lists_of_characters/*)
		+ set of named entities of category person not recognized by standard NER model (see *protagonist_tagger/data/ner_training_sets/training_set_1/not_recognized_named_entities_person/*)
	+ testing sets (small and large) along with gold standards annotated with general tag PERSON and gold standards annotated with full names of literary characters (see *protagonist_tagger/data/testing_sets/*); additionaly both testing sets are annotated using fine-tuned NER model an **protagonistTagger** (see *protagonist_tagger/data/results*)
	+ two training sets for fine-tuning NER model (see *protagonist_tagger/data/ner_training_sets/*)
	+ fine-tuned NER model to be reused or fine-tuned further (see *protagonist_tagger/data/results/ner_model/fine_tuned/fine_tuned_ner_model/*)
3. **ProtagonistTagger** tool itself with several scripts that make it extremely easy to reuse it.

## How to use the protagonistTagger
In order to make the tool easy to use, there are several scripts offering most important functionalities. The scripts are located in *protagonist_tagger/tool/scripts* and they can be simply lauched from terminal with a set of necessary arguments. The following scripts are available:
+ *compute_metrics.py*  - given testing set annotated only by fine-tuned NER model or by protagonistTagger and a gold standard, it computes metrics such as precision, recall, F-measure
+ *fine_tune_ner_model.py* - given training set(s), it fine-tunes a predefined spacy NER model and saves it to a given directory to be reused
+ *generate_test_data.py* - given full plain texts of novels, it extracts sentences contiaing named entites of category *person*  in order to create testing sets
+ *prepare_training_set_with_common_names_for_ner_fine_tuning.py* - given a set of common English names, lists of literary characters and full plain texts of novels, it creates training set for fine-tuning NER model by injecting to sentences extracted from novels (and containing at least one named entity of category *person*) common English names
+ *prepare_training_set_with_not_recognized_named_entities_for_ner_fine_tuning.py* - given sets of named entities of category *person* not recognized by standard NER model and full plain tests of novels, it creates training set for NER model fine-tuning by extracting from novels and semi-automatically annotating with general tag PERSON sentences with not recognized named entites
+ *test_matcher_algorithm.py* - given list of literary characters, novels texts (either full or only some extracted sentences), fine-tuned NER model and precision (for approximate string matching), it annotates the given text with names of literary characters from the list
+ *test_ner_model.py* - given NER model, testing sets and gold standards annotated with full names of literary characters, it generalizes the gold standard (by replacing character-specific annotation with general tag PERSON) and annotates the testing set with general tag PERSON using given NER model; created annotated sets are prepared to be compared and analysed by computing metrics


Detailed information about input arguments and functionalities implemented in each script can be found in comments in corresponding python files. Each script can be run from terminal (from \protagonist_tagger directory) according to the same schema:

`$ python -m tool.scripts.<script_name> arg1 arg2 arg3 arg4`

Arguments should be separated with single space and given without any quotation marks. For example:

`$ python -m tool.scripts.generate_test_data C:\users\john_smith\desktop\titles.txt C:\users\john_smith\desktop\novels_texts\ 20 C:\users\john_smith\desktop\my_results\`

## Prerequisites
In order to successfully use the provided scripts the following requirements need to be fullfilled:
- Python 3.6
- PyYAML 5.3
- gensim 3.8
- numpy 1.18.2
- pytorch.transformers 1.2
- scikit-learn 0.22
- scipy 1.4.1
- spacy 2.2.4

Additionally the following packages are used:
- fuzzywuzzy 0.18 - https://pypi.org/project/fuzzywuzzy/
- gender-guesser 0.4 - https://pypi.org/project/gender-guesser/
- nickname-and-diminutive-names-lookup - https://github.com/carltonnorthern/nickname-and-diminutive-names-lookup

## More information
More information about the prepared corpus, data sets and the tool itself can be found in the presentation and the paper located in this repository.
