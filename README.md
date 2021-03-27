# Protagonists' Catcher in Novels -- A Dataset and A Method
Semantic annotation of long texts, such as novels, remains an open challenge in the field of Semantic Web (SW) and Natural Language Processing (NLP). Recognizing and identifying literary characters in full-text novels is the first step for a more detailed literary text analysis. This research investigates the problem of ontology population, i.e. recognizing people (especially main characters) in novels and annotating them. We prepared a tool -- **protagonistTagger** -- for this annotation and a dataset to test it. 

Our process of identifying literary characters in a text, implemented in **protagonistTagger**, comprises two stages: (1) named entity recognition (NER) of persons, (2) matching method of each recognized person with the literary character's full name associated with it, based on **approximate text matching**. 

The performance of **protagonistTagger** in thirteen full-text novels shows that the tool achieved both precision and recall above 80\%. The test datasets comprise 1300 sentences from classic novels of different genres that a novel reader had annotated. 

Exemplary annotations performed by **protagonistTagger**:
>"Her disappointment in **Charlotte --Charlotte Lucas--** made her turn with fonder regard to her sister, of whose rectitude and delicacy she was sure her opinion could never be shaken, and for whose happiness she grew daily more anxious, as **Bingley --Charles Bingley--** had now been gone a week and nothing more was heard of his return. **Jane --Jane Bennet--** had sent **Caroline --Caroline Bingley--** an early answer to her letter and was counting the days till she might reasonably hope to hear again. The promised letter of thanks from **Mr. Collins --Mr William Collins--** arrived on Tuesday, addressed to their father, and written with all the solemnity of gratitude which a twelvemonth’s abode in the family might have prompted."  

#General Project Workflow
The process of creating the corpus of annotated novels and the **protagonistTagger** tool comprises several stages:
- Gathering an initial corpus with plain novels' texts without annotations. 
- Creating a list of full names of all protagonists for each novel in the initial corpus. These names are the predefined tags that will be used in further steps for annotations.
- Recognizing named entity of category **person** in the texts of the novels in the initial corpus. Training NER model from scratch for this specific problem is not reasonable due to the amount of time and computing power it requires. It is possible to use some pre-trained NER model and fine-tune it using a sample of manually annotated data. The evaluation of the NER mechanism is done on a testing set extracted from the full texts of novels. The task is quite complex and may include several iterations.
- Each named entity of category **person**  recognized by NER in the previous step is a potential candidate to be annotated with one of our tags predefined in step 3. At this point, an algorithm (let us call it **matching algorithm** for reference), based on approximate string matching, is used to choose from the list of predefined tags the one that matches most accurately the recognized named entity. 
- The annotations did by the **matching algorithm** are accessed according to their accuracy and correctness.
The **protagonistTagger** (fine-tuned NER + matching algorithm) is used to annotate more novels in order to create the corpus of annotated novels. 
