This corpus release contains 4,993 documents annotated with (P)articipants, (I)nterventions, and (O)utcomes.
The files included in this release are as follows

documents/
  Documents are labeled by their PubMed identification number (PMID).
  Each document has two files:
    1) PMID.text - the raw text of the abstract
    2) PMID.tokens - the space-separated tokens (from nltk's punkt tokenizer) that labels are assigned to

annotations/
  All annotation files are presented as a space-separated list of labels for the corresponding document tokens.
  The first division of annotations is in to either:
  1) individual/
      This folder contains each individual annotation provided by each worker, with multiple annotations per document
      Annotators are assigned unique worker id (WID) numbers. The annotation files are labeled as:
        PMID_WID.ann
  2) aggregated/
      This folder contains the aggregated (cleaner, less noisy) annotations with only one file per document
  Within these folders, the annotations are separated by the two annotation phases:
  1) starting_spans/
      These are the first-phase annotations where workers highlighted spans containing target information.
  2) hierarchical_labels/
      These are the second-phase annotations where workers received the previous PMID_AGGREGATED.ann annotations
      for each document and assigned more specific labels to whichever already labeled tokens deemed relevant.
  After this, the files are separated by PICO element and then the train/test partitions. The test folder contains two versions:
  1) test/gold/
      These annotations were collected from medical professionals and are the true target testing set
  2) test/crowd/
      This annotations were collected on AMT and used to validate the quality of the crowd-sourced labels

      The label mappings for each PIO element are:

      participants/
        0: No label
        1: Age
        2: Sex
        3: Sample size
        4: Condition

      interventions/
        0: No label
        1: Surgical
        2: Physical
        3: Pharmacological
        4: Educational
        5: Psychological
        6: Other
        7: Control

      outcomes/
        0: No label
        1: Physical
        2: Pain
        3: Mortality
        4: Adverse effects
        5: Mental
        6: Other

        Here, two sections of the hierarchy have been collapsed due to frequent cross-labeling:
        
        Mental =
          Mental health
          Mental behavioral impact
          Participant behavior

        Other =
          Satisfaction with care
          Non-health outcomes
          Quality of intervention
          Resource use
          Withdrawl from study

        The full labels are available upon request (nye.b@husky.neu.edu)
