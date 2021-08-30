from enum import Enum


class Data(Enum):
    
    train = {"entity": "../data/chemprot_training/chemprot_training_entities.tsv",
            "abstract": "../data/chemprot_training/chemprot_training_abstracts.tsv",
             "relations": "../data/chemprot_training/chemprot_training_relations.tsv",
             "gold_standard": "../data/chemprot_training/chemprot_training_gold_standard.tsv"
            }
    
    test = {"entity": "../data/chemprot_test_gs/chemprot_test_entities_gs.tsv",
            "abstract": "../data/chemprot_test_gs/chemprot_test_abstracts_gs.tsv",
             "relations": "../data/chemprot_test_gs/chemprot_test_relations_gs.tsv",
             "gold_standard": "../data/chemprot_test_gs/chemprot_test_gold_standard.tsv"
            }
    
    dev = {"entity": "../data/chemprot_development/chemprot_development_entities.tsv",
            "abstract": "../data/chemprot_development/chemprot_development_abstracts.tsv",
             "relations": "../data/chemprot_development/chemprot_development_relations.tsv",
             "gold_standard": "../data/chemprot_development/chemprot_development_gold_standard.tsv"
            }
            

class Pickle_Path(Enum):
    
    train_sentence_split_path = "../pickles/geniass_sentence_split_frame.pickle"
    train_multilabel_path = "../pickles/sentence_label_only_include_gen_chem_multilabel_not_duplicated.pickle"
    train_tagged_sentence_path = "../pickles/inner_sentence_only_include_gen_chem_bert_not_duplicated.pickle"
    
    dev_sentence_split_path = "../pickles/geniass_sentence_split_development.pickle"
    dev_multilabel_path = "../pickles/sentence_label_only_include_gen_chem_multilabel_not_duplicated_dev.pickle"
    dev_tagged_sentence_path = "../pickles/inner_sentence_only_include_gen_chem_bert_not_duplicated_dev.pickle"
    
    test_sentence_split_path = "../pickles/geniass_sentence_split_frame_test_gs.pickle"
    test_multilabel_path = "../pickles/sentence_label_only_include_gen_chem_multilabel_test_gs.pickle"
    test_tagged_sentence_path = "../pickles/inner_sentence_only_include_gen_chem_bert_test_gs.pickle"
    
    