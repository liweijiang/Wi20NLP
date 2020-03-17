# Filename: different_queries.py
# Author: Liwei Jiang
# Description: Code for understanding how the performance of a pretrained language 
# model varies with different ways of querying for a particular fact
# Date: 02/25/2020

import sys
sys.path.append('')
sys.path.append('../')

import utils
import random
import jsonlines
import os.path
from os import path
from lama.modules import build_model_by_name
from lama.utils import print_sentence_predictions, load_vocab
import lama.options as options
import lama.evaluation_metrics as evaluation_metrics

NUM_EVIDENCE = 10

def load_relations():
	"""
	Load the relations 
	"""
	relations = {}
	with jsonlines.open('data/relations.jsonl') as reader:
		for obj in reader:
			relations[obj['relation']] = obj['label']
	return relations


def load_data_with_relation(relation_id):
    """
    Load the data with the given relation id
    """
    data = []
    data_path = 'data/TREx/' + relation_id + '.jsonl'
    if path.exists(data_path):
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                random.shuffle(obj['evidences'])
                obj['evidences'] = obj['evidences'][:NUM_EVIDENCE]
                data.append(obj)
        # Filter out the facts with less than NUM_EVIDENCE evidence sentences
        data_filtered = [d for d in data if len(d['evidences']) >= NUM_EVIDENCE]

        # Randomly keep at most 100 facts
        # random.shuffle(data_filtered)
        return data_filtered[:100]
    else:
        return None



def format_fact(fact):
    formatted_fact = []
    for i in range(len(fact["evidences"])):
        new_fact = {
        "obj_label": fact["obj_label"],
        "sub_label": fact["sub_label"],
        }
        new_fact["masked_sentences"] = [fact["evidences"][i]["masked_sentence"]]
        formatted_fact.append(new_fact)
    return formatted_fact


def reformat_data(data):
    """
    Reformat the data
    """
    all_data = []
    all_fact_pair = []
    for fact in data:
        formatted_fact = format_fact(fact)
        all_data.extend(formatted_fact)
        all_fact_pair.append({"object": fact["obj_label"], "subject": fact["sub_label"]})
    return all_data, all_fact_pair


def parse_facts():
    """
    Parse the fact data
    """
    sampled_facts = []
    all_fact_pair = []
    relations = load_relations()

    for (id, label) in relations.items():
        data = load_data_with_relation(id)
        if data is not None:
            data, fact_pair = reformat_data(data)
            sampled_facts.extend(data)
            all_fact_pair.extend(fact_pair)
            # print(data)

    print(len(sampled_facts))
    print(len(all_fact_pair))

    utils.save_data_to_jsonl("reproduction/data/TREx_filter/different_queries.jsonl", sampled_facts) # 31220
    utils.save_data_to_jsonl("reproduction/data/TREx_filter/different_queries_facts.jsonl", all_fact_pair) # 3122
    return sampled_facts


if __name__ == "__main__":
	facts = parse_facts()
