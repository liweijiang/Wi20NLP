# Filename: different_queries_experiment.py
# Author: Liwei Jiang
# Description: Code for understanding how the performance of a pretrained language 
# model varies with different ways of querying for a particular fact
# Date: 02/25/2020

import sys
sys.path.append('')
import argparse
from batch_eval_KB_completion import main as run_evaluation
from batch_eval_KB_completion import load_file
from lama.modules import build_model_by_name
import pprint
import statistics
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict

LMs = [
    {
        "lm": "transformerxl",
        "label": "transformerxl",
        "models_names": ["transformerxl"],
        "transformerxl_model_name": "transfo-xl-wt103",
        "transformerxl_model_dir": "pre-trained_language_models/transformerxl/transfo-xl-wt103/",
    },
    {
        "lm": "elmo",
        "label": "elmo",
        "models_names": ["elmo"],
        "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway",
        "elmo_vocab_name": "vocab-2016-09-10.txt",
        "elmo_model_dir": "pre-trained_language_models/elmo/original",
        "elmo_warm_up_cycles": 10,
    },
    {
        "lm": "elmo",
        "label": "elmo5B",
        "models_names": ["elmo"],
        "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway_5.5B",
        "elmo_vocab_name": "vocab-enwiki-news-500000.txt",
        "elmo_model_dir": "pre-trained_language_models/elmo/original5.5B/",
        "elmo_warm_up_cycles": 10,
    },
    {
        "lm": "bert",
        "label": "bert_base",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-12_H-768_A-12",
    },
    {
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
]



def run_experiments(
    data_path_pre,
    data_path_post,
    input_param={
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
    use_negated_probes=False,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)


    for i in range(1):
        PARAMETERS = {
            "dataset_filename": "reproduction/data/TREx_filter/different_queries.jsonl",
            "fact_pair_filename": "reproduction/data/TREx_filter/different_queries_facts.jsonl",
            "common_vocab_filename": "pre-trained_language_models/common_vocab_cased.txt",
            "template": "",
            "bert_vocab_name": "vocab.txt",
            "batch_size": 10,
            "logdir": "output",
            "full_logdir": "output/results/{}/{}".format(
                input_param["label"], "different_queries"
            ),
            "lowercase": False,
            "max_sentence_length": 100,
            "threads": -1,
            "interactive": False,
            "use_negated_probes": use_negated_probes,
        }

        PARAMETERS.update(input_param)
        args = argparse.Namespace(**PARAMETERS)

        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name(model_type_name, args)

        Precision1 = run_evaluation(args, shuffle_data=False, model=model)
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)


    mean_p1 = statistics.mean(all_Precision1)
    print("@@@ {} - mean P@1: {}".format(input_param["label"], mean_p1))

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            input_param["label"],
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    return mean_p1, all_Precision1


def get_TREx_parameters(data_path_pre="data/"):
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return data_path_pre, data_path_post


def run_all_LMs(parameters):
    for ip in LMs:
        print(ip["label"])
        run_experiments(*parameters, input_param=ip, use_negated_probes=False)


if __name__ == "__main__":
    print("T-REx")
    parameters = get_TREx_parameters()
    run_all_LMs(parameters)

