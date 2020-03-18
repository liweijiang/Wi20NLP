# Filename: batch_eval_KB_completion.py
# Author: Liwei Jiang
# Description: The data evaluation code for different queries experiment.
# Date: 03/10/2020

import sys
sys.path.append('')
sys.path.append('../')
from lama.modules import build_model_by_name
import lama.utils as utils
from lama.utils import print_sentence_predictions, load_vocab
import lama.options as options
from tqdm import tqdm
from random import shuffle
import os
import json
import spacy
import lama.modules.base_connector as base
from pprint import pprint
import logging.config
import logging
import pickle
from multiprocessing.pool import ThreadPool
import multiprocessing
import reproduction.evaluation_metrics as metrics
import time, sys
from utils import *

P_AT_K = 10

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def create_logdir_with_timestamp(base_logdir, modelname):
    timestr = time.strftime("%Y%m%d_%H%M%S")

    # create new directory
    log_directory = "{}/{}_{}/".format(base_logdir, modelname, timestr)
    os.makedirs(log_directory)

    path = "{}/last".format(base_logdir)
    try:
        os.unlink(path)
    except Exception:
        pass
    os.symlink(log_directory, path)
    return log_directory


# def parse_template(template, subject_label, object_label):
#     SUBJ_SYMBOL = "[X]"
#     OBJ_SYMBOL = "[Y]"
#     template = template.replace(SUBJ_SYMBOL, subject_label)
#     template = template.replace(OBJ_SYMBOL, object_label)
#     return [template]


def init_logging(log_directory):
    logger = logging.getLogger("LAMA")
    logger.setLevel(logging.DEBUG)

    os.makedirs(log_directory, exist_ok=True)

    # logging format
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # file handler
    fh = logging.FileHandler(str(log_directory) + "/info.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.propagate = False

    return logger


def batchify(data, batch_size):
    msg = ""
    list_samples_batches = []
    list_sentences_batches = []
    current_samples_batch = []
    current_sentences_batches = []
    c = 0

    # sort to group togheter sentences with similar length
    for sample in sorted(
        data, key=lambda k: len(" ".join(k["masked_sentences"]).split())
    ):
        masked_sentences = sample["masked_sentences"]
        current_samples_batch.append(sample)
        current_sentences_batches.append(masked_sentences)
        c += 1
        if c >= batch_size:
            list_samples_batches.append(current_samples_batch)
            list_sentences_batches.append(current_sentences_batches)
            current_samples_batch = []
            current_sentences_batches = []
            c = 0

    # last batch
    if current_samples_batch and len(current_samples_batch) > 0:
        list_samples_batches.append(current_samples_batch)
        list_sentences_batches.append(current_sentences_batches)

    return list_samples_batches, list_sentences_batches, msg


def run_thread(arguments):

    msg = ""

    # 1. compute the ranking metrics on the filtered log_probs tensor
    sample_MRR, sample_P, experiment_result, return_msg = metrics.get_ranking(
        arguments["filtered_log_probs"],
        arguments["masked_indices"],
        arguments["vocab"],
        label_index=arguments["label_index"],
        index_list=arguments["index_list"],
        print_generation=arguments["interactive"],
        topk=10000,
        P_AT=P_AT_K
    )
    msg += "\n" + return_msg

    sample_perplexity = 0.0

    return experiment_result, sample_MRR, sample_P, sample_perplexity, msg


# def lowercase_samples(samples, use_negated_probes=False):
    # new_samples = []
    # for sample in samples:
    #     sample["obj_label"] = sample["obj_label"].lower()
    #     sample["sub_label"] = sample["sub_label"].lower()
    #     lower_masked_sentences = []
    #     for sentence in sample["masked_sentences"]:
    #         sentence = sentence.lower()
    #         sentence = sentence.replace(base.MASK.lower(), base.MASK)
    #         lower_masked_sentences.append(sentence)
    #     sample["masked_sentences"] = lower_masked_sentences

    #     if "negated" in sample and use_negated_probes:
    #         for sentence in sample["negated"]:
    #             sentence = sentence.lower()
    #             sentence = sentence.replace(base.MASK.lower(), base.MASK)
    #             lower_masked_sentences.append(sentence)
    #         sample["negated"] = lower_masked_sentences

    #     new_samples.append(sample)
    # return new_samples


def filter_samples(model, samples, vocab_subset, max_sentence_length, template):
    msg = ""
    new_samples = []
    samples_exluded = 0
    for sample in samples:
        excluded = False
        if "obj_label" in sample and "sub_label" in sample:

            obj_label_ids = model.get_id(sample["obj_label"])

            if obj_label_ids:
                recostructed_word = " ".join(
                    [model.vocab[x] for x in obj_label_ids]
                ).strip()
            else:
                recostructed_word = None

            # print(sample)

            excluded = False
            if not template or len(template) == 0:
                masked_sentences = sample["masked_sentences"]
                text = " ".join(masked_sentences)
                if len(text.split()) > max_sentence_length:
                    msg += "\tEXCLUDED for exeeding max sentence length: {}\n".format(
                        masked_sentences
                    )
                    samples_exluded += 1
                    excluded = True

            # MAKE SURE THAT obj_label IS IN VOCABULARIES
            if vocab_subset:
                for x in sample["obj_label"].split(" "):
                    if x not in vocab_subset:
                        excluded = True
                        msg += "\tEXCLUDED object label {} not in vocab subset\n".format(
                            sample["obj_label"]
                        )
                        samples_exluded += 1
                        break

            if excluded:
                pass
            elif obj_label_ids is None:
                msg += "\tEXCLUDED object label {} not in model vocabulary\n".format(
                    sample["obj_label"]
                )
                samples_exluded += 1
            elif not recostructed_word or recostructed_word != sample["obj_label"]:
                msg += "\tEXCLUDED object label {} not in model vocabulary\n".format(
                    sample["obj_label"]
                )
                samples_exluded += 1
            else:
                new_samples.append(sample)
        else:
            msg += "\tEXCLUDED since 'obj_label' not sample or 'sub_label' not in sample: {}\n".format(
                sample
            )
            samples_exluded += 1
    msg += "samples exluded  : {}\n".format(samples_exluded)
    return new_samples, msg


def main(args, shuffle_data=True, model=None):

    if len(args.models_names) > 1:
        raise ValueError('Please specify a single language model (e.g., --lm "bert").')

    msg = ""
    [model_type_name] = args.models_names

    # print("------- Model: {}".format(model))
    # print("------- Args: {}".format(args))
    if model is None:
        model = build_model_by_name(model_type_name, args)

    if model_type_name == "fairseq":
        model_name = "fairseq_{}".format(args.fairseq_model_name)
    elif model_type_name == "bert":
        model_name = "BERT_{}".format(args.bert_model_name)
    elif model_type_name == "elmo":
        model_name = "ELMo_{}".format(args.elmo_model_name)
    else:
        model_name = model_type_name.title()

    # initialize logging
    if args.full_logdir:
        log_directory = args.full_logdir
    else:
        log_directory = create_logdir_with_timestamp(args.logdir, model_name)
    logger = init_logging(log_directory)
    msg += "model name: {}\n".format(model_name)

    # deal with vocab subset
    vocab_subset = None
    index_list = None
    msg += "args: {}\n".format(args)
    if args.common_vocab_filename is not None:
        vocab_subset = load_vocab(args.common_vocab_filename)
        msg += "common vocabulary size: {}\n".format(len(vocab_subset))

        # optimization for some LM (such as ELMo)
        model.optimize_top_layer(vocab_subset)

        filter_logprob_indices, index_list = model.init_indices_for_filter_logprobs(
            vocab_subset, logger
        )

    logger.info("\n" + msg + "\n")

    # dump arguments on file for log
    with open("{}/args.json".format(log_directory), "w") as outfile:
        json.dump(vars(args), outfile)

    # Mean reciprocal rank
    MRR = 0.0

    # Precision at (default 10)
    Precision = 0.0
    Precision1 = 0.0
    Precision_negative = 0.0
    Precision_positivie = 0.0

    data = load_file(args.dataset_filename)
    # data = data[:2000]
    fact_pair = load_file(args.fact_pair_filename)

    # print("@@@@@@@@@@@@@@")
    # print(fact_pair)
    # print(len(fact_pair))
    # print("$$$$$$$$$$$$$$")

    all_samples, ret_msg = filter_samples(
        model, data, vocab_subset, args.max_sentence_length, args.template
    )

    # print("!!!!!!!!!!!!!")
    # print(len(all_samples)) # 30847

    logger.info("\n" + ret_msg + "\n")

    # for sample in all_samples:
    #     sample["masked_sentences"] = [sample['evidences'][0]['masked_sentence']]

    # create uuid if not present
    i = 0
    for sample in all_samples:
        if "uuid" not in sample:
            sample["uuid"] = i
        i += 1

    # shuffle data
    if shuffle_data:
        shuffle(all_samples)

    samples_batches, sentences_batches, ret_msg = batchify(all_samples, args.batch_size)
    logger.info("\n" + ret_msg + "\n")

    # ThreadPool
    num_threads = args.threads
    if num_threads <= 0:
        # use all available threads
        num_threads = multiprocessing.cpu_count()
    pool = ThreadPool(num_threads)
    
    list_of_results = {d['subject'] + "_" + d['object']: [] for d in fact_pair}
    list_of_ranks = {d['subject'] + "_" + d['object']: [] for d in fact_pair}
    for i in tqdm(range(len(samples_batches))):

        samples_b = samples_batches[i]
        sentences_b = sentences_batches[i]

        (
            original_log_probs_list,
            token_ids_list,
            masked_indices_list,
        ) = model.get_batch_generation(sentences_b, logger=logger)

        if vocab_subset is not None:
            # filter log_probs
            filtered_log_probs_list = model.filter_logprobs(
                original_log_probs_list, filter_logprob_indices
            )
        else:
            filtered_log_probs_list = original_log_probs_list

        label_index_list = []
        for sample in samples_b:
            obj_label_id = model.get_id(sample["obj_label"])

            # MAKE SURE THAT obj_label IS IN VOCABULARIES
            if obj_label_id is None:
                raise ValueError(
                    "object label {} not in model vocabulary".format(
                        sample["obj_label"]
                    )
                )
            elif model.vocab[obj_label_id[0]] != sample["obj_label"]:
                raise ValueError(
                    "object label {} not in model vocabulary".format(
                        sample["obj_label"]
                    )
                )
            elif vocab_subset is not None and sample["obj_label"] not in vocab_subset:
                raise ValueError(
                    "object label {} not in vocab subset".format(sample["obj_label"])
                )

            label_index_list.append(obj_label_id)

        arguments = [
            {
                "original_log_probs": original_log_probs,
                "filtered_log_probs": filtered_log_probs,
                "token_ids": token_ids,
                "vocab": model.vocab,
                "label_index": label_index[0],
                "masked_indices": masked_indices,
                "interactive": args.interactive,
                "index_list": index_list,
                "sample": sample,
            }
            for original_log_probs, filtered_log_probs, token_ids, masked_indices, label_index, sample in zip(
                original_log_probs_list,
                filtered_log_probs_list,
                token_ids_list,
                masked_indices_list,
                label_index_list,
                samples_b,
            )
        ]
        # single thread for debug
        # for isx,a in enumerate(arguments):
        #     print(samples_b[isx])
        #     run_thread(a)

        # multithread
        res = pool.map(run_thread, arguments)
 
        for idx, result in enumerate(res):
            result_masked_topk, sample_MRR, sample_P, sample_perplexity, msg = result
            logger.info("\n" + msg + "\n")

            sample = samples_b[idx]

            element = {}
            obj = sample['obj_label']
            sub = sample['sub_label']
            element["masked_sentences"] = sample["masked_sentences"][0]
            element["uuid"] = sample["uuid"]
            element["subject"] = sub
            element["object"] = obj
            element["rank"] = int(result_masked_topk['rank'])
            element["sample_Precision1"] = result_masked_topk["P_AT_1"]
            # element["sample"] = sample
            # element["token_ids"] = token_ids_list[idx]
            # element["masked_indices"] = masked_indices_list[idx]
            # element["label_index"] = label_index_list[idx]
            # element["masked_topk"] = result_masked_topk
            # element["sample_MRR"] = sample_MRR
            # element["sample_Precision"] = sample_P
            # element["sample_perplexity"] = sample_perplexity
            
            list_of_results[sub + "_" + obj].append(element)
            list_of_ranks[sub + "_" + obj].append(element["rank"])

            # print("~~~~~~ rank: {}".format(result_masked_topk['rank']))
            MRR += sample_MRR
            Precision += sample_P
            Precision1 += element["sample_Precision1"]

            append_data_line_to_jsonl("reproduction/data/TREx_filter/{}_rank_results.jsonl".format(args.label), element) # 3122

            # list_of_results.append(element)

    pool.close()
    pool.join()

    # stats
    # Mean reciprocal rank
    # MRR /= len(list_of_results)

    # # Precision
    # Precision /= len(list_of_results)
    # Precision1 /= len(list_of_results)

    # msg = "all_samples: {}\n".format(len(all_samples))
    # # msg += "list_of_results: {}\n".format(len(list_of_results))
    # msg += "global MRR: {}\n".format(MRR)
    # msg += "global Precision at 10: {}\n".format(Precision)
    # msg += "global Precision at 1: {}\n".format(Precision1)

    # logger.info("\n" + msg + "\n")
    # print("\n" + msg + "\n")

    # dump pickle with the result of the experiment
    # all_results = dict(
    #     list_of_results=list_of_results, global_MRR=MRR, global_P_at_10=Precision
    # )
    # with open("{}/result.pkl".format(log_directory), "wb") as f:
    #     pickle.dump(all_results, f)

    # print()
    # model_name = args.models_names[0]
    # if args.models_names[0] == "bert":
    #     model_name = args.bert_model_name
    # elif args.models_names[0] == "elmo":
    #     if args.bert_model_name == args.bert_model_name
    # else:

    # save_data_line_to_jsonl("reproduction/data/TREx_filter/{}_rank_results.jsonl".format(args.label), list_of_results) # 3122
    # save_data_line_to_jsonl("reproduction/data/TREx_filter/{}_rank_dic.jsonl".format(args.label), list_of_ranks) # 3122
    save_data_line_to_jsonl("reproduction/data/TREx_filter/{}_rank_list.jsonl".format(args.label), list(list_of_ranks.values())) # 3122
 

    return Precision1


if __name__ == "__main__":
    parser = options.get_eval_KB_completion_parser()
    args = options.parse_args(parser)
    main(args)
