import pickle
import os


result_file = "output/results/bert_large/test/result.pkl"

with open(result_file, "rb") as fin:
    results = pickle.load(fin)["list_of_results"]

csv_file = os.path.join(os.path.dirname(result_file), "preds.tsv")
with open(csv_file, "w") as fout:
    for result in results:
        masked_sentence = result["sample"]["masked_sentences"][0]
        obj_label = result["sample"]["obj_label"]
        precision_at_1 = result["sample_Precision1"]
        top_k_preds = [it["token_word_form"] for it in result["masked_topk"]]
        fout.write(
            "{}\t{}\t{}\t{}\n".format(
                masked_sentence,
                obj_label,
                precision_at_1,
                "\t".join(top_k_preds)
            )
        )

