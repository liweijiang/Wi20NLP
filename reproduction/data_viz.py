# Filename: data_viz.py
# Author: Liwei Jiang
# Description: Visualize the data results
# Date: 03/10/2020


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt 
from statistics import mean 

transformerxl_rank = "transformerxl_rank_list.jsonl"
elmo_rank = "elmo_rank_list.jsonl"
elmo5B_rank = "elmo5B_rank_list.jsonl"
bert_base = "bert_base_rank_list.jsonl"
bert_large = "bert_large_rank_list.jsonl"


def bar_plot(df, filename, x_label="", y_label="", ylim=1750):
	plt.figure()
	fig = df.plot.box(grid='True', showfliers=False)
	fig = fig.get_figure()
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.ylim((0, ylim))
	fig.savefig(filename)


def scatter_plot(df, filename, x_label="", y_label="", logx=True):
	plt.figure()
	fig = df.plot.line(grid='True', logx=logx)
	fig = fig.get_figure()
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.xlim((1, 100))
	fig.savefig(filename)


def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def load_ranks(filename, rank_threshold=1000):
	# data = load_file(filename)[0]
	# data = [int(mean(d)) for d in data if len(d) > 0]
	# return np.asarray(data)
	data = load_file(filename)[0]
	new_data = []
	zero_data_index = []
	for i in range(len(data)):
		new_d = [x for x in data[i] if (x <= rank_threshold and x > 0)]
		if len(new_d) <= 0:
			zero_data_index.append(i)
			new_d = [0]
		new_data.append(int(mean(new_d)))
	return np.asarray(new_data), zero_data_index


def get_p_at_k(data, k=100):
	rank_count_dic = {i: 0 for i in range(1, k + 1)}
	for d in data:
		for i in range(1, k + 1):
			if d <= i and d > 0:
				rank_count_dic[i] += 1
	return [0] + [i/len(data) * 100 for i in rank_count_dic.values()]


def plot_different_queries():
	data_path = "TREx_filter_all/"
	Txl_data, Txl_zeros = load_ranks_2(data_path + transformerxl_rank)
	Eb_data, Eb_zeros = load_ranks_2(data_path + elmo_rank)
	E5B_data, E5B_zeros = load_ranks_2(data_path + elmo5B_rank)
	Bb_data, Bb_zeros = load_ranks_2(data_path + bert_base)
	Bl_data, Bl_zeros = load_ranks_2(data_path + bert_large)

	data = np.asarray([Txl_data, Eb_data, E5B_data, Bb_data, Bl_data])
	zeros = list(set(Txl_zeros + Eb_zeros + E5B_zeros + Bb_zeros + Bl_zeros))
	date_clean = np.delete(data, zeros, axis=1)
	df = pd.DataFrame(date_clean.T, columns=['Txl', 'Eb', 'E5B', 'Bb', 'Bl'])	
	bar_plot(df, "plot/TREx_filter_all.pdf", y_label="rank") 


def plot_p_at_k():
	data_path = "P_AT_K/"
	transformerxl_data = load_file(data_path + transformerxl_rank)
	elmo_data = load_file(data_path + elmo_rank)
	elmo5B_data = load_file(data_path + elmo5B_rank)
	bert_base_data = load_file(data_path + bert_base)
	bert_large_data = load_file(data_path + bert_large)

	df = pd.DataFrame(np.zeros((101, 5)), columns=['Txl', 'Eb', 'E5B', 'Bb', 'Bl'])
	df['Txl'] = get_p_at_k(transformerxl_data)
	df['Eb'] = get_p_at_k(elmo_data)
	df['E5B'] = get_p_at_k(elmo5B_data)
	df['Bb'] = get_p_at_k(bert_base_data)
	df['Bl'] = get_p_at_k(bert_large_data)

	scatter_plot(df, "plot/P_AT_K.pdf", y_label="mean P@k", x_label="k") 

if __name__ == "__main__":
	# plot_different_queries()
	plot_p_at_k()
