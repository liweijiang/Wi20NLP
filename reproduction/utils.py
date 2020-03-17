# Filename: utils.py
# Author: Liwei Jiang
# Description: Helper functions
# Date: 02/25/2020

import jsonlines
import json

def save_data_to_jsonl(output_file, data):
	"""
	Save all data to a jsonl file
	"""
	with jsonlines.open(output_file, mode='w') as writer:
		writer.write_all(data)


def save_data_line_to_jsonl(output_file, data):
	"""
	Save the data line to a jsonl file
	"""
	with jsonlines.open(output_file, mode='w') as writer:
		writer.write(data)


def append_data_line_to_jsonl(output_file, data):
	"""
	Save the data line to a jsonl file
	"""
	with jsonlines.open(output_file, mode='a') as writer:
		writer.write(data)
