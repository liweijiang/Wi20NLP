##!/bin/bash
#
#python scripts_for_extra_exps/finetune_bert_for_kb.py \
#       --train_data_file data/Google_RE/place_of_birth_test_500.jsonl \
#       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
#       --num_train_examples 10 --output_dir lm_output/birth_place_10
#
#cp lm_output/birth_place_10/config.json lm_output/birth_place_10/bert_config.json
#
#python scripts_for_extra_exps/finetune_bert_for_kb.py \
#       --train_data_file data/Google_RE/place_of_birth_test_500.jsonl \
#       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
#       --num_train_examples 50 --output_dir lm_output/birth_place_50
#
#cp lm_output/birth_place_50/config.json lm_output/birth_place_50/bert_config.json
#
#python scripts_for_extra_exps/finetune_bert_for_kb.py \
#       --train_data_file data/Google_RE/place_of_birth_test_500.jsonl \
#       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
#       --num_train_examples 100 --output_dir lm_output/birth_place_100
#
#cp lm_output/birth_place_100/config.json lm_output/birth_place_100/bert_config.json
#
#python scripts_for_extra_exps/finetune_bert_for_kb.py \
#       --train_data_file data/Google_RE/place_of_birth_test_500.jsonl \
#       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
#       --num_train_examples 200 --output_dir lm_output/birth_place_200
#
#cp lm_output/birth_place_200/config.json lm_output/birth_place_200/bert_config.json
#
#python scripts_for_extra_exps/finetune_bert_for_kb.py \
#       --train_data_file data/Google_RE/place_of_birth_test_500.jsonl \
#       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
#       --num_train_examples 300 --output_dir lm_output/birth_place_300
#
#cp lm_output/birth_place_300/config.json lm_output/birth_place_300/bert_config.json
#
#python scripts_for_extra_exps/finetune_bert_for_kb.py \
#       --train_data_file data/Google_RE/place_of_birth_test_500.jsonl \
#       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
#       --num_train_examples 400 --output_dir lm_output/birth_place_400
#
#cp lm_output/birth_place_400/config.json lm_output/birth_place_400/bert_config.json
#
#python scripts_for_extra_exps/finetune_bert_for_kb.py \
#       --train_data_file data/Google_RE/place_of_birth_test_500.jsonl \
#       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
#       --num_train_examples 500 --output_dir lm_output/birth_place_500
#
#cp lm_output/birth_place_500/config.json lm_output/birth_place_500/bert_config.json
#
#
#python scripts_for_extra_exps/finetune_bert_for_kb.py \
#       --train_data_file data/Google_RE/date_of_birth_test_500.jsonl \
#       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
#       --num_train_examples 10 --output_dir lm_output/birth_date_10
#
#cp lm_output/birth_date_10/config.json lm_output/birth_date_10/bert_config.json
#
#python scripts_for_extra_exps/finetune_bert_for_kb.py \
#       --train_data_file data/Google_RE/date_of_birth_test_500.jsonl \
#       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
#       --num_train_examples 50 --output_dir lm_output/birth_date_50
#
#cp lm_output/birth_date_50/config.json lm_output/birth_date_50/bert_config.json
#
#python scripts_for_extra_exps/finetune_bert_for_kb.py \
#       --train_data_file data/Google_RE/date_of_birth_test_500.jsonl \
#       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
#       --num_train_examples 100 --output_dir lm_output/birth_date_100
#
#cp lm_output/birth_date_100/config.json lm_output/birth_date_100/bert_config.json
#
#python scripts_for_extra_exps/finetune_bert_for_kb.py \
#       --train_data_file data/Google_RE/date_of_birth_test_500.jsonl \
#       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
#       --num_train_examples 200 --output_dir lm_output/birth_date_200
#
#cp lm_output/birth_date_200/config.json lm_output/birth_date_200/bert_config.json
#
#python scripts_for_extra_exps/finetune_bert_for_kb.py \
#       --train_data_file data/Google_RE/date_of_birth_test_500.jsonl \
#       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
#       --num_train_examples 300 --output_dir lm_output/birth_date_300
#
#cp lm_output/birth_date_300/config.json lm_output/birth_date_300/bert_config.json
#
#python scripts_for_extra_exps/finetune_bert_for_kb.py \
#       --train_data_file data/Google_RE/date_of_birth_test_500.jsonl \
#       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
#       --num_train_examples 400 --output_dir lm_output/birth_date_400
#
#cp lm_output/birth_date_400/config.json lm_output/birth_date_400/bert_config.json
#
#python scripts_for_extra_exps/finetune_bert_for_kb.py \
#       --train_data_file data/Google_RE/date_of_birth_test_500.jsonl \
#       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
#       --num_train_examples 500 --output_dir lm_output/birth_date_500
#
#cp lm_output/birth_date_500/config.json lm_output/birth_date_500/bert_config.json

python scripts_for_extra_exps/finetune_bert_for_kb.py \
       --train_data_file data/Google_RE/place_of_death_test_500.jsonl \
       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
       --num_train_examples 10 --output_dir lm_output/death_place_10

cp lm_output/death_place_10/config.json lm_output/death_place_10/bert_config.json

python scripts_for_extra_exps/finetune_bert_for_kb.py \
       --train_data_file data/Google_RE/place_of_death_test_500.jsonl \
       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
       --num_train_examples 50 --output_dir lm_output/death_place_50

cp lm_output/death_place_50/config.json lm_output/death_place_50/bert_config.json

python scripts_for_extra_exps/finetune_bert_for_kb.py \
       --train_data_file data/Google_RE/place_of_death_test_500.jsonl \
       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
       --num_train_examples 100 --output_dir lm_output/death_place_100

cp lm_output/death_place_100/config.json lm_output/death_place_100/bert_config.json

python scripts_for_extra_exps/finetune_bert_for_kb.py \
       --train_data_file data/Google_RE/place_of_death_test_500.jsonl \
       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
       --num_train_examples 200 --output_dir lm_output/death_place_200

cp lm_output/death_place_200/config.json lm_output/death_place_200/bert_config.json

python scripts_for_extra_exps/finetune_bert_for_kb.py \
       --train_data_file data/Google_RE/place_of_death_test_500.jsonl \
       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
       --num_train_examples 300 --output_dir lm_output/death_place_300

cp lm_output/death_place_300/config.json lm_output/death_place_300/bert_config.json

python scripts_for_extra_exps/finetune_bert_for_kb.py \
       --train_data_file data/Google_RE/place_of_death_test_500.jsonl \
       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
       --num_train_examples 400 --output_dir lm_output/death_place_400

cp lm_output/death_place_400/config.json lm_output/death_place_400/bert_config.json

python scripts_for_extra_exps/finetune_bert_for_kb.py \
       --train_data_file data/Google_RE/place_of_death_test_500.jsonl \
       --model_type bert --model_name_or_path bert-base-cased --logging_steps 1 --overwrite_output_dir \
       --num_train_examples 500 --output_dir lm_output/death_place_500

cp lm_output/death_place_500/config.json lm_output/death_place_500/bert_config.json