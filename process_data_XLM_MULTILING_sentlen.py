import csv
import json
from collections import defaultdict
import numpy as np
import torch
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-ln', type=str, default='en')
    parser.add_argument('-pm', type=str)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-bs', type=int, default=4)
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-seed', type=int, default=42)

    args = parser.parse_args()

    for single_lang in ['en', 'de', 'nl', 'zh', 'ru', 'hi']:
        #single_lang = args.ln
        print(single_lang)
        #print('stack adapters xlm-roberta')

        parameter = args.pm
        #print(parameter)

        learning_rate = args.lr
        batch_size = args.bs
        epochs = args.epoch

        seed = args.seed
        torch.manual_seed(seed)

        #print(f'LR {learning_rate}, BS {batch_size}, EP {epochs}, SEED {seed}')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        datapaths = {'train': 'data/training_data2022/training_data/train.csv'}
                     #'val': 'data/training_data2022/training_data/dev.csv'}

        header = ''

        original_data = dict()

        for split in datapaths:

            print(split)
            path = datapaths[split]

            dataset_param = defaultdict(dict)

            with open(path, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=",")

                for idx, line in enumerate(csvreader):

                    if idx == 0:
                        print(idx, line)  # header
                        header = line

                    else:
                        #print(line)
                        if split in ['train', 'val']:
                            language, sentence_id, word_id, word, FFDAvg, FFDStd, TRTAvg, TRTStd = line
                            #print(word)

                            if language == single_lang:

                                if parameter == 'ffd_avg':
                                    param = float(FFDAvg)
                                elif parameter == 'ffd_std':
                                    param = float(FFDStd)
                                elif parameter == 'trt_avg':
                                    param = float(TRTAvg)
                                elif parameter == 'trt_std':
                                    param = float(TRTStd)

                                if sentence_id in dataset_param[language]:
                                    dataset_param[language][sentence_id]['text'].append(word)
                                    dataset_param[language][sentence_id]['labels'].append(param)

                                else:
                                    dataset_param[language][sentence_id] = {'text': [word], "labels": [param]}

            original_data[split] = dataset_param

        from transformers import XLMRobertaTokenizer

        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

        sos = tokenizer.convert_tokens_to_ids('<s>')
        eos = tokenizer.convert_tokens_to_ids('</s>')
        pad = tokenizer.convert_tokens_to_ids('<pad>')

        processed_data = dict()


        for split in original_data:

            dataset_param = original_data[split]  # TODO

            dataset_param_new = []

            for ds in [dataset_param]:

                for lang in ds:

                    sentence_count = 0
                    longest_count = 0

                    longest_real = 0
                    longest_sentid = 0

                    for sent_id in ds[lang]:

                        print(lang, sentence_count)
                        sentence_count += 1

                        tokenized = tokenizer(ds[lang][sent_id]['text'], add_special_tokens=False)

                        #tokenizer.batch_encode_plus()
                        #TODO check pad token id
                        # check G with dot
                        # XLM ids

                        input_ids = [sos]
                        attention_masks = [1]
                        new_labels = [-1]

                        orig_labels = ds[lang][sent_id]['labels']

                        for w_i, wps in enumerate(tokenized['input_ids']):

                            for wp in wps:

                                input_ids.append(wp)
                                attention_masks.append(1)
                                new_labels.append(orig_labels[w_i]) # TODO labels for subsequent labels use -100, check loss default

                        if len(input_ids) > 199:

                            longest_count += 1
                            print(len(input_ids))
                            # truncate to 199 because we will also add eos
                            input_ids = input_ids[:199]
                            attention_masks = attention_masks[:199]
                            new_labels = new_labels[:199]

                        input_ids.append(eos)
                        attention_masks.append(1)
                        new_labels.append(-1)

                        if len(input_ids) > longest_real:
                            longest_real = len(input_ids)
                            longest_sentid = sent_id

                    print('longest', sentence_count, longest_count)
                    print('longest real', longest_real)
                    print('longest sentid', longest_sentid)