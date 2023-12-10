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

    single_lang = args.ln
    print(single_lang)
    print('from scratch adapters xlm-roberta')

    parameter = args.pm
    print(parameter)

    learning_rate = args.lr
    batch_size = args.bs
    epochs = args.epoch

    seed = args.seed
    torch.manual_seed(seed)

    print(f'LR {learning_rate}, BS {batch_size}, EP {epochs}, SEED {seed}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    datapaths = {'train': 'data/training_data2022/training_data/train.csv',
                 'val': 'data/training_data2022/training_data/dev.csv'}

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

                        # truncate to 199 because we will also add eos
                        input_ids = input_ids[:199]
                        attention_masks = attention_masks[:199]
                        new_labels = new_labels[:199]

                    input_ids.append(eos)
                    attention_masks.append(1)
                    new_labels.append(-1)

                    pad_size = 200 - len(input_ids)

                    input_ids += [pad] * pad_size
                    attention_masks += [0] * pad_size
                    new_labels += [-1] * pad_size

                    dataset_param_new.append({'input_ids': input_ids,
                                                      'attention_mask': attention_masks,
                                                      'labels': new_labels
                                                      })

        print(len(dataset_param_new))
        processed_data[split] = dataset_param_new

    # TODO for all params

    from transformers import AutoConfig, AutoModelWithHeads

    config = AutoConfig.from_pretrained(
        "xlm-roberta-base",
        num_labels=1,
        problem_type="regression"
    )
    model = AutoModelWithHeads.from_pretrained(
        "xlm-roberta-base",
        config=config,
    )

    # from transformers import RobertaConfig, RobertaModelWithHeads
    #
    # config = RobertaConfig.from_pretrained(
    #     "roberta-base",
    #     num_labels=1,
    #     problem_type="regression"
    # )
    #
    # model = RobertaModelWithHeads.from_pretrained('roberta-base', config=config)
    #
    # ner_adapter = model.load_adapter("AdapterHub/roberta-base-pf-conll2003", source="hf")
    # model.set_active_adapters(ner_adapter)
    #
    # model.delete_head(ner_adapter)

    from transformers.adapters.composition import Stack

    adapter_name_lang = 'xlm_cmcl_' + single_lang

    model.add_adapter(adapter_name_lang)

    adapter_name_lang2 = 'xlm_cmcl_' + single_lang + '2'

    model.add_adapter(adapter_name_lang2)

    model.add_tagging_head(
        adapter_name_lang2,
        num_labels=1
      )

    # Unfreeze and activate stack setup
    model.active_adapters = Stack(adapter_name_lang, adapter_name_lang2)

    # only train the task adapter
    model.train_adapter([adapter_name_lang, adapter_name_lang2])

    from transformers import TrainingArguments, AdapterTrainer, EvalPrediction

    training_args = TrainingArguments(
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        logging_steps=200,
        output_dir="./training_output_roberta_2adpxlm_" + parameter + "_" + single_lang + '_' + str(seed) + '_' + str(learning_rate) + '_'
                   + str(batch_size) + '_' + str(epochs),
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        #evaluation_strategy='steps',
        #eval_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        save_on_each_node=True,
        seed=seed
    )

    def compute_accuracy(p: EvalPrediction):
      preds = p.predictions
      return {"acc": (np.abs(preds.squeeze() - p.label_ids)).mean()}


    subset_size = -1 # TODO

    if subset_size == -1:
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_data['train'],
            eval_dataset=processed_data['val'],
            compute_metrics=compute_accuracy,
        )
    else:
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_data['train'][:subset_size],
            eval_dataset=processed_data['val'][:subset_size],
            compute_metrics=compute_accuracy,
        )

    """Start the training 🚀"""

    trainer.train()

    """Looks good! Let's evaluate our adapter on the validation split of the dataset to see how well it learned:"""

    trainer.evaluate()

    model.save_adapter("./final_2adapter_roberta_xlm_" + parameter + "_" + single_lang + '_' + str(seed) + '_' + str(learning_rate) + '_'
                       + str(batch_size) + '_' + str(epochs), 'xlm_cmcl_' + single_lang)
    model.save_adapter("./final_2adapter_roberta_xlm_" + parameter + "_" + single_lang + '_' + str(seed) + '_' + str(
        learning_rate) + '_'
                       + str(batch_size) + '_' + str(epochs), 'xlm_cmcl_' + single_lang + '2')

    # TODO TEST set convert to id actual token etc. labels for answers
    datapaths_tests = {'test1': 'data/test_data_subtask1/test_data_subtask1/test.csv',
                       'test2': 'data/test_data_subtask2/test_data_subtask2/test.csv'}

    for split in datapaths_tests:

        print(split)

        if split == 'test1':
            test_text = parameter + '_test_1_2adpxlm_' + single_lang + '_' + str(seed) + '_' + str(learning_rate) + '_' \
                        + str(batch_size) + '_' + str(epochs) + '.txt'
        elif split == 'test2':
            test_text = parameter + '_test_2_2adpxlm_' + single_lang + '_' + str(seed) + '_' + str(learning_rate) + '_' \
                        + str(batch_size) + '_' + str(epochs) + '.txt'

        with open(test_text, 'w') as fts:
            path = datapaths_tests[split]
            dataset_param = defaultdict(dict)

            with open(path, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=",")

                for idx, line in enumerate(csvreader):

                    if idx == 0:
                        print(idx, line)  # header

                    else:
                        # test
                        language, sentence_id, word_id, word = line
                        # print(word)

                        if language == single_lang:
                            if sentence_id in dataset_param[language]:
                                dataset_param[language][sentence_id]['text'].append(word)

                            else:
                                dataset_param[language][sentence_id] = {'text': [word]}

            for lang in dataset_param:
                for sent_id in dataset_param[lang]:

                    mapping_idx = {}

                    test_sent = dataset_param[lang][sent_id]['text']
                    input = tokenizer(test_sent, add_special_tokens=False)['input_ids']

                    if len(input) > 198:
                        # truncate
                        input = input[:198]
                        print(len(test_sent))
                    #print(sos)
                    #print(input)

                    flattened_input = []
                    count_fi = 0

                    for ip in range(len(input)):

                        for fwp in input[ip]:

                            flattened_input.append(fwp)
                            mapping_idx[count_fi] = ip
                            count_fi += 1

                    final_input = torch.tensor([sos] + flattened_input + [eos]).to(device)
                    output = model(final_input)
                    #print(output.logits.squeeze())

                    final_output = output.logits.squeeze()[1:-1]

                    label_map = defaultdict(list)

                    for fo in range(len(final_output)):

                        actual_id = mapping_idx[fo]
                        #print(fo, actual_id, test_sent[actual_id], final_output[fo])

                        label_map[actual_id].append(final_output[fo].item())

                    # print(label_map)
                    # print(test_sent)
                    print(lang, sent_id)
                    assert len(label_map) == len(test_sent)

                    for lm in label_map:

                        if len(label_map[lm]) > 1:

                            # multiple wordpieces
                            label_map[lm] = np.mean(label_map[lm])

                        else:
                            label_map[lm] = label_map[lm][0]

                    for lll in label_map:

                        #answer2write = f'{lang},{sent_id},{lll},{test_sent[lll]},{label_map[lll]}'
                        answer2write = f'{label_map[lll]}'
                        #print(answer2write)
                        fts.write(answer2write)
                        fts.write('\n')


                # todo only save output once for wordpieces

    #!ls -lh final_adapter

    """**Share your work!**
    
    The next step after training is to share our adapter with the world via _AdapterHub_. [Read our guide](https://docs.adapterhub.ml/contributing.html) on how to prepare the adapter module we just saved and contribute it to the Hub!
    
    ➡️ Also continue with [the next Colab notebook](https://colab.research.google.com/github/Adapter-Hub/adapter-transformers/blob/master/notebooks/02_Adapter_Inference.ipynb) to learn how to use adapters from the Hub.
    """
