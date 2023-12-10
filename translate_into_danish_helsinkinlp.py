import csv
import os

if not os.path.isdir('data/translated'):
    os.mkdir('data/translated')

datapaths_danish = {'train': 'data/translated/train_da_en.csv',
                 'val': 'data/translated/dev_da_en.csv'}

datapaths = {'train': 'data/training_data2022/training_data/train.csv',
             'val': 'data/training_data2022/training_data/dev.csv'}

germanic = ['en'] #, 'de'] # no nl-da model, 'nl']
target_lan = 'da'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_enda = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-da")
model_enda = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-da")

tokenizer_deda = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-da")
model_deda = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-da")

# gem-gem for nl-da wasn't working fine
# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-gem-gem")
#
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-gem-gem")
#a sentence initial language token is required in the form of >>id<< (id = valid target language ID)
#

with open(datapaths_danish['train'], 'w') as tf:
    csv_tf = csv.writer(tf, delimiter=",")

    with open(datapaths_danish['val'], 'w') as vf:
        csv_vf = csv.writer(vf, delimiter=",")

        for split in datapaths:

            print(split)
            path = datapaths[split]

            with open(path, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=",")

                for idx, line in enumerate(csvreader):

                    if idx == 0:
                        print(idx, line)  # header
                        header = ','.join(line) + '\n'

                        if split == 'train':
                            csv_tf.writerow(header)
                        else:
                            csv_vf.writerow(header)

                    else:
                        #print(line)
                        if split in ['train', 'val']:
                            language, sentence_id, word_id, word, FFDAvg, FFDStd, TRTAvg, TRTStd = line
                            #print(word)

                            if language in germanic:

                                print(idx)
                                if language == 'en':

                                    batch = tokenizer_enda([word], return_tensors="pt")
                                    gen = model_enda.generate(**batch)  # , num_beams=3)
                                    translated_word = tokenizer_enda.batch_decode(gen, skip_special_tokens=True)[0]

                                elif language == 'de':

                                    batch = tokenizer_deda([word], return_tensors="pt")
                                    gen = model_deda.generate(**batch)  # , num_beams=3)
                                    translated_word = tokenizer_deda.batch_decode(gen, skip_special_tokens=True)[0]

                                if len(translated_word.split()) > 1:
                                    # just truncate
                                    if word in translated_word:  # sometimes NERs are not translated correctly but still exist
                                        translated_word = word
                                    else:
                                        translated_word = translated_word.split()[0]

                                    print(word, translated_word)

                                # if ',' in translated_word:
                                #     # comma messes up with the parsing of the csv
                                #     translated_word = f'"{translated_word}"'

                                #new_line = f'{target_lan},{sentence_id},{word_id},{translated_word},{FFDAvg},{FFDStd},{TRTAvg},{TRTStd}\n'
                                new_row =[target_lan, sentence_id, word_id, translated_word, FFDAvg, FFDStd, TRTAvg, TRTStd]

                                if split == 'train':
                                    csv_tf.writerow(new_row)
                                else:
                                    csv_vf.writerow(new_row)

