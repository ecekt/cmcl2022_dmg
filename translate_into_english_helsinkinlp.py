import csv
import os
#https://huggingface.co/Helsinki-NLP/opus-mt-da-en

if not os.path.isdir('data/translated'):
    os.mkdir('data/translated')

datapaths_test2 = {'test2': 'data/test_data_subtask2/test_data_subtask2/test.csv'}

datapaths_translated2 = {'test2': 'data/translated/test2_en.csv'}

source_lan = 'da'
target_lan = 'en'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_daen = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-da-en")
model_daen = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-da-en")

# gem-gem for nl-da wasn't working fine
# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-gem-gem")
#
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-gem-gem")
#a sentence initial language token is required in the form of >>id<< (id = valid target language ID)
#

with open(datapaths_translated2['test2'], 'w') as tf:
    csv_tf = csv.writer(tf, delimiter=",")

    for split in datapaths_test2:

        print(split)
        path = datapaths_test2[split]

        with open(path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")

            for idx, line in enumerate(csvreader):

                if idx == 0:
                    print(idx, line)  # header
                    #header = ','.join(line) + '\n'
                    csv_tf.writerow(line)

                else:

                    language, sentence_id, word_id, word = line
                    #print(word)

                    print(idx)
                    if language == 'da':

                        batch = tokenizer_daen([word], return_tensors="pt")
                        gen = model_daen.generate(**batch)  # , num_beams=3)
                        translated_word = tokenizer_daen.batch_decode(gen, skip_special_tokens=True)[0]

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

                    if '...' in translated_word:
                        translated_word = translated_word.split('...')[0]

                    #new_line = f'{target_lan},{sentence_id},{word_id},{translated_word},{FFDAvg},{FFDStd},{TRTAvg},{TRTStd}\n'
                    new_row =[target_lan, sentence_id, word_id, translated_word]

                    csv_tf.writerow(new_row)

