import json
from datasets import load_metric,Dataset,DatasetDict
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, BartForConditionalGeneration
import os
from transformers import GPT2Tokenizer,  GPTNeoForCausalLM, GPT2LMHeadModel,T5Tokenizer, T5Model,T5ForConditionalGeneration
import pandas as pd
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # For CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['WANDB_DISABLED']="true"
from tqdm import tqdm
from transformers import GPT2Tokenizer,  GPTNeoForCausalLM, GPT2LMHeadModel,T5Tokenizer, T5Model,T5ForConditionalGeneration
import pandas as pd
import os
from tqdm import tqdm
# !pip install transformers
from copy import deepcopy

import torch
torch.cuda.empty_cache()
# print(torch.cuda.memory_summary(device=None, abbreviated=False))
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import json
import wandb
from transformers.tokenization_utils_base import PaddingStrategy

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

print("\nDEVICE:\t",device)
model_checkpoint = "facebook/bart-large"
metric = load_metric("rouge.py")

TEST_SUMMARY_ID = 1

def transform_single_dialogsumm_file(file):
    
    # result = {"sentence":[],"context":[],"question":[]}                                       # (Chapter Title, Transcript) 
    # result = {"sentence":[],"question":[]}                                                    # (Chapter Title)
    # result = {"sentence":[],"context":[],"question":[], "video_title":[]}                     # (Chapter Title, Video Title, Transcript)
    # result = {"sentence":[],"video_title":[],"question":[]}                                   # (Chapter Title, Video Title)
    # result = {"sentence":[],"image_caption":[],"question":[]}                                 # (Chapter Title, Frame Captions)
    # result = {"sentence":[],"video_title":[],"image_caption":[],"question":[]}                # (Chapter Title, Video Title, Frame Captions)
    # result = {"sentence":[],"context":[],"video_title":[],"image_caption":[],"question":[]}   # (Chapter Title, Transcript, Frame Captions, Video Title)              
    result = {"sentence":[],"video_title":[],"question":[],"summary":[]}                        # (Chapter Title, Video Title, Summary)
    
    for i in range(len(file)):
        
        # print("\n")
        result["sentence"].append(file[i]["sentence"])                                          # Chapter Title (C)
        # result["context"].append(file[i]["context"])                                          # Transcript (T)
        result["question"].append(file[i]["question"])                                          # Gold Question 
        result["video_title"].append(file[i]["video_title"])                                    # Video Title (V)
        # result["image_caption"].append(str(file[i]["image_caption"]))                         # Image Caption (F)
        result["summary"].append(str(file[i]["summary"]))                                       # Summary(F,T) (S)

    return Dataset.from_dict(result)

def transform_test_file(file):
    
    # result = {"sentence":[],"context":[],"question":[]}                                       # (Chapter Title, Transcript) 
    # result = {"sentence":[],"question":[]}                                                    # (Chapter Title)
    # result = {"sentence":[],"context":[],"question":[], "video_title":[]}                     # (Chapter Title, Video Title, Transcript)
    # result = {"sentence":[],"video_title":[],"question":[]}                                   # (Chapter Title, Video Title)
    # result = {"sentence":[],"image_caption":[],"question":[]}                                 # (Chapter Title, Frame Captions)
    # result = {"sentence":[],"video_title":[],"image_caption":[],"question":[]}                # (Chapter Title, Video Title, Frame Captions)
    # result = {"sentence":[],"context":[],"video_title":[],"image_caption":[],"question":[]}   # (Chapter Title, Transcript, Frame Captions, Video Title)              
    result = {"sentence":[],"video_title":[],"question":[],"summary":[]}                        # (Chapter Title, Video Title, Summary)
    
    for i in range(len(file)):
        
        # print("\n")
        result["sentence"].append(file[i]["sentence"])                                          # Chapter Title (C)
        # result["context"].append(file[i]["context"])                                          # Transcript (T)
        result["question"].append(file[i]["question"])                                          # Gold Question 
        result["video_title"].append(file[i]["video_title"])                                    # Video Title (V)
        # result["image_caption"].append(str(file[i]["image_caption"]))                         # Image Caption (F)
        result["summary"].append(str(file[i]["summary"]))                                       # Summary(F,T) (S)

 
    return Dataset.from_dict(result)

def transform_dialogsumm_to_huggingface_dataset(train,validation,test):

    train = transform_single_dialogsumm_file(train)
    validation = transform_single_dialogsumm_file(validation)
    test = transform_test_file(test)
    return DatasetDict({"train":train,"validation":validation,"test":test})




max_input_length = 1024 #2600 #516 #4600 #256 #1024

num_epochs = 50 
batch_size = 2 # 4
path = "Path to Dataset"


bert_model = "BART"

config = "V_SUMM_inTok"+str(max_input_length)+"_bs_"+str(batch_size)
filename_model= bert_model+"_ep_"+str(num_epochs)+"_OUT_1024_"+config

print(filename_model)
wandb.init(save_code=True, name=filename_model, project="VQG")

MODEL_PATH_CHECKPOINT = path+"Model Path/"+filename_model+"_Loss_Checkpoints.pt"

MODEL_PATH = path+"Model Path/"+filename_model
is_cuda = torch.cuda.is_available()


import pickle

#SUMM
filename_dataset = "sample_data"


data = pd.read_excel(path+filename_dataset+".xlsx")


# Filter

for i in data.index:
    if data["class"][i] == "SC":
        data.drop(i , inplace=True)          


    elif data["class"][i] == "UL":
        
        data.drop(i , inplace=True)
        
    elif data["class"][i] == "NSC" and pd.isnull(data['Question Type: SC'][i]) and pd.isnull(data['Question Type: NSC'][i]):
        data.drop(i , inplace=True)
            
    elif data["class"][i] == "NSC" and pd.isnull(data['Question Type: SC'][i]) and pd.notnull(data['Question Type: NSC'][i]):        

            data.at[i,'Question Type: SC'] = str(data["Question Type: NSC"][i])
            
    elif pd.isnull(data['Question Type: SC'][i]) and pd.notnull(data['Question Type: NSC'][i]):        

            data.at[i,'Question Type: SC'] = str(data["Question Type: NSC"][i])
            
            
            
    


indexAge = data[ (data['Question Type: SC'].isnull()) ].index
data.drop(indexAge , inplace=True)
dataset_full=[]

titles = data["chapter title"].values
questions = data["Question Type: SC"].values
video_ids =  data["video_id"].values
context = data["context"].values
video_titles =  data["video_title"].values
image_captions = data["image_captions"].values
summaries = data["summary"].values

for i in range(len(questions)):

    video_id = video_ids[i]

    split_text =  questions[i].split("?/")
    if "?" not in  split_text[0] and  split_text[0] != "NA":
        questions[i] = [str(split_text[0])+"?"]

    else:
        questions[i] = split_text

    
    # dataset_full.append({"sentence":titles[i],"context": str(context[i]),"question":questions[i][0]})                                                                         # (Chapter Title, Transcript)
    # dataset_full.append({"sentence":titles[i],"question":questions[i][0]})                                                                                                    # (Chapter Title)
    # # dataset_full.append({"sentence":titles[i],"question":questions[i][0], "video_title": video_titles[i]})                                                                  # (Chapter Title, Video Title)
    # dataset_full.append({"sentence":titles[i],"question":questions[i][0], "image_caption": image_captions[i]})                                                                # (Chapter Title, Frame Captions)
    # dataset_full.append({"sentence":titles[i],"question":questions[i][0], "image_caption": image_captions[i], "video_title": video_titles[i]})                                # (Chapter Title, Video Title, Frame Captions)
    # dataset_full.append({"sentence":titles[i],"context": str(context[i]),"question":questions[i][0], "video_title": video_titles[i]})                                         # (Chapter Title, Video Title, Transcript)
    # dataset_full.append({"sentence":titles[i],"context": str(context[i]),"question":questions[i][0], "video_title": video_titles[i], "image_caption": image_captions[i]})     # (Chapter Title, Transcript, Frame Captions, Video Title)
    dataset_full.append({"sentence":titles[i],"summary": str(summaries[i]),"question":questions[i][0],"video_title": video_titles[i]})                                          # (Chapter Title, Video Title, Summary)
 



# """
from sklearn.model_selection import train_test_split
import random

train_size = 0.8
val_size = 0.1
test_size = 0.1

train_data, val_test_data = train_test_split(dataset_full, train_size=train_size, random_state=42)
val_data, test_data = train_test_split(val_test_data, train_size=val_size/(val_size + test_size), random_state=42)

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")


import pickle

raw_datasets = transform_dialogsumm_to_huggingface_dataset(train_data,val_data,test_data)

# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_target_length = 512 * 2

def preprocess_function(examples):

    # inputs = [doc for doc in examples["sentence"]]    # (Chapter Title)

    # inputs = [str(str(sent)+" "+str(con)) for (sent, con) in zip(examples["sentence"], examples["context"])]  # (Chapter Title, Transcript)

    # inputs = [str(str(sent)+" "+str(v_title)) for (sent, v_title) in zip(examples["sentence"], examples["video_title"])]  # (Chapter Title, Video Title)

    # inputs = [str(str(sent)+" "+str(img_cap)) for (sent, img_cap) in zip(examples["sentence"], examples["image_caption"])]    # (Chapter Title, Frame Captions)

    # inputs = [str(str(sent)+" "+str(v_title)+" "+str(img_cap)) for (sent, v_title, img_cap) in zip(examples["sentence"], examples["video_title"], examples["image_caption"])] # (Chapter Title, Video Title, Frame Captions)

    # inputs = [str(str(sent)+" "+str(con)+" "+str(v_title)) for (sent, con, v_title) in zip(examples["sentence"], examples["context"], examples["video_title"])]   # (Chapter Title, Video Title, Transcript)

    # inputs = [str(str(sent)+" "+str(con)+" "+str(img_cap)+" "+str(v_title)) for (sent, con, v_title, img_cap) in zip(examples["sentence"], examples["context"], examples["video_title"], examples["image_caption"])]  # (Chapter Title, Transcript, Frame Captions, Video Title)

    # inputs = [str(str(sent)+" "+str(img_cap)+" "+str(v_title)) for (sent, v_title, img_cap) in zip(examples["sentence"], examples["video_title"], examples["image_caption"])] # (Chapter Title, Frame Captions, Video Title)

    inputs = [str(str(sent)+" "+str(v_title)+" "+str(summ)) for (sent, v_title, summ) in zip(examples["sentence"], examples["video_title"], examples["summary"])]   # (Chapter Title, Video Title, Summary)

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    label_input =[]
    for qs in examples["question"]:
        label_input.append(qs)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():

        labels = tokenizer(label_input, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
 
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


args = Seq2SeqTrainingArguments(
    "BART-LARGE_"+config+"_ep_"+str(num_epochs),
    evaluation_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    save_strategy="epoch",
    metric_for_best_model="eval_rouge1",
    greater_is_better=True,
    seed=42,
    generation_max_length=max_target_length,
    logging_strategy = "epoch",report_to="wandb"
)



data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

import nltk
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v , 4) for k, v in result.items()}



trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset= tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# trainer.train().to(device)
trainer.train()




# out = trainer.predict(tokenized_datasets["test"],num_beams=5).to(device)
out = trainer.predict(tokenized_datasets["test"], num_beams=5, max_length=max_target_length)

predictions, labels ,metric= out
print(metric)


decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
decoded_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
decoded_labels = [" ".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]


# output summaries on test set
with open(MODEL_PATH+"test_output.txt","w") as f: 
    for i in decoded_preds:
        print(i)
        f.write(i.replace("\n","")+"\n")
