import os
import pandas as pd
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # For CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import openai

openai.api_key = "Your Api Key"
import nltk

import time

path = "Path to the dataset"

filename_dataset="sample_data"

data = pd.read_excel(path+filename_dataset+".xlsx")


dataset_full=[]

titles = data["chapter title"].values
questions = data["Question Type: SC"].values
video_ids =  data["video_id"].values
context = data["context"].values
video_titles =  data["video_title"].values
image_captions = data["image_captions"].values
start_time = data["start_time"].values
for i in range(len(questions)):

    
    video_id = video_ids[i]
    split_text =  questions[i].split("?/")
    if "?" not in  split_text[0] and  split_text[0] != "NA":
        questions[i] = [str(split_text[0])+"?"]

    else:
        questions[i] = split_text

    
     dataset_full.append({"sentence":titles[i],"context": str(context[i]),"question":questions[i][0], "video_title": video_titles[i], "image_caption": image_captions[i], "video_id": video_id,"start_time": start_time[i]})
    



data["summary"]=None

import re


def response_chat(prompt):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.7,
    max_tokens=1024)
    # print the chat completion
    input_text = response['choices'][0]['message']['content']
    return input_text



prompt_directive_1 = "Use the given video transcript and frame captions to describe what is likely happening in the video. Do not add any new keywords that are not present in the given information:\n"

prompt_directive_2 = "Use the given video title and frame captions to describe what is likely happening in the video. Do not add any new keywords that are not present in the given information:\n"


video_trans = "Video Transcript: "

frame_captions = "Frame Captions: "


video_title_ = "Video Title: "


# Iterate over the dataset and make API requests
for i in dataset_full:
    retries = 0
    while retries < 999:  # Retry a maximum of 999 times
        try:
             
            if str(i["context"]) == "nan":
                # print("str(i[context]) is empty:", str(i["video_title"]))
                prompt = str(prompt_directive_1 + video_trans + str(i["context"]) + "\n" + frame_captions + str(i["image_caption"]))
                
            else:
                
                prompt = str(prompt_directive_2 + video_title_ + str(i["context"]) + "\n" + frame_captions + str(i["image_caption"]))
            
            res = response_chat(prompt)
            data.loc[(data["video_id"] == i["video_id"]) & (data["start_time"] == i["start_time"]), "summary"] = str(res)
            break  # Break out of the retry loop if successful
        except Exception as e:
            print(f"Error processing data: {e}")
            time.sleep(15)  # Wait for 15 seconds before retrying
            retries += 1
            print("retries: ",retries)


data.to_excel(path+filename_dataset+"_SUMM.xlsx")

print("DONE")
