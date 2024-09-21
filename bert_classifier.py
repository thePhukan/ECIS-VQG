import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np

num_epochs = 100

path = "The path to the dataset"

bert_model = "bert-base-uncased"
filename_model= bert_model+"LAST6"+str(num_epochs)
MODEL_PATH_CHECKPOINT = path+"/Model Path/"+filename_model+"_Loss_Checkpoints.pt"
MODEL_PATH = path+"/Model Path/"+filename_model+".pt"

is_cuda = torch.cuda.is_available()
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    

print("\nDEVICE:\t",device)
import pickle


# build word index
filename_dataset = "sample_data"
data = pd.read_excel(path+filename_dataset+".xlsx")

from sklearn.model_selection import train_test_split
titles = data["title"].values
class_labels = data["class"].values
X_train, X_test, y_train, y_test = train_test_split(titles,class_labels, test_size=0.2, stratify=class_labels, random_state=42)

df_y_test = pd.DataFrame(y_test, columns = ['class'])

print("y_test: \n",df_y_test['class'].value_counts(ascending=True))

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.2, stratify=y_train, random_state=42)

n_samples = len(y_train)
df_y_train = pd.DataFrame(y_train, columns = ['class'])

temp = df_y_train['class'].value_counts()
n_samples_UNC = temp["UNC"]
n_samples_SC = temp["SC"]
n_samples_NSC = temp["NSC"]
n_samples_UL = temp["UL"]
print("y_train: \n",temp)


df_y_val = pd.DataFrame(y_val, columns = ['class'])

print("y_val: \n",df_y_val['class'].value_counts(ascending=True))

from tqdm.notebook import tqdm

word2index, cnt = {}, 0
for line in data["title"]:
    line = line.split('__')[0]
    # print(line)
    for word in line.split(' '):
        if word not in word2index:
            word2index[word] = cnt
            cnt += 1


from transformers import BertTokenizer, BertModel, DistilBertTokenizer

MAX_LEN = 500

tokenizer = BertTokenizer.from_pretrained(bert_model)


def create_sequence(word2index, mode):   # returns index for words and its y label
    sequence, y_seq, y_seq2 = [], [], []
    for s in tqdm(mode, desc='Split and tokenize'):
        text = s.split('__split__')[0]
        one_seq = []
        words = text.split(' ')
        
        global MAX_LEN
        if len(words) > MAX_LEN:
            MAX_LEN = len(words)

        for w in words:
            if w in word2index.keys():
                one_seq.append(word2index[w])
            else:
                one_seq.append(1)
    
        sequence.append(tokenizer(text, padding='max_length', max_length=MAX_LEN, return_tensors='pt'))
    return sequence

# from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformersDataset(Dataset):
    def __init__(self, seq, lb1):
        super(TransformersDataset, self).__init__()
        self.seq = seq
        self.lb1= lb1
        
    def __getitem__(self, idx):
        return self.seq[idx], self.lb1[idx]
    
    def __len__(self):
        return self.seq.__len__()
        
batch_size = 2
sequence = create_sequence(word2index, X_train)
y_seq = y_train
label_seq = list(sorted(set(y_seq))) # sorting labels to give it indices
print("label_seq: ",label_seq)


n_classes = len(label_seq)
def get_label_id1(label):
        return label_seq.index(label)
    
def createlabelseq1(countries): # convert labels in indices
    country_ids = [get_label_id1(country) for country in countries]
    return torch.LongTensor(country_ids)

labels1 = createlabelseq1(y_seq).to(device)


train_data = TransformersDataset(sequence, labels1)
loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


num_classes1 = n_classes 


learning_rate = 0.001 # 3e-4
input_size = 300
hidden_size = 768
num_layers = 5

sequence = create_sequence(word2index, X_val)
y_seq = y_val
labels1 = createlabelseq1(y_seq).to(device)

val_data = TransformersDataset(sequence, labels1)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

from transformers import BertForPreTraining

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bert = BertModel.from_pretrained(bert_model)

        
        for param in self.bert.parameters():
            param.requires_grad = True            
        
        self.fc = nn.Linear(6 * hidden_size, 2*hidden_size)
        self.linear1 = nn.Linear(2*hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, num_classes1)
        self.act = nn.ReLU()
        self.d = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.bert(**x, output_hidden_states=True)

        # Last 6 Layer
        x = torch.cat((x[2][-1][:, 0, ...],
                       x[2][-2][:, 0, ...],
                       x[2][-3][:, 0, ...],
                       x[2][-4][:, 0, ...],
                       x[2][-5][:, 0, ...],
                       x[2][-6][:, 0, ...]), -1)


        x=  self.fc(x)
        x= self.d(self.act(x))
        # x= self.bn_1(x)

        _out1 = self.linear1(x)
        _out1 = self.d(self.act(_out1))

        
    
        out1 = self.fc1(_out1)


        return out1

model = RNN(input_size, hidden_size, num_layers).to(device)



file = open(path+"/Model Path/"+'epochs_'+filename_model+'.txt','w')


# Class Weights

weight_0_NSC = np.float(n_samples/(n_classes * n_samples_NSC))
weight_1_SC = np.float(n_samples/(n_classes * n_samples_SC))
weight_2_UL = np.float(n_samples/(n_classes * n_samples_UL))
weight_3_UNC = np.float(n_samples/(n_classes * n_samples_UNC))

weight = torch.tensor([weight_0_NSC, weight_1_SC, weight_2_UL, weight_3_UNC]) 


criterion1 = nn.CrossEntropyLoss(weight=weight)

model = model.to(device)
criterion1 = criterion1.to(device)

# init optimizer and scheduler
from transformers import AdamW, get_linear_schedule_with_warmup

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)


best_model, best_loss = model, 10000
checkpoint=99999999
batch_idx = 0
for epoch in range(num_epochs):
    batch_idx = batch_idx + 1
    train_loss, val_loss = 0., 0.
    
    model.train()
    for idx, (x, y) in enumerate(tqdm(loader, desc=f'Epoch train {epoch}')):
        # print("len(x)",x)
        optimizer.zero_grad()
        x = {k: v.squeeze(1).to(device) for k, v in x.items()}

        

        output1= model(x)
        loss1 = criterion1(output1, y)
        
        loss = loss1
        loss.backward()
    
        optimizer.step()
        if idx % 100 == 0:
            scheduler.step()
            
        if checkpoint >= loss.item():

            checkpoint=loss.item()
            torch.save(model, MODEL_PATH_CHECKPOINT)
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
    
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f'Epoch val {epoch}'):
            x = {k: v.squeeze(1).to(device) for k, v in x.items()}
            
            output1 = model(x)
            loss1 = criterion1(output1, y)
           
            loss = loss1
            
            val_loss = val_loss + ((1 / (batch_idx + 1)) * (loss.data - val_loss))
            
    if val_loss < best_loss:
        best_loss, best_model = val_loss, model
        
    display_text_epoch = 'Epoch [{}/{}], Train loss: {:.4f} - Val loss: {:.4f}\n'.format(epoch + 1, num_epochs, train_loss, val_loss)
    print(display_text_epoch)
    file.write(display_text_epoch)

model = best_model
sequence= create_sequence(word2index, X_test)
y_seq = y_test
labels1 = createlabelseq1(y_seq).to(device)

dataset = TransformersDataset(sequence, labels1)
test_loader = DataLoader(dataset, batch_size=batch_size)

predicted_list1 = []
predicted_list2 = []
actual_list1 = []
actual_list2 = []

with torch.no_grad():    
    n_correct1 = 0
    n_correct2 = 0
    n_correct = 0
    n_samples = 0
  
    for x, y in tqdm(test_loader, desc='Evaluating'):
        x = {k: v.squeeze(1).to(device) for k, v in x.items()}
        output1 = model(x)
        _, predicted1 = torch.max(output1.data, 1)
        predicted_list1.append(predicted1.tolist())
        n_samples += y.size(0)
        n_correct1 += (predicted1 == y).sum().item()
       
        acc1 = 100.0 * n_correct1 / n_samples


        n_correct += ((predicted1 == y)).sum().item()
        acc = 100.0 * n_correct / n_samples

        actual_list1.append(y.tolist())

print(f'Accuracy of the network on category is: {acc1} %')

def flat(ini_list):
    return list(numpy.concatenate(ini_list).flat)

torch.save(model,MODEL_PATH)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    print("\n Triggered!!!!!!!!!\n")
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(path+"/Model Path/"+filename_model+'_cm.png', bbox_inches='tight')
    plt.show()

import numpy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(flat(predicted_list1), flat(actual_list1))
print("###############################before_cm######################################")
print(cm)
print("#################################after_cm####################################")
target_names = label_seq
plot_confusion_matrix(cm, target_names)

from sklearn.metrics import classification_report
print("#####################################################################")
print(classification_report(flat(predicted_list1), flat(actual_list1), target_names=label_seq))
file.write(classification_report(flat(predicted_list1), flat(actual_list1), target_names=label_seq))
print("#####################################################################")
print(filename_model+" saved")
print("#####################################################################")

file.close()
