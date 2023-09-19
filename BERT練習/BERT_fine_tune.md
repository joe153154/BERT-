import os
import re
import csv
import time
import torch
import numpy as np
import pandas as pd
import transformers
import torch.nn as nn
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

def super_parameter():
    setting_batch_size = 32
    setting_epochs = 1
    setting_lr = 1e-4
    setting_max_len = 128
    if __name__ == '__main__':
        print(f"batch_size: {setting_batch_size}, epochs: {setting_epochs}, learning_rate: {setting_lr}")
    return setting_batch_size, setting_epochs, setting_lr, setting_max_len

def data_process(id):
    data = load_dataset(id)
    df_1  = pd.DataFrame(data["train"])
    df_2 = pd.DataFrame(data["test"])
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = df.dropna()
    df = shuffle(df).reset_index(drop=True)
    df['label'] = df['label']
    return df

def tokenized(df, tokenizer, max_len):
    sentences = df.text.values
    labels = df.label.values
    input_ids = []
    attention_masks = []
    for sentence in sentences:
        encoded_sentence = tokenizer.encode(sentence, add_special_tokens=True)
        input_ids.append(encoded_sentence)
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    for sentence in input_ids:
        att_mask = [int(token_id > 0) for token_id in sentence]
        attention_masks.append(att_mask)
    train_inputs, valid_inputs, train_labels, valid_labels = train_test_split(input_ids, labels, random_state=42, train_size=0.8)
    train_masks, valid_masks = train_test_split(attention_masks, random_state=42, train_size=0.8)
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(train_inputs, train_labels,random_state=42, train_size=0.75)
    train_masks, test_masks = train_test_split(train_masks, random_state=42, train_size=0.75)
    return train_inputs, train_labels, train_masks, valid_inputs, valid_labels, valid_masks,test_inputs, test_labels, test_masks

def data_to_dataloader(inputs, labels, masks, batch_size):
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    masks = torch.tensor(masks)
    dataset = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader

def changeTarget(e):
    categories = ["0", "1"]
    return categories.index(e)

def getLabel(labels):
    categories = ["Positive", "negative"]
    return categories[labels]

def flat_accuracy(pred, label):
    pred_flat = np.argmax(pred, axis=1).flatten()
    label_flat = label.flatten()
    return np.sum(pred_flat == label_flat) / len(label_flat)

def train(model, device, dataloader, optimizer, scheduler):
    total_loss = 0
    model.train()
    r = tqdm(dataloader)
    for _, data in enumerate(r, 0):
        input_ids = data[0].to(device)
        input_mask = data[1].to(device)
        labels = data[2].to(device)
        model.zero_grad()
        outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(dataloader)
    #print(f"Average training loss: {avg_train_loss}")

def valid(model, device, dataloader):
    print(f'Running Validation.........')
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    model.eval()
    with torch.no_grad():
        t = tqdm(dataloader)
        for _, data in enumerate(t, 0):
            input_ids = data[0].to(device, dtype=torch.long)
            masks = data[1].to(device, dtype=torch.long)
            labels = data[2].to(device, dtype=torch.long)
            with torch.no_grad():
                outputs = model(input_ids, token_type_ids=None, attention_mask=masks)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
        #print(f"Average valid loss: {eval_loss}")

def prediction(model, device, dataloader):
    predictions, true_labels = [], []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend(np.argmax(logits, axis=1))
        true_labels.extend(label_ids)
        return predictions, true_labels

def main():
    categories = ["Positive", "negative"]
    # 設定超參數亂數種子
    Seed = 42
    torch.manual_seed(Seed)
    np.random.seed(Seed)
    batch_size, epochs, lr, max_len = super_parameter()

    # 調用GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        device = torch.device(f'cuda:{0}')
        print("using the GPU{%s}" % torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("using the CPU")

    # 載入BERT模型和其Tokenizer
    model_id = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained(model_id, num_labels = len(categories), )
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(model_id)

    # 載入並處理數據集
    dataset_id = "imdb"
    df = data_process(dataset_id)
    print(df)
    train_inputs, train_labels, train_masks, valid_inputs, valid_labels, valid_masks,test_inputs, test_labels, test_masks = tokenized(df, tokenizer, max_len)
    train_dataloader = data_to_dataloader(train_inputs, train_labels, train_masks, batch_size)
    valid_dataloader = data_to_dataloader(valid_inputs, valid_labels, valid_masks, batch_size)
    test_dataloader = data_to_dataloader(test_inputs, test_labels, test_masks, batch_size)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)

    for epoch in range(epochs):
        train(model, device, train_dataloader, optimizer, scheduler)
        valid(model, device, valid_dataloader)

    prediction_labels, true_labels = prediction(model, device, test_dataloader)
    result_df = pd.DataFrame({'true_category': true_labels, 'predicted_category': prediction_labels})
    result_df['true_category_index'] = result_df['true_category'].map(getLabel)
    result_df['predicted_category_index'] = result_df['predicted_category'].map(getLabel)
    result_df.head()

    print(f"Accuracy is {accuracy_score(result_df['true_category'], result_df['predicted_category'])}")

    confusion_mat = confusion_matrix(y_true=result_df['true_category_index'],
                                     y_pred=result_df['predicted_category_index'], labels=categories)

    df_cm = pd.DataFrame(confusion_mat, index=categories, columns=categories)
    sns.heatmap(df_cm, annot=True)
    plt.show()

if __name__ == "__main__":
    main()
