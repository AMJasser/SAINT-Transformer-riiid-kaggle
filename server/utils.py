import yaml
import torch
import pickle
import sys
import numpy as np

import torch.nn.functional as F

import pandas as pd

# supress warnings
import warnings
warnings.filterwarnings("ignore")

group = pd.read_pickle("../data/processed/inference_group")

with open('../data/processed/inference_last_timestamp.pickle', 'rb') as handle:
    last_timestamp = pickle.load(handle)

boundaries = [120,600,1800,3600,10800,43200,86400,259200,604800]


# Loading the model
sys.path.append('..')

from src.models.model import TransformerModel


with open('../config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


# Transformer hyperparameter
d_model = config["d_model"]

decoder_layers = config["decoder_layers"]
encoder_layers = config["encoder_layers"]


correct_start_token = config["correct_start_token"]
user_answer_start_token = config["user_answer_start_token"]
seq_len = config["seq_len"]

dropout = config["dropout"]
ff_model = d_model*4
att_heads = d_model // 64


# Loading questions, and every question corresponding part
que_data = pd.read_csv("../data/raw/questions.csv")
part_valus = que_data.part.values
unique_ques = len(que_data)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
part_valus = torch.LongTensor(part_valus).to(device)
que_emb_size = unique_ques

model = TransformerModel(que_emb_size, hidden=d_model, part_arr=part_valus, dec_layers=decoder_layers,
                         enc_layers=encoder_layers, dropout=dropout, nheads=att_heads, ff_model=ff_model).to(device)

model.load_state_dict(torch.load("../models/model_best.torch"))
model.eval()

# Recreates the timestamp encoding
def get_timestamp(ts, user_id):

    if last_timestamp.get(user_id, -1) == -1:
        return 0

    diff = (ts - last_timestamp[user_id])/1000

    if diff < 0:
        return 0

    if diff <= 60:
        return int(diff)

    for i, boundary in enumerate(boundaries):
        if boundary > diff:
            break

    if i == len(boundaries) - 1:
        return 60+i+1

    return 60+i


# Input must be (eval_batch, 3): ["user_id", "content_id", "content_type_id", "timestamp"]
def pred_users(vals):

    eval_batch = vals.shape[0]

    tensor_question = np.zeros((eval_batch, seq_len), dtype=np.long)
    tensor_answers = np.zeros((eval_batch, seq_len), dtype=np.long)
    tensor_ts = np.zeros((eval_batch, seq_len), dtype=np.long)
    tensor_user_answer = np.zeros((eval_batch, seq_len), dtype=np.long)

    val_len = []
    preds = []
    group_index = group.index

    for i, line in enumerate(vals):

        if line[2] == True:
            val_len.append(0)
            continue

        user_id = line[0]
        question_id = line[1]
        # Compute timestamp difference correctly
        timestamp = get_timestamp(line[3], user_id)

        que_history = np.array([], dtype=np.int32)
        answers_history = np.array([], dtype=np.int32)
        ts_history = np.array([], dtype=np.int32)
        user_answer_history = np.array([], dtype=np.int32)

        if user_id in group_index:

            cap = seq_len-1
            que_history, answers_history, ts_history, user_answer_history = group[user_id]

            que_history = que_history[-cap:]
            answers_history = answers_history[-cap:]
            ts_history = ts_history[-cap:]
            user_answer_history = user_answer_history[-cap:]

        # Decoder data, add start token
        answers_history = np.concatenate(
            ([correct_start_token], answers_history))
        user_answer_history = np.concatenate(
            ([user_answer_start_token], user_answer_history))

        # Decoder data
        que_history = np.concatenate(
            (que_history, [question_id]))  # Add current question
        ts_history = np.concatenate((ts_history, [timestamp]))

        tensor_question[i][:len(que_history)] = que_history
        tensor_answers[i][:len(que_history)] = answers_history
        tensor_ts[i][:len(que_history)] = ts_history
        tensor_user_answer[i][:len(que_history)] = user_answer_history

        val_len.append(len(que_history))

    tensor_question = torch.from_numpy(tensor_question).long().T.to(device)
    tensor_answers = torch.from_numpy(tensor_answers).long().T.to(device)
    tensor_ts = torch.from_numpy(tensor_ts).long().T.to(device)
    tensor_user_answer = torch.from_numpy(
        tensor_user_answer).long().T.to(device)

    with torch.no_grad():  # Disable gradients so prediction runs faster
        out = F.sigmoid(model(tensor_question, tensor_answers,
                        tensor_ts, tensor_user_answer)).squeeze(dim=-1).T

    for j in range(len(val_len)):
        preds.append(out[j][val_len[j]-1].item())

    return preds
