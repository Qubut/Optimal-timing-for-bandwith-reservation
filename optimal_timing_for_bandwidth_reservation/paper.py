import torch, math
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler







    
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

def get_batch(source, i):
    inp, target = [], []
    seq_len = min(bsize, len(source) - 1 - i)
    for t in range(i, i+seq_len):
        inp.append(source[t][0].unsqueeze(-1))
        target.append(source[t][1])
    inp = torch.stack(inp, 1)
    target = torch.stack(target).flatten()
    return inp.to(device), target.to(device)

if __name__ == '__main__':
    # loading training data
    df = pd.read_csv('train.csv', sep=',')
    load = df['Lane 1 Flow (Veh/5 Minutes)'].values.astype(float)
    hourly_load = load.reshape((-1, 12)).mean(1)

    # loading test data
    df2 = pd.read_csv('test.csv', sep=',')
    load2 = df2['Lane 1 Flow (Veh/5 Minutes)'].values.astype(float)
    hourly_load2 = load2.reshape((-1, 12)).mean(1)

    # normalizing training and test data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data = scaler.fit_transform(hourly_load.reshape(-1, 1))
    train_data = torch.FloatTensor(train_data).flatten()
    test_data = scaler.transform(hourly_load2.reshape(-1, 1))
    test_data = torch.FloatTensor(test_data).view(-1, delta).to(device)
    
    train_inout_seq = create_inout_sequences(train_data, delta)
    
    # Defining LSTM model
    lstm_model = LSTM().to(device)
    lstm_loss = nn.MSELoss()
    lstm_opt = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
    
    # training LSTM model
    for i in range(epochs):
        tot_loss = 0.0
        for s in range(0, 623, bsize):
            seq, labels = get_batch(train_inout_seq, s)
            y_pred = lstm_model(seq).squeeze()
            
            loss = lstm_loss(y_pred, labels)
            lstm_opt.zero_grad()
            loss.backward()
            lstm_opt.step()
            tot_loss += loss.item()

        if i%freq_printing == 0:
            print(f'epoch: {i:3} loss: {tot_loss:10.8f}')

    # Defining Transformer model
    transformer_model = TransformerModel(100, 10, 10, 1, 0.2).to(device)    
    transformer_loss = nn.MSELoss()
    transformer_opt = torch.optim.AdamW(transformer_model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(transformer_opt, 1.0, gamma=0.95)
    src_mask = transformer_model.generate_square_subsequent_mask(delta).to(device)

    # Training Transformer model
    for i in range(epochs):
        tot_loss = 0.0
        for s in range(0, 623, bsize):
            seq, labels = get_batch(train_inout_seq, s)
            y_pred = transformer_model(seq, src_mask).squeeze()

            loss = transformer_loss(y_pred, labels)
            transformer_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), 0.7)
            transformer_opt.step()
            tot_loss += loss.item()
    
        if i%freq_printing == 0:
            print(f'epoch: {i:3} loss: {tot_loss:10.8f}')
        scheduler.step()