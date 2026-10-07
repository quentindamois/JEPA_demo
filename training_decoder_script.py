import torch
import argparse
torch.manual_seed(42)
import numpy as np
from dataset_img_rec import dataset_img_rec
from torch.utils.data import DataLoader
from jepa import vision_encoder
from tqdm import tqdm
from vision_decoder import vision_decoder

# Accelerator identification
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")


prediction_error_weight = 1
prediction_error = torch.nn.MSELoss()
epsilon = 1e-6

criterion = lambda a, b: prediction_error_weight * prediction_error(a, b)


def collate_function(list_dict_batch):
    return {
        "input": torch.nn.utils.rnn.pad_sequence([element["input"] for element in list_dict_batch], batch_first=True, padding_value=-np.inf),
        "target": torch.nn.utils.rnn.pad_sequence([element["target"] for element in list_dict_batch], batch_first=True, padding_value=-np.inf)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script is use to to trained the decoder to transforme the embedding into the images")
    parser.add_argument("-f", "--path_data_file", help="The path to the data")
    parser.add_argument("-c", "--trained_encoder_path", help="The path to the trained encoder")
    parser.add_argument("-e", "--epochs", help="The number of epoch to train", type=int)
    parser.add_argument("-d", "--hidden_dim", help="The embbeding dimension", type=int, default=128)
    parser.add_argument("-b", "--batch_size", help="Specify the batch size", type=int, default=8)
    parser.add_argument("-w", "--num_workers", help="Specify the number of worker", type=int, default=0)
    arg = parser.parse_args()


    base_data = np.load(arg.path_data_file)
    training_dataset = dataset_img_rec(base_data['train'])
    validation_dataset = dataset_img_rec(base_data['val'])
    
    training_dataloader = DataLoader(training_dataset, arg.batch_size, shuffle=True, num_workers=arg.num_workers, collate_fn=collate_function)
    validation_dataloader = DataLoader(training_dataset, arg.batch_size, shuffle=False, num_workers=arg.num_workers, collate_fn=collate_function)
    tuple_dim_img = base_data['train'].shape[2:]
    emb_encoder = vision_encoder(tuple_dim_img, arg.hidden_dim)
    emb_encoder.load_state_dict(torch.load("emb_encoder.pth", weights_only=True))
    for param in emb_encoder.parameters():
        param.requires_grads = False
    emb_encoder = emb_encoder.to(device)
    emb_encoder.eval()

    emb_decoder = vision_decoder(tuple_dim_img, arg.hidden_dim)
    emb_decoder = emb_decoder.to(device)
    learining_rate = 1e-6
    optim = torch.optim.AdamW(emb_decoder.parameters(), lr=learining_rate)
    for epoch in range(arg.epochs):
        emb_decoder.train()
        sum_train_loss = 0
        ### Training the model ###
        c = None
        for train_batch in tqdm(training_dataloader):
            input_tensor = train_batch["input"].to(device)
            target_tensor = train_batch["target"].to(device)
            input_emb = emb_encoder(input_tensor)
            rec_tensor = emb_decoder(input_emb)
            loss = criterion(rec_tensor, target_tensor)
            
            loss.backward()
            
            optim.step()
            sum_train_loss += loss.item()
        # Updating the trainer encoder at the end of each epochs

        ### Validation data ###
        emb_decoder.eval()
        sum_val_loss = 0
        with torch.no_grad():
            for val_batch in tqdm(validation_dataloader):
                input_tensor = val_batch["input"].to(device)
                target_tensor = val_batch["target"].to(device)
                input_emb = emb_encoder(input_tensor)
                rec_tensor = emb_decoder(input_emb)
                loss = criterion(rec_tensor, target_tensor)
                sum_val_loss += loss.item()
        print(f"Epoch {epoch} train loss : {sum_train_loss} val loss : {sum_val_loss}")
    print(f"Final eval loss : {sum_val_loss}")
    torch.save(emb_decoder.state_dict(), "emb_decoder.pth")