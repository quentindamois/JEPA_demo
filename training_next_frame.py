import torch
torch.manual_seed(42)
import argparse
from dataset_img import dataset_img
import numpy as np
from torch.utils.data import DataLoader
from jepa import vision_encoder, embedding_predictor
from tqdm import tqdm
from individual_loss_tracker import individual_tracker
from vision_decoder import vision_decoder
# Accelerator identification
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")


# The loss initialization

prediction_error_weight = 1
prediction_error = torch.nn.MSELoss()
epsilon = 1e-6



variance_weight = 0.01
variance_loss = lambda a: torch.nn.functional.hinge_embedding_loss(torch.sqrt(torch.var(a.reshape([-1] + [a.size()[-1]]), dim=-2) + epsilon), torch.ones([a.size()[-1], a.size()[-1]]).to(device))

covariance_loss_weight = 0.01
covariance_loss = lambda a: (1/(a.size()[-1])) * (torch.sum(torch.square(torch.cov(a.reshape([-1] + [a.size()[-1]]).T))) - torch.sum(torch.square(torch.diag(torch.cov(a.reshape([-1] + [a.size()[-1]]).T)))))

ind_tracker = individual_tracker()
ind_tracker_eval = individual_tracker()
criterion = lambda a, b: prediction_error_weight * ind_tracker.inc("prediction_error", prediction_error(a, b))

criterion_eval = lambda a, b: prediction_error_weight * ind_tracker_eval.inc("prediction_error", prediction_error(a, b))

# The collate function

def collate_function(list_dict_batch):
    return {
        "input": torch.nn.utils.rnn.pad_sequence([element["input"] for element in list_dict_batch], batch_first=True, padding_value=-np.inf),
        "target": torch.nn.utils.rnn.pad_sequence([element["target"] for element in list_dict_batch], batch_first=True, padding_value=-np.inf)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script is used for the training")
    parser.add_argument("-f", "--path_data_file", help="The path to the data file.")
    parser.add_argument("-d", "--hidden_dim", help="The embbeding dimension", type=int, default=128)
    parser.add_argument("-b", "--batch_size", help="Specify the batch size", type=int, default=8)
    parser.add_argument("-w", "--num_workers", help="Specify the number of worker", type=int, default=0)
    parser.add_argument("-e", "--epochs", help="The number of epoch to train the model", type=int)
    arg = parser.parse_args()

    base_data = np.load(arg.path_data_file)
    training_dataset = dataset_img(base_data['train'])
    validation_dataset = dataset_img(base_data['val'])

    training_dataloader = DataLoader(training_dataset, arg.batch_size, shuffle=True, num_workers=arg.num_workers, collate_fn=collate_function)
    validation_dataloader = DataLoader(training_dataset, arg.batch_size, shuffle=False, num_workers=arg.num_workers, collate_fn=collate_function)
    # Getting the dimension of the picture
    tuple_dim_img = base_data['train'].shape[2:]
    emb_encoder = vision_encoder(tuple_dim_img, arg.hidden_dim)
    emb_encoder = emb_encoder.to(device)
    next_state_predictor = embedding_predictor(arg.hidden_dim)
    next_state_predictor = next_state_predictor.to(device)
    emb_decoder = vision_decoder(tuple_dim_img, arg.hidden_dim)
    emb_decoder = emb_decoder.to(device)

    # Setting up the optmizers
    learining_rate = 1e-6
    optim = torch.optim.AdamW(list(emb_encoder.parameters()) + list(next_state_predictor.parameters()), lr=learining_rate)

    temperature_softmax_sharp = 1e-7
    # the rate update paremerter(from arxiv:2104.14294v2)
    m = 0.45
    for epoch in range(arg.epochs):
        emb_encoder.train()
        next_state_predictor.train()
        emb_decoder.train()
        sum_train_loss = 0
        ### Training the model ###
        c = None
        for train_batch in tqdm(training_dataloader):
            input_tensor = train_batch["input"].to(device)
            target_tensor = train_batch["target"].to(device)
            input_emb = emb_encoder(input_tensor)
            pred_emb = next_state_predictor(input_emb)
            pred_tensor = emb_decoder(pred_emb)
            loss = criterion(pred_tensor, target_tensor)
            
            loss.backward()
            
            optim.step()
            sum_train_loss += loss.item()
        ### Validation data ###
        emb_encoder.eval()
        next_state_predictor.eval()
        emb_decoder.eval()
        sum_val_loss = 0
        c = None
        with torch.no_grad():
            for val_batch in tqdm(validation_dataloader):
                input_tensor = val_batch["input"].to(device)
                target_tensor = val_batch["target"].to(device)
                input_emb = emb_encoder(input_tensor)
                pred_emb = next_state_predictor(input_emb)
                pred_tensor = emb_decoder(pred_emb)
                loss = criterion_eval(pred_tensor, target_tensor)
                sum_val_loss += loss.item()
        print(f"Epoch {epoch} train loss : {sum_train_loss} val loss : {sum_val_loss}\ntraining info |{ind_tracker.display_loss()}\nval info | {ind_tracker_eval.display_loss()}")
        ind_tracker.res_loss()
        ind_tracker_eval.res_loss()
    print(f"Final eval loss : {sum_val_loss}")
    torch.save(emb_encoder.state_dict(), "frame_encoder.pth")
    torch.save(next_state_predictor.state_dict(), "next_state_predictor_frame.pth")
    torch.save(emb_decoder.state_dict(), "frame_decoder.pth")
        