import torch
torch.manual_seed(42)
import argparse
from dataset_img import dataset_img
import numpy as np
from torch.utils.data import DataLoader
from jepa import vision_encoder, embedding_predictor
from tqdm import tqdm
from individual_loss_tracker import individual_tracker
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
criterion = lambda a, b: prediction_error_weight * ind_tracker.inc("prediction_error", prediction_error(a, b)) + variance_weight * ind_tracker.inc("variance_loss", variance_loss(a)) + covariance_loss_weight * ind_tracker.inc("covariance_loss", covariance_loss(a))

criterion_eval = lambda a, b: prediction_error_weight * ind_tracker_eval.inc("prediction_error", prediction_error(a, b)) + variance_weight * ind_tracker_eval.inc("variance_loss", variance_loss(a)) + covariance_loss_weight * ind_tracker_eval.inc("covariance_loss", covariance_loss(a))

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
    trainer_emb_encoder = vision_encoder(tuple_dim_img, arg.hidden_dim)
    trainer_emb_encoder = trainer_emb_encoder.to(device)
    with torch.no_grad():
        for param, param_trainer in zip(emb_encoder.parameters(), trainer_emb_encoder.parameters()):
            param_trainer.copy_(param)
            param_trainer.requires_grads = False

    # Setting up the optmizers
    learining_rate = 1e-6
    optim = torch.optim.AdamW(list(emb_encoder.parameters()) + list(next_state_predictor.parameters()), lr=learining_rate)

    temperature_softmax_sharp = 1e-7
    # the rate update paremerter(from arxiv:2104.14294v2)
    m = 0.45
    for epoch in range(arg.epochs):
        emb_encoder.train()
        next_state_predictor.train()
        trainer_emb_encoder.eval()
        sum_train_loss = 0
        ### Training the model ###
        c = None
        for train_batch in tqdm(training_dataloader):
            input_tensor = train_batch["input"].to(device)
            target_tensor = train_batch["target"].to(device)
            input_emb = emb_encoder(input_tensor)
            pred_emb = next_state_predictor(input_emb)
            target_emb = trainer_emb_encoder(target_tensor)
            if c is None:
                c = torch.mean(target_emb.view([-1] + [target_emb.size()[-1]]), dim=0).detach()
            else:
                c = m * c + (1 - m) * torch.mean(target_emb.view([-1] + [target_emb.size()[-1]]), dim=-2).detach()
            target_emb = target_emb - c
            target_emb = torch.nn.functional.softmax(target_emb/temperature_softmax_sharp, dim=-1)
            loss = criterion(pred_emb, target_emb)
            
            loss.backward()
            
            optim.step()
            sum_train_loss += loss.item()
        # Updating the trainer encoder at the end of each epochs
        coe_update = 0.996 + 0.5 * (1 - 0.996) * (1 + np.cos(epoch * np.pi))
        ### Validation data ###
        emb_encoder.eval()
        next_state_predictor.eval()
        sum_val_loss = 0
        c = None
        with torch.no_grad():
            for param, param_trainer in zip(emb_encoder.parameters(), trainer_emb_encoder.parameters()):
                param_trainer.copy_((coe_update) * param_trainer + (1 - coe_update) * param)
            for val_batch in tqdm(validation_dataloader):
                input_tensor = val_batch["input"].to(device)
                target_tensor = val_batch["target"].to(device)
                input_emb = emb_encoder(input_tensor)
                pred_emb = next_state_predictor(input_emb)
                target_emb = trainer_emb_encoder(target_tensor)
                if c is None:
                    c = torch.mean(target_emb.view([-1] + [target_emb.size()[-1]]), dim=-2)
                else:
                    c = m * c + (1 - m) * torch.mean(target_emb.view([-1] + [target_emb.size()[-1]]), dim=-2)
                target_emb = target_emb - c
                target_emb = torch.nn.functional.softmax(target_emb/temperature_softmax_sharp, dim=-1)
                loss = criterion_eval(pred_emb, target_emb)
                sum_val_loss += loss.item()
        print(f"Epoch {epoch} train loss : {sum_train_loss} val loss : {sum_val_loss}\ntraining info |{ind_tracker.display_loss()}\nval info | {ind_tracker_eval.display_loss()}")
        ind_tracker.res_loss()
        tem_save_detail_eval = ind_tracker_eval.display_loss()
        ind_tracker_eval.res_loss()
    print(f"Final eval dataset: {sum_val_loss}")
    print(tem_save_detail_eval)
    torch.save(emb_encoder.state_dict(), "emb_encoder.pth")
    torch.save(next_state_predictor.state_dict(), "next_state_predictor.pth")
        