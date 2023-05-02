import os
import time
import datetime

import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup



def load_data(file_path="./data/"):
    essay_step1 = pd.read_csv(file_path + "data_step1.csv", index_col=0)
    essay_step2 = pd.read_csv(file_path + "data_step2.csv", index_col=0)

    content_step1 = essay_step1["reading_text"].values.tolist()
    content_step2 = essay_step2["reading_text"].values.tolist()

    return content_step1, content_step2


class GPT2Dataset(Dataset):
    # dataset for GPT-2 input
    def __init__(self, txt_list, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def get_dataloaders(content, model_type, tokenizer, batch_size):
    if model_type == "gpt2":
        dataset = GPT2Dataset(content, tokenizer, max_length=768)
    elif model_type == "gpt2-medium":
        dataset = GPT2Dataset(content, tokenizer, max_length=1024)
    else:
        print("unkown model_type")

    # split into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    train_dataloader = DataLoader(
                train_dataset,
                sampler=RandomSampler(train_dataset),
                batch_size=batch_size
            )

    validation_dataloader = DataLoader(
                val_dataset,
                sampler=SequentialSampler(val_dataset),
                batch_size=batch_size
            )
    
    return train_dataloader, validation_dataloader


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))



def finetune_gpt2(
    file_path="./data/", 
    batch_size=2, 
    epochs=[3, 5],
    learning_rate=[5e-4, 1e-5],
    epsilon=[1e-8, 1e-8],
    warmup_steps=[1e2, 1e2],
    model_type="gpt2-medium",
    seed_val=42
):
    # load data
    content_step1, content_step2 = load_data(file_path)

    # set up for step-1 fine-tune
    tokenizer = GPT2Tokenizer.from_pretrained(model_type, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    configuration = GPT2Config.from_pretrained(model_type, output_hidden_states=False)
    
    model = GPT2LMHeadModel.from_pretrained(model_type, config=configuration)
    model.resize_token_embeddings(len(tokenizer))

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    train_dataloader, validation_dataloader = get_dataloaders(
                content=content_step1,
                model_type=model_type,
                tokenizer=tokenizer,
                batch_size=batch_size
            )
    
    total_steps = len(train_dataloader) * epochs[0]

    optimizer = AdamW(model.parameters(), lr=learning_rate[0], eps=epsilon[0])
    scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_steps[0], 
                num_training_steps=total_steps
            )
    
    device = torch.device("cuda")

    # step-1 fine-tune
    train(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs[0],
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader
    )


    # set up for step-2 fine-tune
    train_dataloader, validation_dataloader = get_dataloaders(
                content=content_step2,
                model_type=model_type,
                tokenizer=tokenizer,
                batch_size=batch_size
            )
    
    total_steps = len(train_dataloader) * epochs[1]

    optimizer = AdamW(model.parameters(), lr=learning_rate[1], eps=epsilon[1])
    scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps = warmup_steps[1], 
                num_training_steps = total_steps
            )

    # step-2 fine-tune
    train(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs[1],
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader
    )



def train(model, tokenizer, optimizer, scheduler, device, epochs, train_dataloader, validation_dataloader):
    total_t0 = time.time()
    model = model.to(device)

    for epoch_i in range(0, epochs):
        # =================== Training ============================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        model.train()
        train_loss = []

        for batch_index, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            model.zero_grad()

            outputs = model(
                    b_input_ids,
                    labels=b_labels, 
                    attention_mask=b_masks,
                    token_type_ids=None
                )
            
            loss = outputs[0]  
            
            batch_loss = loss.item()
            total_train_loss += batch_loss
            train_loss.append(batch_loss)
            ts = time.time()
            
            if batch_index % 10 == 0:
                print('epoch:',epoch_i)
                print("batch {}/{} loss: {:4f} time: {} s".format(
                            batch_index, len(train_dataloader), batch_loss, ts - t0))
            
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("")
        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Training epoch took: {:}".format(training_time))
        
        output_dir = "./model/" + str(epoch_i)
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Saving model to %s" % output_dir)
        
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # ========== Validation ===============
        print("")
        print("Running Validation...")
        t0 = time.time()
        model.eval()
        total_eval_loss = 0
        valid_loss = []
        
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            
            with torch.no_grad():        
                outputs = model(
                        b_input_ids,
                        attention_mask=b_masks,
                        labels=b_labels
                    )
            
                loss = outputs[0]

            batch_loss = loss.item()
            total_eval_loss += batch_loss  
            valid_loss.append(batch_loss)

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)    
        print("Validation Loss: {0:.2f}".format(avg_val_loss))
        print("Validation took: {:}".format(validation_time))

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))



if __name__ == "__main__":
    finetune_gpt2()


