import torch
import random
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2LMHeadModel

from run_pplm_essay import run_pplm_example
# if you'd like to use "gpt-2 without finetuning + PPLM", please import:
# from run_pplm import run_pplm_example



def get_gpt2_text(
    model_path="./model/finetuned_gpt2", 
    prompt="<|startoftext|>", 
    essay_num=30, 
    max_essay_len=350
):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    sample_outputs = model.generate(
                    generated,
                    do_sample=True,   
                    top_k=50, 
                    max_length=max_essay_len,
                    top_p=0.95, 
                    num_return_sequences=essay_num
                    )

    essay_list = []
    for i, sample_output in enumerate(sample_outputs):
        decoded_text = tokenizer.decode(sample_output, skip_special_tokens=True)
        essay_list.append(decoded_text)
        print("{}: {}\n\n".format(i, decoded_text))
    
    essay_df = pd.DataFrame(np.array(essay_list).T, columns=["essay_content"])
    return essay_df



def get_pplm_gpt2_text(
    model_path="./model/finetuned_gpt2",
    keywords_list = "./topics/keywords_school_30.txt",
    essay_num=30,
    max_essay_len=350,
    step_size=0.035,
    gm_scale=0.9,
    kl_scale=0.01,
    grad_length=100
):
    generated_text = run_pplm_example(
        pretrained_model=model_path,
        #cond_text="The potato",
        uncond=True,
        num_samples=essay_num,
        bag_of_words=keywords_list,
        length=max_essay_len,
        stepsize=step_size,
        sample=True,
        num_iterations=3,
        window_length=5,
        gamma=1.5,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        verbosity='quiet',
        grad_length=grad_length,
        colorama=False,
        seed=random.randint(1, 10000)
    )
    
    essay_df = pd.DataFrame(np.array(generated_text).T, columns=["essay_content"])
    return essay_df



if __name__ == "__main__":
    get_gpt2_text()
    get_pplm_gpt2_text()



