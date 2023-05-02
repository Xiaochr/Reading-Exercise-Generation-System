import pandas as pd
import torch
import random

import nltk
nltk.download('punkt')
from gensim.models import Word2Vec

from run_pplm_essay import run_pplm_example



def train_w2v(
    input_data="./data/data_step2.csv", 
    output_model="./topics/w2v.model", 
    vector_size=512, 
    window=5, 
    min_count=5
):
    essay_df = pd.read_csv(input_data, index_col=0)
    essay_content = essay_df["reading_text"].values.tolist()

    tokens = []
    for essay in essay_content:
        sents = nltk.sent_tokenize(essay)
        
        temp_words = []
        for sent in sents:
            words = nltk.word_tokenize(sent)
            words = [word.lower() for word in words if word.isalpha()]
            temp_words.append(words)
        
        tokens.extend(temp_words)

    model = Word2Vec(
        sentences=tokens, 
        vector_size=vector_size, 
        window=window, 
        min_count=min_count, 
        workers=4
    )
    model.save(output_model)
    # generate keyword lists using "model.wv.most_similar(keywords, topn=topn)"



# tune PPLM
def tune_params(
    finetuned_model="./model/finetuned_gpt2",
    keywords_type=0, 
    topn=30, 
    step_size=0.03, 
    gm_scale=0.9, 
    kl_scale=0.01, 
    grad_length=100,
    n_samples=5,
    max_essay_len=350
):
    if keywords_type == 0:
        keywords_list = "./topics/keywords_school_{}.txt".format(topn)
    elif keywords_type == 1:
        keywords_list = "./topics/keywords_life_{}.txt".format(topn)
    elif keywords_type == 2:
        keywords_list = "./topics/keywords_nature_{}.txt".format(topn)

    params_info = "keywords_type: {}, topn: {}, step_size: {}, \
        gm_scale: {}, kl_scale: {}, grad_length: {}".format(
            keywords_type, 
            topn, 
            step_size, 
            gm_scale, 
            kl_scale, 
            grad_length
        )
    print(params_info)

    generated_text = run_pplm_example(
        pretrained_model=finetuned_model,
        #cond_text="The potato",
        uncond=True,
        num_samples=n_samples,
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

    torch.cuda.empty_cache()



def main():
    """
        To lossen the control: 
            1. decrease step_size
            2. increase kl_scale (default 0.01), decrease gm_scale (default 0.9)
            3. increase grad_length
        
        topn: [10, 20, 30]
        step_size: [0.02, 0.025, 0.03, 0.035, 0.04]
        gm_scale: [0.7, 0.75, 0.8, 0.85, 0.9]
        kl_scale: [0.01, 0.02, 0.03, 0.04, 0.05]
        grad_length: [100, 1000, 10000]
    """

    topn = 30
    step_size = 0.03
    gm_scale = 0.9
    kl_scale = 0.01
    grad_length = 100
    keywords_type = 0 # 0: shool; 1: life; 2: nature

    tune_params(
        finetuned_model="./model/finetuned_gpt2",
        keywords_type=keywords_type,
        topn=topn,
        step_size=step_size,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        grad_length=grad_length
    )



if __name__ == "__main__":
    main()


