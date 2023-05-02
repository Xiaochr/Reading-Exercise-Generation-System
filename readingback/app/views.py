from django.shortcuts import render
from django.http.response import *

import openai

# proxies = {
#     "http": "127.0.0.1:2022", 
#     "https": "127.0.0.1:2022"
# }
# openai.proxy = proxies

openai.api_key = ""


def gen_essay(request):
    data_dict = request.POST
    topics = data_dict.get("topics")
    cefr = data_dict.get("cefr")
    essay_len = data_dict.get("essay_len")
    genres = data_dict.get("genres")
    is_example = data_dict.get("is_example")
    example = data_dict.get("example")

    if is_example == "1":
        prompt = """
            Please generate a writing (without a title) that is similar to the given example and satisfies the following requirements:
            Topics: {}
            Length: no more than {} words
            Genre: {}
            CEFR level: {}
            Example: {}
        """.format(topics, str(essay_len), genres, cefr, example)
    else:
        prompt = """
            Please generate a writing (without a title) satisfying the following requirements:
            Topics: {}
            Length: no more than {} words
            Genre: {}
            CEFR level: {}
        """.format(topics, str(essay_len), genres, cefr)
    
    print(prompt)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant to generate reading comprehension materials for Chinese middle school English learners. Your responses should not include any toxic content. "},
                {"role": "user", "content": prompt}
            ]
    )

    response_content = response["choices"][0]["message"]["content"]
    print(response_content)
    return HttpResponse(response_content)


def gen_questions(request):
    data_dict = request.POST
    essay = data_dict.get("essay")
    q_num = data_dict.get("q_num")
    a_num = data_dict.get("a_num")
    q_type = data_dict.get("q_type")

    if q_type == "random":
        prompt = """
            Please generate {} multiple choice questions (each question with {} choices), the corresponding answers and explanations for the following reading comprehension exercise.
            Exercise: {}
        """.format(q_num, a_num, essay)
    else:
        prompt = """
            Please generate {} multiple choice questions (each question with {} choices), the corresponding answers and explanations for the following reading comprehension exercise. The type of questions should be {}. 
            Exercise: {}
        """.format(q_num, a_num, q_type, essay)
    
    print(prompt)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant to generate reading comprehension exercise questions for Chinese middle school English learners. Your responses should not include any toxic content. "},
                {"role": "user", "content": prompt}
            ]
    )

    response_content = response["choices"][0]["message"]["content"]
    print(response_content)
    return HttpResponse(response_content)

