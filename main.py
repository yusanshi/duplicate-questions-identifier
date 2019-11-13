from train import train
from apply import apply
from pathlib import Path
from config import MODEL_PATH, LOG_PATH
import os
import shutil


def main(question_pairs, force_retrain=False):
    if force_retrain:
        if Path(MODEL_PATH).is_dir():
            shutil.rmtree(MODEL_PATH)

    if Path(LOG_PATH).is_dir():
        shutil.rmtree(LOG_PATH)

    if not Path(MODEL_PATH).is_dir():  # not exists
        os.mkdir(MODEL_PATH)

    if not os.listdir(MODEL_PATH):  # blank
        train()

    result = apply(question_pairs)

    for i, question in enumerate(question_pairs):
        print('Question Pair %d:' % i)
        print('Q1: %s' % question[0])
        print('Q2: %s' % question[1])
        print('Probability of beging duplicate: %.4f' % result[i])
        print()


if __name__ == '__main__':
    question_pairs = [
        ['How are you?', 'How do you do?'],
        ['Why light travels so fast?', 'Why light travels so fast?'],
        ["What's your name?", 'Could you tell me you name?'],
        ['What is this?', 'I love you.'],
        ['Is it possible to reduce 17 lbs in one month?',
            'What are the best ways to lose weight?'],
        ['What should I do to prepare for NDA?'	,
            'How can I prepare for NDA in 6 months?'],
        ['Are most blind people completely blind?',
            'Are most blind people fully blind?'],
        ['What are the best interview questions ever asked?',
            'What are the best interview questions to ask?'],
        ['What are the best interview questions ever asked?',
            'What are the worst interview questions to ask?']
    ]
    main(question_pairs)
