import random

from datasets import load_dataset


def init_aime2025(train_full_size=None):
    train_split = [
        {"input": x["problem"], "additional_context": {"solution": x["solution"]}, "answer": "### " + str(x["answer"])}
        for x in load_dataset("AI-MO/aimo-validation-aime")["train"]
    ]
    if train_full_size is not None:
        train_split = train_split[:train_full_size]
    random.Random(0).shuffle(train_split)
    test_split = [
        {"input": x["problem"], "answer": "### " + str(x["answer"])}
        for x in load_dataset("MathArena/aime_2025")["train"]
    ]

    trainset = train_split[: len(train_split) // 2]
    valset = train_split[len(train_split) // 2 :]
    testset = test_split * 1

    return trainset, valset, testset


def init_math500(train_full_size=None):
    dataset_dir = 'HuggingFaceH4/MATH-500'
    # Load original training data
    full_train = [
        {"input": x["problem"], 
         "additional_context": {"solution": x["solution"]}, 
         "answer": "### " + str(x["answer"])}
        for x in load_dataset(dataset_dir)["test"]
    ]
    if train_full_size is not None:
        full_train = full_train[:train_full_size]
    
    # Use only the first 5 samples for training
    trainset = full_train[: len(full_train) // 2]
    valset = full_train[len(full_train) // 2 :]

    # Test split uses the entire 2025 dataset
    testset = [
        {"input": x["problem"], "answer": "### " + str(x["answer"])}
        for x in load_dataset(dataset_dir)["test"]
    ]

    return trainset, valset, testset

def init_amc23(train_full_size=None):
    dataset_dir = 'math-ai/amc23'
    # Load original training data
    full_train = [
        {"input": x["question"], 
         "additional_context": {"solution": ""}, 
         "answer": "### " + str(x["answer"])}
        for x in load_dataset(dataset_dir)["test"]
    ]
    if train_full_size is not None:
        full_train = full_train[:train_full_size]
    
    # Use only the first 5 samples for training
    trainset = full_train[: len(full_train) // 2]
    valset = full_train[len(full_train) // 2 :]

    # Test split uses the entire 2025 dataset
    testset = [
        {"input": x["question"], "answer": "### " + str(x["answer"])}
        for x in load_dataset(dataset_dir)["test"]
    ]

    return trainset, valset, testset

def init_aime2024(train_full_size=None):
    dataset_dir = 'Maxwell-Jia/AIME_2024'
    input_col_name = 'Problem'
    answer_col_name = 'Answer'
    solution_col_name = 'Solution'
    # Load original training data
    full_train = [
        {"input": x[input_col_name], 
         "additional_context": {"solution": x["Solution"]}, 
         "answer": "### " + str(x[answer_col_name])}
        for x in load_dataset(dataset_dir)["train"]
    ]
    if train_full_size is not None:
        full_train = full_train[:train_full_size]
    
    # Use only the first 5 samples for training
    trainset = full_train[: len(full_train) // 2]
    valset = full_train[len(full_train) // 2 :]

    # Test split uses the entire 2025 dataset
    testset = [
        {"input": x[input_col_name], "answer": "### " + str(x[answer_col_name])}
        for x in load_dataset(dataset_dir)["train"]
    ]

    return trainset, valset, testset