import requests
import json
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datasets import load_dataset
import os
import gc

######
# GlobalVars
######
file_path = os.path.abspath(os.path.dirname(__file__))

class DataProcessor(ABC):
    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

    @abstractmethod
    def get_train_examples(self):
        pass

    @abstractmethod
    def get_test_examples(self):
        pass

    @abstractmethod
    def evaluate(self, predictor, test_exs):
        pass

    @abstractmethod
    def stringify_prediction(self, pred):
        pass



class ClassificationTask(DataProcessor):

    def run_evaluate(self, predictor, prompt, test_exs, n=100):
        labels = []
        preds = []
        texts = []
        for test in test_exs[:n]:
            pred = predictor.inference(test, prompt)
            texts.append(test['text'])
            labels.append(test['label'])
            preds.append(pred)

        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='micro')
        cfmat = confusion_matrix(labels, preds)
        return f1, accuracy, cfmat, texts, labels, preds

    def evaluate(self, predictor, prompt, test_exs, n=100):
        while True:
            try:
                f1, accuracy, cfmat, texts, labels, preds = self.run_evaluate(predictor, prompt, test_exs, n=n)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError):
                pass
        return f1, accuracy, cfmat, texts, labels, preds


class ClimateBinaryTask(ClassificationTask):
    categories = ['Supports', 'Refutes']
    raw_dataset = load_dataset("climate_fever", split='test', cache_dir=file_path+"/../../cache/datasets")

    def get_train_examples(self):
        exs = []
        true_statements = self.raw_dataset.filter(lambda x: x["claim_label"] == 0)
        false_statements = self.raw_dataset.filter(lambda x: x["claim_label"] == 1)
        for i, row in enumerate(true_statements.select(range(100,200))):
            exs.append({'id': f'train-{i}', 'label': row['claim_label'], 'text': row['claim']})
        for i, row in enumerate(false_statements.select(range(100,200))):
            exs.append({'id': f'train-{i}', 'label': row['claim_label'], 'text': row['claim']})
        return exs
    
    def get_test_examples(self):
        exs = []
        true_statements = self.raw_dataset.filter(lambda x: x["claim_label"] == 0)
        false_statements = self.raw_dataset.filter(lambda x: x["claim_label"] == 1)
        for i, row in enumerate(true_statements.select(range(0,100))):
            exs.append({'id': f'test-{i}', 'label': row['claim_label'], 'text': row['claim']})
        for i, row in enumerate(false_statements.select(range(0,100))):
            exs.append({'id': f'test-{i}', 'label': row['claim_label'], 'text': row['claim']})
        return exs
    
    def stringify_prediction(self, pred):
        return ClimateBinaryTask.categories[pred]


class PolitifactBinaryTask(ClassificationTask):
    categories = ['Supports', 'Refutes']

    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

        # Initialize an empty list to store the JSON objects
        self.data = []

        # Open the file and read it line by line
        with open(file_path+"/datasets/politifact_factcheck_data.json", 'r') as file:
            for line in file:
                # Load each line as a JSON object
                json_object = json.loads(line.strip())

                if json_object['verdict'] in {'true', 'false'}:
                    # Append the JSON object to the list
                    self.data.append(json_object)

    # exclude first 100 true and first 100 false statements
    def get_train_examples(self):
        exs = []
        true_statements = [d for d in self.data if d['verdict'] == "true"]
        false_statements = [d for d in self.data if d['verdict'] == "false"]

        for i, row in enumerate(true_statements[100:200]):
            exs.append({'id': f'train-{i}', 'label': 0, 'text': row['statement']})
        for i, row in enumerate(false_statements[100:200]):
            exs.append({'id': f'train-{i}', 'label': 1, 'text': row['statement']})

        return exs
    
    # first 100 true and first 100 false statements as test
    def get_test_examples(self):
        exs = []
        true_statements = [d for d in self.data if d['verdict'] == "true"]
        false_statements = [d for d in self.data if d['verdict'] == "false"]

        for i, row in enumerate(true_statements[:100]):
            exs.append({'id': f'test-{i}', 'label': 0, 'text': row['statement']})
        for i, row in enumerate(false_statements[:100]):
            exs.append({'id': f'test-{i}', 'label': 1, 'text': row['statement']})
        return exs
    
    def stringify_prediction(self, pred):
        return PolitifactBinaryTask.categories[pred]