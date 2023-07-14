from src.orchestator import OrchestatorProcess
import os
import pandas as pd
import argparse

def run(args):
    process = OrchestatorProcess(args)
    process.run_process()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exec', type=str, help='Type of execution', default="train")
    args = parser.parse_args()

    parser.add_argument('--file_name', type=str, help='Name dataset file', default="diabetes_prediction_dataset.csv")
    args = parser.parse_args()

    parser.add_argument('--y_name', type=str, help='Column name objective varible', default="diabetes")
    args = parser.parse_args()

    parser.add_argument('--metric', type=str, help='Metric to select the model', default="accuracy")
    args = parser.parse_args()

    

    run(args)