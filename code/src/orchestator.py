import os
import pandas as pd
from src.dataset import (resume_data, create_dataframe)
from src.train import BinariClassificator
from src.data_preprocess import DataCleaning

class OrchestatorProcess:
    def __init__(self,args) -> None:
        self.excecution = args.exec
        self.root = os.getcwd()
        self.data_path = os.path.join(self.root,"code", "data", args.file_name)    
        self.model_path = os.path.join(self.root,"code", "models", "model.sav")
        self.y_name = args.y_name
        self.metric = args.metric

    def run_process(self):
        if self.excecution == "join":
            data_folder = os.path.join(self.root,"code", "data", "dataset")
            df = create_dataframe(data_folder)
            df.to_csv(self.data_path, index=False)

            print(f"Se completa la unificación de la data, con un total de {len(df)} datos.")
        
        elif self.excecution == "resume":
            resume_data(self.data_path)

        elif self.excecution == "train":
            data_clean = DataCleaning(self.data_path)
            data_clean.handle_missing_values()
            data_clean.handle_categorical_features()
            data_clean.split_data(self.y_name, 0.2)

            print(f"Cantidad de datos en set de entrenamiento: {len(data_clean.X_train)}")
            print(f"Cantidad de datos en set de pruebas: {len(data_clean.X_test)}")

            model_process = BinariClassificator(self.metric)
            model_process.train_model(data_clean.X_train, data_clean.y_train)
            model_process.save_model(self.model_path)