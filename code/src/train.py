import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.competition import MetricsCompetition


class BinariClassificator:
    def __init__(self, metric) -> None:
        self.models = {
            "random_forest":{ 
                "model": RandomForestClassifier(),
                "hiperparameters": { 'n_estimators': [200, 500],
                                     'max_features': ['auto', 'sqrt', 'log2'],
                                     'max_depth' : [4,5,6,7,8],
                                     'criterion' :['gini', 'entropy']}},
        
            "desicion_tree":{ 
                "model": DecisionTreeClassifier(),
                "hiperparameters": { 'max_features': ['auto', 'sqrt', 'log2'],
                                    'ccp_alpha': [0.1, .01, .001],
                                    'max_depth' : [5, 6, 7, 8, 9],
                                    'criterion' :['gini', 'entropy']}}
        }
        self.metric = metric
        self.results = {}
        self.best_model = None
        self.best_score = 0.0
        self.best_name = ""

    #Creamos función para entrenar los modelos
    def train_model(self, X_train, y_train):
        for name,object  in self.models.items():
            model = object["model"]
            params = object["hiperparameters"]
            gs =  GridSearchCV(model, params, cv = 5, verbose=3, scoring=self.metric)
            gs.fit(X_train, y_train)
            object["model"] = gs.best_estimator_
            print(f"Modelo {name} ha sido entrenado")

    #Creamos una función para guardar el modelo entrenado
    def save_model(self, path):
        model = self.models["random_forest"]
        joblib.dump(model, path)

        print("El modelo ha sido serializado con exito.")


    def evaluate_models(self, X_test, y_test):
        
        for name, config in self.models.items():
            model = config["model"]
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            self.results[name] = {'Accuracy': accuracy, 'Recall': recall, 'F1-Score': f1}

            print("-"*40)
            print(f"Métricas de modelo {name}:")
            print(f"--Accuracy: {accuracy}")
            print(f"--Recall: {recall}")
            print(f"--F1-Score: {f1}")
    
    def select_best_model(self):
        competition = MetricsCompetition(self.results,'count')
        winner = competition.evaluated_best_model()
        self.best_name = winner
        self.best_score = self.results["winner"]
        self.best_model = self.models[winner]["model"]




    
