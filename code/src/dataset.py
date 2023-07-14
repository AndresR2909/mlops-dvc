import pandas as pd
import os


def resume_data(path):
    data = pd.read_csv(path)

    print(f"La cantidad de datos en el dataset es: {len(data)}")


def create_dataframe(folder_path):
    dfs = [] 

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)

    unified_df = pd.concat(dfs, ignore_index=True)

    return unified_df