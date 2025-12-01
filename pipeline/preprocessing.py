import pandas as pd


class Preprocessor:
    def __init__(self, dataset=list[list]):
        self.dataset = dataset

    def combine_datasets(self):
        combined_data = pd.concat(self.dataset)
        combined_data = combined_data.sort_values(["Ticker", "Date"])
        
        combined_data.to_csv("data/processed/combined_data.csv", index=False)
        
        return combined_data
