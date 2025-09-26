import os
import pandas as pd
from abc import ABC,abstractmethod

class dataIngestor(ABC):
    @abstractmethod
    def ingest(self,file_path_or_link:str) -> pd.DataFrame:
        pass

class dataIngestorCSV(dataIngestor):
    def ingest(self, file_path_or_link):
        return pd.read_csv(file_path_or_link)

class dataIngestorExcel(dataIngestor):
    def ingest(self, file_path_or_link):
        return pd.read_excel(file_path_or_link)