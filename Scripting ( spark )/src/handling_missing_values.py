import groq
import logging
import pandas as pd
from enum import Enum
from typing import Optional, List, Union
from dotenv import load_dotenv
from pydantic import BaseModel
from abc import ABC,abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.ml.feature import Imputer
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

class MissingValueHandlingStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def handle(self, df:pd.DataFrame)->pd.DataFrame:
        pass

class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self,critical_columns : List[str] = None, spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.critical_columns = critical_columns or []
        logging.info(f"Dropping rows with missing values in critical columns: {self.critical_columns}")

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.dropna(subset=self.critical_columns)
        n_dropped = df.count() - df_cleaned.count()
        logging.info(f"{n_dropped} has been dropped")
        return df_cleaned

class Gender(str,Enum):
    MALE = 'Male'
    FEMALE = 'Female'

class GenderPrediction(BaseModel):
    firstname:str
    lastname:str
    pred_gender:Gender


class GenderImputer:
    def __init__(self):
        self.groq_client = groq.Groq()
    
    def _predict_gender(self,firstname,lastname):
        prompt = f"""
            What is the most likely gender (Male or Female) for someone with the first name '{firstname}'
            and last name '{lastname}' ?

            Your response only consists of one word: Male or Female
            """
        response = self.groq_client.chat.completions.create(
                                                            model='llama-3.3-70b-versatile',
                                                            messages=[{"role": "user", "content": prompt}],
                                                            )
        predicted_gender = response.choices[0].message.content.strip()
        prediction = GenderPrediction(firstname=firstname, lastname=lastname, pred_gender=predicted_gender)
        logging.info(f'Predicted gender for {firstname} {lastname}: {prediction}')
        return prediction.pred_gender

    def impute(self,df):

        ############### PANDAS CODES ###########################
        # missing_gender_index = df['Gender'].isnull()
        # for idx in df[missing_gender_index].index:
        #     first_name = df.loc[idx, 'Firstname']
        #     last_name = df.loc[idx, 'Lastname']
        #     gender = self._predict_gender(first_name, last_name)
            
        #     if gender:
        #         df.loc[idx, 'Gender'] = gender
        #         print(f"{first_name} {last_name} : {gender}")
        #     else:
        #         print(f"{first_name} {last_name} : No Gender Detected")

        # return df

        ############### PYSPARK CODES ###########################
        # Create a UDF (User Defined Functions) for gender prediction

        predict_gender_udf = F.udf(self._predict_gender, StringType())
        missing_gender_df = df.filter(
            F.col('Gender').isNull() | (F.col('Gender')=='')
        ).select('FirstName','LastName')

        missing_count = missing_gender_df.count()

        predictions_df = missing_gender_df.withColumn(
            'PredictedGender',
             predict_gender_udf(F.col('Firstname'), F.col('Lastname'))
        )

        df_with_predictions = df.join(
            predictions_df,
            on=['Firstname', 'Lastname'],
            how='left'
        )
        
        # Fill missing gender with predictions
        df_imputed = df_with_predictions.withColumn(
            'Gender',
            F.when(
                F.col('Gender').isNull() | (F.col('Gender') == ''),
                F.col('PredictedGender')
            ).otherwise(F.col('Gender'))
        ).drop('PredictedGender')

        return df_imputed

class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self,
                method='mean',
                fill_value = None,
                relavant_column=None,
                is_custom_computer = False,
                custom_imputer = None,
                spark: Optional[SparkSession] = None
            ):
        self.method = method
        self.fill_value = fill_value
        self.relavant_column = relavant_column
        self.is_custom_computer = is_custom_computer
        self.custom_imputer = custom_imputer
    
    def handle(self,df):
        if self.is_custom_computer:

            return self.custom_imputer.impute(df)

        if self.relavant_column:
            if self.method == 'mean':
                ############### PANDAS CODES ###########################
                # mean_value = df[self.relevant_column].mean()
                # df_filled = df.fillna({self.relevant_column: mean_value})
                
                ############### PYSPARK CODES ###########################
                mean_value = df.select(F.mean(F.col(self.relavant_column))).collect[0][0]
                df_filled = df.fillna({self.relevant_column: mean_value})
            
            elif self.method == 'median':
                 ############### PANDAS CODES ###########################
                # median_value = df[self.relevant_column].median()
                # df_filled = df.fillna({self.relevant_column: median_value})

                ############### PYSPARK CODES ###########################
                median_value = df.approxQuantile(self.relavant_column,[0.5], 0.01)[0]
                df_filled = df.fillna({self.relevant_column: median_value})
            
            elif self.method == 'mode':
                ############### PANDAS CODES ###########################
                # mode_value = df[self.relevant_column].mode()[0]
                # df_filled = df.fillna({self.relevant_column: mode_value})
                
                ############### PYSPARK CODES ###########################
                mode_value = df.groupBy(self.relevant_column).count().orderBy(F.desc('count')).first()[0]
                df_filled = df.fillna({self.relevant_column: mode_value})
                
            elif self.method == 'constant' and self.fill_value is not None:
                df_filled = df.fillna({self.relevant_column: self.fill_value})
                
            else:
                raise ValueError(f"Invalid method '{self.method}' or missing fill_value")


        return df_filled
