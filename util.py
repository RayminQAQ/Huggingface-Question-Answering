import os, json
import pandas as pd
from datasets import Dataset, DatasetDict

class Data:
    def __init__(self, path_to_dir: str):
        self.path_to_dir = path_to_dir
        
        # File information: local file under self.path_to_dir
        self.files = [
            {"data_format": 'FAQ', "fileName": "101_faq.json"},
            {"data_format": "KE", "fileName": "2024-06-25_api_calls_output_KE.xlsx",},
            {"data_format": "KE", "fileName": "2024-06-27_api_calls_output_KE.xlsx",},
            {"data_format": "KE", "fileName": "2024-06-28_api_calls_output_KE.xlsx",},
            # Add your files here #
        ]
        
        # File format: column mapping
        self.column_mappings = {
            'FAQ': {"question": "Question", "answer": "Answer"},
            #'FAQ': {'Question': 'question', 'Answer': 'answer'},
            'KE': {'Query': 'Question', 'Response': 'Answer'},
            
            # Add your file format here #
            'custom1': {'input_text1': 'Question', 'target_text1': 'Answer'},
            'custom2': {'input_text2': 'Question', 'target_text2': 'Answer'},
        }

    def read_local_dataset(self, file_path: str) -> pd.DataFrame:
        file_type = file_path.split(".")[-1]
        if file_type == "xlsx":
            df = pd.read_excel(file_path, engine="openpyxl")
        elif file_type == "json":
            # Note: pd.read_json 有問題
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported file type.")
        return df

    def format_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        if dataset_type in self.column_mappings:
            # change column names
            mapping = self.column_mappings[dataset_type]
            df.rename(columns=mapping, inplace=True)
            
            # Error checking
            required_columns = ['Question', 'Answer']
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing columns in dataset '{dataset_type}': {missing_columns}")
            
            # only save required_columns
            df = df[required_columns]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Add a column to indicate the dataset type
        df['source'] = dataset_type
        
        # **新增功能：移除 'Answer' 列中包含列表的行**
        df = df[df['Answer'].apply(lambda x: not isinstance(x, list))]
        return df

    def merge_datasets(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        merged_df = pd.concat(dfs, ignore_index=True)
        return merged_df

    def load_data(self) -> Dataset:
        # flag
        dataset = []
        
        # Load data from local files
        for file_info in self.files:
            # file information
            dataset_type = file_info['data_format']
            file_name = file_info['fileName']
            file_path = os.path.join(self.path_to_dir, file_name)
            
            # error checking
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}, Dataset name: {file_name}")
            
            # read file
            df = self.read_local_dataset(file_path)
            df = self.format_data(df, dataset_type)
            dataset.append(df)
        
        merged_df = self.merge_datasets(dataset)
        hf_dataset = Dataset.from_pandas(merged_df)
        
        return hf_dataset
    
    def test(self):
        # flag
        dataset = []
        
        # Load data from local files
        for file_info in self.files:
            # file information
            dataset_type = file_info['data_format']
            file_name = file_info['fileName']
            file_path = os.path.join(self.path_to_dir, file_name)
            
            # error checking
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}, Dataset name: {file_name}")
            
            # read file
            df = self.read_local_dataset(file_path)
            df = self.format_data(df, dataset_type)
            dataset.append(df)
            print(type(df))
        
        merged_df = self.merge_datasets(dataset)
        
        # 找出 Answer 列中包含 list 类型数据的行
        has_list_in_answer = merged_df['Answer'].apply(lambda x: isinstance(x, list))


        # 打印出包含 list 数据的行
        list_rows = merged_df[has_list_in_answer]
        print("Rows with list data in 'Answer' column:")
        print(list_rows)
        
        hf_dataset = Dataset.from_pandas(merged_df)
        return hf_dataset