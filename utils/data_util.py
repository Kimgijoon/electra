import re
import collections
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from transformers import ElectraTokenizer


class DataHelper(object):
    
    def __init__(self, model_size: str):
        """Utilization class for data preprocessing

        Args:
            model_size (str): A parameter that pass whether it is base or small
        """
        if model_size == 'base':
            self.tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
        else:
            self.tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-small-v3-discriminator')
    
    def read_parquet(self, file_path: str) -> pd.DataFrame:
        
        df = pd.read_parquet(f'{file_path}/data.parquet')
        return df
    
    def tokenize(self, sentence_a: str, sentence_b: Optional[str]=None, seq_len: int=512) -> Dict[str, List[int]]:
        """Tokenize sentence into several info

        Args:
            sentence_a (str): The sequence to be encoded
            sentence_b (Optional[str]): A second sequence to be encoded with the first
            seq_len (int): Pad to a length specified by the seq_len argument 
            
        Returns:
            Dict[str, List[int]]: Tokenized sentence
        """
        result = self.tokenizer(text=sentence_a,
                                text_pair=sentence_b,
                                max_length=seq_len,
                                padding='max_length',
                                truncation=True)
        return result

    def encode(self, sentence: str) -> List[int]:
        """Encode given sentence(or word) to token id(s)

        Args:
            sentence (str): Input sentence

        Returns:
            List[int]: Encoded sentence
        """
        ids = self.tokenizer.encode(sentence)
        return ids
    
    def decode(self, ids: str) -> str:
        """Decode token id(s) to sentence(word)

        Args:
            sentence (str): _description_

        Returns:
            str: Decoded ids
        """        
        sentence = self.tokenizer.decode(ids)
        return sentence
    

def main():

    dh = DataHelper('base')
    print(dh.tokenize('안녕하세요'))
    
    
if __name__ == '__main__':
    
    main()

