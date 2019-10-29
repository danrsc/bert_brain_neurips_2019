import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Paths:

    default_relative_pre_trained_path = 'uncased_L-12_H-768_A-12'
    default_relative_cache_path = 'bert_cache'
    default_relative_harry_potter_path = 'harry_potter'
    default_relative_glue_path = os.path.join('GLUE', 'glue_data')

    pre_trained_base_path: str = '/share/volume0/drschwar/BERT'
    result_path: str = '/share/volume0/drschwar/bert_erp/results/'
    data_set_base_path: str = '/share/volume0/drschwar/data_sets/'
    model_path: str = '/share/volume0/drschwar/bert_erp/models/'

    pre_trained_path: Optional[str] = None
    cache_path: Optional[str] = None
    harry_potter_path: Optional[str] = None
    glue_path: Optional[str] = None

    def __post_init__(self):
        if self.pre_trained_path is None:
            self.pre_trained_path = os.path.join(self.pre_trained_base_path, Paths.default_relative_pre_trained_path)
        if self.cache_path is None:
            self.cache_path = os.path.join(self.data_set_base_path, Paths.default_relative_cache_path)
        if self.harry_potter_path is None:
            self.harry_potter_path = os.path.join(self.data_set_base_path, Paths.default_relative_harry_potter_path)
        if self.glue_path is None:
            self.glue_path = os.path.join(self.data_set_base_path, Paths.default_relative_glue_path)
