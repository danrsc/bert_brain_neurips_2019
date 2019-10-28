from collections import OrderedDict

from pytorch_pretrained_bert import BertTokenizer

from .spacy_token_meta import make_tokenizer_model
from .corpus_base import CorpusBase


__all__ = ['CorpusLoader']


class CorpusLoader(object):

    def __init__(self, cache_path, bert_pre_trained_model_name='bert-base-uncased'):
        self.bert_pre_trained_model_name = bert_pre_trained_model_name
        self.cache_path = cache_path

    def make_bert_tokenizer(self):
        return BertTokenizer.from_pretrained(self.bert_pre_trained_model_name, self.cache_path, do_lower_case=True)

    def load(
            self,
            index_run,
            corpora,
            data_preparer=None,
            force_cache_miss=False,
            paths_obj=None):

        bert_tokenizer = self.make_bert_tokenizer()
        spacy_tokenizer_model = make_tokenizer_model()

        if isinstance(corpora, CorpusBase):
            corpora = [corpora]

        result = OrderedDict()

        for corpus in corpora:

            key = type(corpus).__name__

            if key in result:
                raise ValueError('Corpus can only be loaded once')

            print('Loading {}...'.format(key), end='', flush=True)
            result[key] = corpus.load(index_run, spacy_tokenizer_model, bert_tokenizer, paths_obj, force_cache_miss)
            print('done')

        if data_preparer is not None:
            result = data_preparer.prepare(result)

        return result
