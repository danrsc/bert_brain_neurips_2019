import warnings

import numpy as np
import torch
from torch.nn import Parameter


__all__ = ['AwdLSTM', 'AwdLSTMLayerOutput', 'AwdLSTMOutput',
           'dropout_mask', 'EmbeddingDropout', 'LockedDropout', 'locked_dropout', 'WeightDrop']


def dropout_mask(x, sz, dropout):
    return x.new(*sz).bernoulli_(1 - dropout) / (1 - dropout)


def locked_dropout(is_training, x, p):
    if not is_training or p == 0:
        return x
    mask = dropout_mask(x.data, (1, x.size(1), x.size(2)), p)
    return mask * x


class LockedDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return locked_dropout(self.training, x, self.p)


class EmbeddingDropout(torch.nn.Module):
    def __init__(self, embedding, dropout=0.1, scale=None):
        super().__init__()
        self.embedding = embedding
        self.dropout = dropout
        self.scale = scale

    def forward(self, words):
        if self.training and self.dropout != 0:
            size = (self.embedding.weight.size(0), 1)
            mask = dropout_mask(self.embedding.weight.data, size, self.dropout)
            masked_embed_weight = self.embedding.weight * mask
        else:
            masked_embed_weight = self.embedding.weight

        if self.scale:
            masked_embed_weight = self.scale * masked_embed_weight

        padding_idx = self.embedding.padding_idx
        if padding_idx is None:
            padding_idx = -1

        # noinspection PyProtectedMember
        return torch.nn.functional.embedding(
            words, masked_embed_weight, padding_idx, self.embedding.max_norm,
            self.embedding.norm_type, self.embedding.scale_grad_by_freq, self.embedding.sparse)


# taken from https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/weight_drop.py


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights=('weight_hh_l0',), dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            # print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda:
                    mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, Parameter(w))

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class AwdLSTMLayerOutput(object):

    def __init__(self, dropped_out_input, h_init, c_init, h_final, c_final, h):
        self.dropped_out_input = dropped_out_input
        self.h_init = h_init
        self.c_init = c_init
        self.h_final = h_final
        self.c_final = c_final
        self.h = h


class AwdLSTMOutput(object):

    def __init__(self, dropped_out_embedding, layer_outputs, dropped_out_final):
        self.dropped_out_embedding = dropped_out_embedding
        self.layer_outputs = layer_outputs
        self.dropped_out_final = dropped_out_final


class _AwdLSTMLayer(torch.nn.Module):

    def __init__(self, input_size, hidden_size, dropout_input, dropout_recurrent_weights):
        super().__init__()
        self.dropout_input_layer = LockedDropout(dropout_input)
        rnn_layer = torch.nn.LSTM(input_size, hidden_size, num_layers=1)
        if dropout_recurrent_weights is not None:
            rnn_layer = WeightDrop(rnn_layer, dropout=dropout_recurrent_weights)
        self.rnn_layer = rnn_layer
        self._hidden_size = hidden_size
        self._input_size = input_size
        self.hidden = None

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def input_size(self):
        return self._input_size

    def reset(self, batch_size):
        first = None
        for p in self.parameters():
            first = p
            break
        hidden = torch.zeros(1, batch_size, self.hidden_size, dtype=first.dtype, device=first.device)
        self.hidden = (hidden, hidden)

    def forward(self, x):
        """ Invoked during the forward propagation of the RNNEncoder module.
        Args:
            x (Tensor): input of shape (sentence length x batch_size)

        Returns:
            output, hidden (see pytorch docs)
        """
        # noinspection PyCallingNonCallable
        x = self.dropout_input_layer(x)
        if self.hidden is None or x.size(1) != self.hidden[0].size(1):
            self.reset(x.size(1))
        h_init, c_init = self.hidden
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output, hidden = self.rnn_layer(x, self.hidden)
            self.hidden = hidden[0].detach(), hidden[1].detach()

        return AwdLSTMLayerOutput(
            dropped_out_input=x, h_init=h_init, c_init=c_init, h_final=hidden[0], c_final=hidden[1], h=output)


class PretrainedModel(torch.nn.Module):
    """ An abstract class to handle weights initialization and
            a simple interface for dowloading and loading pretrained models.
        """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None,
                        from_tf=False, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and _cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the _cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from _cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu' if not torch.cuda.is_available() else None)
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


def load_pre_trained_weights_and_transfer_to_new_token_ids(pre_trained_path, key, *copy_to_keys):

    weights = torch.load(pre_trained_path, map_location='cpu' if not torch.cuda.is_available() else None)
    embedding_weights = weights[key].cpu().numpy()

    mean_word = np.mean(embedding_weights, axis=0)

    new_w = np.zeros((len(numerical_tokens), mean_word.shape[0]), dtype=np.float32)

    pre_trained_numerical_tokens = NumericalTokens.load(numerical_tokens_path)

    for token in numerical_tokens.tokens():
        token_id = numerical_tokens.to_id(token)
        if token_id >= new_w.shape[0]:
            raise ValueError('numerical_tokens has not been compressed: shape: {}, {}, id: {}'.format(
                new_w.shape, token, token_id))

        if token_id == numerical_tokens.unknown_id:
            pre_trained_id = pre_trained_numerical_tokens.unknown_id
        elif token_id == numerical_tokens.padding_id:
            pre_trained_id = pre_trained_numerical_tokens.padding_id
        elif token in pre_trained_numerical_tokens:
            pre_trained_id = pre_trained_numerical_tokens.to_id(token)
        else:
            pre_trained_id = -1

        new_w[token_id] = embedding_weights[pre_trained_id] if pre_trained_id >= 0 else mean_word

    weights[key] = torch.as_tensor(np.ascontiguousarray(new_w))
    for k in copy_to_keys:
        weights[k] = torch.as_tensor(np.ascontiguousarray(np.copy(new_w)))

    return weights


def load_ulmfit(
        pre_trained_path, numerical_tokens_path, numerical_tokens,
        encoder_kwargs=None, decoder_kwargs=None, is_tie_embedding=True):

    encoder_kwargs = encoder_kwargs or {}
    decoder_kwargs = decoder_kwargs or {}

    weights = load_pre_trained_weights_and_transfer_to_new_token_ids(
        pre_trained_path, numerical_tokens_path, numerical_tokens,
        '0.encoder.weight',
        '0.encoder_with_dropout.embed.weight',
        '1.decoder.weight')

    size_embedding = weights['0.encoder.weight'].size(1)
    size_hidden = weights['0.rnns.0.module.weight_hh_l0_raw'].size(1)

    max_idx_layer = -1
    for s in weights.keys():
        if s.startswith('0.rnns.'):
            s = s[len('0.rnns.'):]
            next_dot = s.index('.')
            idx_layer = int(s[:next_dot])
            max_idx_layer = max(max_idx_layer, idx_layer)

    num_layers = max_idx_layer + 1

    encoder = AwdLSTM(
        len(numerical_tokens), size_embedding, size_hidden, num_layers, numerical_tokens.padding_id, **encoder_kwargs)

    encoder_weights = OrderedDict()
    for name in weights:
        orig_name = name
        if not name.startswith('0.'):
            continue
        name = name[len('0.'):]
        if name.startswith('encoder_with_dropout.embed.'):
            name = 'dropped_out_embedding.embedding.' + name[len('encoder_with_dropout.embed.'):]
        elif name.startswith('encoder'):
            continue  # no need to keep this around, only the weights inside the dropout layer are used
        elif name.startswith('rnns.'):
            name = name[len('rnns.'):]
            next_dot = name.index('.')
            name = 'layers.' + name[:next_dot] + '.rnn_layer' + name[next_dot:]
        else:
            raise ValueError('Unable to convert name to new format: {}, {}'.format(orig_name, name))
        encoder_weights[name] = weights[orig_name]

    encoder.load_state_dict(encoder_weights)

    decoder_weights = OrderedDict()

    for name in weights:
        orig_name = name
        if not name.startswith('1.'):
            continue
        name = name[len('1.'):]
        decoder_weights[name] = weights[orig_name]

    tie_encoder = encoder.dropped_out_embedding.embedding if is_tie_embedding else None
    decoder = LinearDecoder(
        len(numerical_tokens), size_embedding, tie_encoder=tie_encoder, **decoder_kwargs)

    return encoder, decoder


class AwdLSTM(torch.nn.Module):

    """A custom RNN encoder network that uses
        - an embedding matrix to encode input,
        - a stack of LSTM layers to drive the network, and
        - variational dropouts in the embedding and LSTM layers
        The architecture for this network was inspired by the work done in
        "Regularizing and Optimizing LSTM Language Models".
        (https://arxiv.org/pdf/1708.02182.pdf)
    """

    def __init__(
            self, num_vocabulary, size_embedding, size_hidden, num_layers, padding_token_id,
            dropout_hidden=0.3, dropout_input=0.65, dropout_embedding=0.1, dropout_recurrent_weights=0.5,
            initialization_range=0.1, dropout_final=0.5, output_transform=None):
        """ Default constructor for the RNN_Encoder class
            Args:
                num_vocabulary (int): number of vocabulary (or tokens) in the source dataset
                size_embedding (int): the embedding size to use to encode each token
                size_hidden (int): number of hidden activation per LSTM layer
                num_layers (int): number of LSTM layers to use in the architecture
                padding_token_id (int): the int value used for padding text.
                dropout_hidden (float): dropout to apply to the activations going from one LSTM layer to another
                dropout_input (float): dropout to apply to the input layer.
                dropout_embedding (float): dropout to apply to the embedding layer.
                dropout_recurrent_weights (float): dropout used for a LSTM's internal (or hidden) recurrent weights.
          """

        if num_layers < 2:
            raise ValueError('Must use at least 2 layers')

        super().__init__()
        self.dropped_out_embedding = EmbeddingDropout(
            torch.nn.Embedding(num_vocabulary, size_embedding, padding_idx=padding_token_id),
            dropout_embedding)
        self.layers = torch.nn.ModuleList()
        for idx_layer in range(num_layers):
            if idx_layer == 0:
                self.layers.append(
                    _AwdLSTMLayer(size_embedding, size_hidden, dropout_input, dropout_recurrent_weights))
            elif idx_layer == num_layers - 1:
                self.layers.append(
                    _AwdLSTMLayer(size_hidden, size_embedding, dropout_hidden, dropout_recurrent_weights))
            else:
                self.layers.append(_AwdLSTMLayer(size_hidden, size_hidden, dropout_hidden, dropout_recurrent_weights))
        self.dropped_out_embedding.embedding.weight.data.uniform_(-initialization_range, initialization_range)
        self.size_output = size_embedding
        self.size_embedding = size_embedding
        self.size_hidden = size_hidden
        self.output_transform = output_transform
        self.dropout_final = dropout_final

    def total_channels(self, indicator_which_layers):
        if len(indicator_which_layers) != len(self.layers) + 1:
            raise ValueError('indicator_which_layers has length {}, but expected {}'.format(
                len(indicator_which_layers), len(self.layers) + 1))

        result = 0
        for idx_layer, is_included in enumerate(indicator_which_layers):
            if not is_included:
                continue
            # idx_layer == 0 is the embedding, idx_layer == 1 is the output of layer 0, i.e. the input to layer 1
            if idx_layer == 0:
                result += self.size_embedding
            elif idx_layer == len(self.layers):
                result += self.size_output
            else:
                result += self.size_hidden

        return result

    def forward(self, x):
        """ Invoked during the forward propagation of the RNNEncoder module.
        Args:
            x (Tensor): input of shape (sentence length x batch_size)

        Returns:
            An AwdLSTMOutput instance
        """
        # noinspection PyCallingNonCallable
        dropped_out_embedding = self.dropped_out_embedding(x)
        x = dropped_out_embedding

        layer_results = list()
        for layer in self.layers:
            layer_results.append(layer(x))
            x = layer_results[-1].h

        result = AwdLSTMOutput(
            dropped_out_embedding=dropped_out_embedding,
            layer_outputs=layer_results,
            dropped_out_final=locked_dropout(self.training, x, self.dropout_final))
        if self.output_transform is not None:
            result = self.output_transform(result)
        return result

    def reset(self, batch_size):
        for layer in self.layers:
            layer.reset(batch_size)

    def adjust_dropouts(
            self, dropout_hidden=None, dropout_input=None, dropout_embedding=None, dropout_recurrent_weights=None,
            dropout_final=None):
        if dropout_embedding is not None:
            self.dropped_out_embedding.dropout = dropout_embedding
        if dropout_input is not None:
            self.layers[0].dropout_input_layer.p = dropout_input
        if dropout_hidden is not None:
            for idx in range(1, len(self.layers)):
                self.layers[idx].dropout_input_layer.p = dropout_hidden
        if dropout_recurrent_weights is not None:
            for layer in self.layers:
                layer.rnn_layer.dropout = dropout_recurrent_weights
        if dropout_final is not None:
            self.dropout_final = dropout_final
