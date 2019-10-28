import itertools
import random
from collections import OrderedDict
from dataclasses import dataclass
import dataclasses
from typing import Tuple

import numpy as np

from .tree_module import DependencyTree, Arc
from .conll_reader import universal_dependency_reader


# based on code in https://github.com/facebookresearch/colorlessgreenRNNs


__all__ = [
    'preprocess_english_morphology', 'collect_paradigms', 'make_token_to_paradigms', 'make_ltm_to_word',
    'morph_contexts_frequencies', 'extract_dependency_patterns', 'choose_random_forms', 'generate_morph_pattern_test',
    'alternate_number_morphology', 'get_alternate_number_form', 'plurality', 'SyntaxPattern', 'GeneratedExample']


@dataclass
class SyntaxPattern:
    arc_direction: str
    context: Tuple[str, ...]
    left_value_1: str
    left_value_2: str

    def delimited(self, field_delimiter='!', context_delimiter='_'):
        d = dataclasses.asdict(self, dict_factory=OrderedDict)
        result = list()
        for field in d:
            if field == 'context':
                result.append(context_delimiter.join(self.context))
            else:
                result.append('{}'.format(d[field]))
        return field_delimiter.join(result)

    @classmethod
    def from_delimited(cls, delimited, field_delimiter='!', context_delimiter='_'):
        fields = dataclasses.fields(cls)
        values = delimited.split(field_delimiter)
        if len(values) != len(fields):
            raise ValueError('Number of fields in input ({}) does not match number of fields in {} ({})'.format(
                len(values), cls, len(fields)))
        d = dict()
        for field, str_value in zip(fields, values):
            if field.name == 'context':
                d[field.name] = str_value.split(context_delimiter)
            else:
                d[field.name] = field.type(str_value)
        return cls(**d)


@dataclass
class GeneratedExample:
    pattern: SyntaxPattern
    construction_id: int
    sentence_id: int
    right_index: int
    right_pos: str
    right_morph: str
    form: str
    number: str
    alternate_form: str
    lemma: str
    left_index: int
    left_pos: str
    prefix: str
    generated_context: str

    def delimited(self, field_delimiter='\t', pattern_field_delimiter='!', pattern_context_delimiter='_'):
        d = dataclasses.asdict(self, dict_factory=OrderedDict)
        result = list()
        for field in d:
            if field == 'pattern':
                result.append(self.pattern.delimited(pattern_field_delimiter, pattern_context_delimiter))
            else:
                result.append('{}'.format(d[field]))
        return field_delimiter.join(result)

    @classmethod
    def from_delimited(
            cls, delimited, field_delimiter='\t', pattern_field_delimiter='!', pattern_context_delimiter='_'):
        fields = dataclasses.fields(cls)
        values = delimited.split(field_delimiter)
        if len(values) != len(fields):
            raise ValueError('Number of fields in input ({}) does not match number of fields in {} ({})'.format(
                len(values), cls, len(fields)))
        d = dict()
        for field, str_value in zip(fields, values):
            if field.name == 'pattern':
                d[field.name] = field.type.from_delimited(str_value, pattern_field_delimiter, pattern_context_delimiter)
            else:
                d[field.name] = field.type(str_value)
        return cls(**d)

    @property
    def agreement_tuple(self):
        return self.generated_context.split(), self.form, self.alternate_form, self.right_index


def preprocess_english_morphology(morph):
    # https://github.com/facebookresearch/colorlessgreenRNNs
    # /blob/8d41f2a2301d612ce25be90dfc1e96f828f77c85/src/data/preprocess_EnglishUD_morph.py
    # annotate non-singular verbs in present as Plural
    if 'Tense=Pres' in morph and 'VerbForm=Fin' in morph and 'Number=Sing' not in morph:
        morph = morph + '|Number=Plur'
        return _sort_morphology(morph)
    elif 'Number=Sing' in morph:
        feats = sorted(morph.split('|'))
        # remove Person=3 annotation (since we don't have it for non-singular cases)
        feats = [f for f in feats if 'Person=3' not in f]
        return '|'.join(feats)
    return morph


def _sort_morphology(morph):
    return '|'.join(sorted(morph.split('|')))


def collect_paradigms(path, reader=universal_dependency_reader, min_freq=5, morphology_preprocess_fn=None):
    paradigms = dict()
    if isinstance(path, str):
        path = [path]
    for sentence, text in reader.iterate_sentences_chain_streams(
            path, morphology_preprocess_fn=morphology_preprocess_fn):
        tree = DependencyTree.from_conll_rows(sentence, reader.root_index, reader.offset, text)
        tree.remerge_segmented_morphemes()  # probably not necessary for English?
        for node in tree.nodes:
            key = node.word, node.lemma, node.pos, _sort_morphology(node.morph)
            if key in paradigms:
                paradigms[key] += 1
            else:
                paradigms[key] = 1
    return dict((k, paradigms[k]) for k in paradigms if paradigms[k] >= min_freq)


def make_token_to_paradigms(paradigms):
    result = dict()
    for token, lemma, pos, morph in paradigms:
        if token not in result:
            result[token] = list()
        result[token].append((lemma, pos, morph, paradigms[(token, lemma, pos, morph)]))
    return result


def make_ltm_to_word(tok_to_paradigms):
    paradigm_lemmas = dict()
    for token in tok_to_paradigms:
        for lemma, tag, morph, freq in tok_to_paradigms[token]:
            if (lemma, tag) not in paradigm_lemmas:
                paradigm_lemmas[(lemma, tag)] = dict()
            if morph not in paradigm_lemmas[(lemma, tag)]:
                paradigm_lemmas[(lemma, tag)][morph] = dict()
            paradigm_lemmas[(lemma, tag)][morph][token] = freq

    result = dict()
    for lemma, tag in paradigm_lemmas:
        for morph in paradigm_lemmas[(lemma, tag)]:
            best_word = sorted(paradigm_lemmas[(lemma, tag)][morph].items(), key=lambda word_freq: -word_freq[1])[0][0]
            if lemma not in result:
                result[lemma] = dict()
            if tag not in result[lemma]:
                result[lemma][tag] = dict()
            result[lemma][tag][morph] = best_word
    return result


def _safe_log(x, minimum=0.0001):
    return np.log(np.maximum(x, minimum))


def _cond_entropy(xy):

    # normalise
    xy = xy / np.sum(xy)

    x_ = np.sum(xy, axis=1)
    y_ = np.sum(xy, axis=0)

    x_y = xy / y_
    # print(x_y)
    y_x = xy / x_.reshape(x_.shape[0], 1)
    # print(y_x)

    # Entropies: H(x|y) H(y|x) H(x) H(y)
    return np.sum(-xy * _safe_log(x_y)), np.sum(-xy * _safe_log(y_x)), np.sum(-x_ * _safe_log(x_)), np.sum(
        -y_ * _safe_log(y_))


def _pos_structure(nodes, arc):
    """ Get a sequence of pos tags for nodes which are direct children of the arc head or the arc child
        nodes - the list of nodes of the context Y, between the head and the child (X, Z) of the arc
    """
    return tuple([n.pos for n in nodes if n.head_id in [arc.head.index, arc.child.index]])


def _inside(tree, a):
    if a.child.index < a.head.index:
        nodes = tree.nodes[a.child.index: a.head.index - 1]
        left = a.child
        right = a.head
    else:
        nodes = tree.nodes[a.head.index: a.child.index - 1]
        left = a.head
        right = a.child
    return nodes, left, right


def _filtered_features(morphology, feature_keys):
    if feature_keys is None:
        return morphology

    all_features = morphology.split('|')
    features = tuple(f for f in all_features if f.split('=')[0] in feature_keys)
    return '|'.join(features)


def morph_contexts_frequencies(trees, feature_keys):
    """
    Collect frequencies for X Y Z tuples, where Y is a context defined by its surface structure
    and X and Z are connected by a dependency

    :param trees: dependency trees
    :param feature_keys: a list or set of feature names such as Number, Gender, ...
    :return: two dictionaries for left and right dependencies
    """

    d_left = dict()
    d_right = dict()
    for t in trees:
        for a in t.arcs:
            if 3 < a.arc_length() < 15 and t.is_projective_arc(a):
                # print("\n".join(str(n) for n in t.nodes))
                nodes, l, r = _inside(t, a)
                substring = (l.pos,) + _pos_structure(nodes, a) + (r.pos,)
                # print(substring)
                if substring:
                    l_features = _filtered_features(l.morph, feature_keys)
                    r_features = _filtered_features(r.morph, feature_keys)
                    if len(l_features) == 0 or len(r_features) == 0:
                        continue
                    key = l_features, r_features
                    # substring = substring + (a.dep_label,)
                    if a.direction == Arc.DirectionLeft:
                        if substring not in d_left:
                            d_left[substring] = dict()
                        if key not in d_left[substring]:
                            d_left[substring][key] = 0
                        d_left[substring][key] += 1
                    if a.direction == Arc.DirectionRight:
                        if substring not in d_right:
                            d_right[substring] = dict()
                        if key not in d_right[substring]:
                            d_right[substring][key] = 0
                        d_right[substring][key] += 1
    return d_left, d_right


def _find_good_patterns(arc_direction, context_dict, freq_threshold):
    """
    :param arc_direction: The arc direction of the context_dict
    :param context_dict: is a dictionary of type { Y context : {(X, Z) : count} }
                         for X Y Z sequences where X and Z could be of any type (tags, morph)

    :param freq_threshold: for filtering out too infrequent patterns
    :return: list of patterns - tuples (context, left1, left2) == (Y, X1, X2)
             (where X1 and X2 occur with different Zs)
    """
    patterns = []
    for context in context_dict:
        left_right_pairs = context_dict[context].keys()
        if len(left_right_pairs) == 0:
            continue
        left, right = zip(*left_right_pairs)

        left_v = set(left)

        d = context_dict[context]
        if len(left_v) < 2:
            continue

        for l1, l2 in itertools.combinations(left_v, 2):
            right_v = [r for (l, r) in left_right_pairs if l in (l1, l2)]
            if len(right_v) < 2:
                continue

            a = np.zeros((2, len(right_v)))
            for i, x in enumerate((l1, l2)):
                for j, y in enumerate(right_v):
                    a[(i, j)] = d[(x, y)] if (x, y) in d else 0

            l_r, _, _, _ = _cond_entropy(a)
            # mi = l_e - l_r

            count_l1 = 0
            count_l2 = 0
            for lt, r in d:
                if lt == l1:
                    count_l1 += d[(lt, r)]
                if lt == l2:
                    count_l2 += d[(lt, r)]

            #  print(l_r, r_l, l_e, r_e, mi)
            if l_r < 0.001 and count_l1 > freq_threshold and count_l2 > freq_threshold:
                patterns.append(SyntaxPattern(arc_direction, context, l1, l2))
                # print(context, l_r, mi)
                # print(l1, l2, count_l1, count_l2)
                #  for l, r in d:
                #      if l in (l1, l2) and d[(l, r)] > 0 :
                #          print(l, r, d[(l, r)])
    return patterns


def _grep_morph_pattern(trees, context, l_values, dep_dir, feature_keys=None):
    """
    :param trees: The DependencyTree sequence in which to look for patterns
    :param context: Y
    :param l_values:  l_values are relevant X values
    :param dep_dir:
    :param feature_keys: The set of features to consider (e.g. Number, Case, ...)
    :return: generator of (context-type, l, r, tree, Y nodes) tuples
    """
    if feature_keys is None:
        feature_keys = ['Number']

    for t in trees:
        for a in t.arcs:
            if 3 < a.arc_length() < 15 and t.is_projective_arc(a):
                if a.child.pos == "PUNCT" or a.head.pos == "PUNCT":
                    continue

                #  print("\n".join(str(n) for n in t.nodes))
                nodes, l, r = _inside(t, a)

                if a.direction != dep_dir:
                    continue

                if not any(m in l.morph for m in l_values):
                    #  print(features(l.morph), l_values)
                    continue
                if _filtered_features(r.morph, feature_keys) != _filtered_features(l.morph, feature_keys):
                    continue
                substring = (l.pos,) + _pos_structure(nodes, a) + (r.pos,)

                if substring == context:
                    #  print(substring, context)
                    yield context, l, r, t, nodes


def _filter_infrequent(context_dependency_counts, minimum_count):
    result = dict()
    for context in context_dependency_counts:
        for key in context_dependency_counts[context]:
            count = context_dependency_counts[context][key]
            if count >= minimum_count:
                if context not in result:
                    result[context] = dict()
                result[context][key] = count
    return result


def extract_dependency_patterns(trees, freq_threshold, feature_keys):

    context_left_dependency_counts, context_right_dependency_counts = morph_contexts_frequencies(trees, feature_keys)

    # filtering very infrequent cases
    filter_threshold = 2
    context_left_dependency_counts = _filter_infrequent(context_left_dependency_counts, filter_threshold)
    context_right_dependency_counts = _filter_infrequent(context_right_dependency_counts, filter_threshold)

    # print('Finding good patterns')
    good_patterns_left = _find_good_patterns(Arc.DirectionLeft, context_left_dependency_counts, freq_threshold)
    good_patterns_right = _find_good_patterns(Arc.DirectionRight, context_right_dependency_counts, freq_threshold)

    # print('Saving patterns and sentences matching them')

    return good_patterns_left + good_patterns_right


def _is_content_word(pos):
    return pos in ["ADJ", "NOUN", "VERB", "PROPN", "NUM", "ADV"]


def _generate_context(nodes, paradigms, tokenizer):
    output = []

    for i in range(len(nodes)):
        substitutes = []
        n = nodes[i]
        # substituting content words
        if _is_content_word(n.pos):
            for word in paradigms:
                if word == n.word:
                    continue
                # matching capitalization and vowel
                if not _match_features(word, n.word):
                    continue

                tag_set = set([p[1] for p in paradigms[word]])
                # use words with unambiguous POS
                if len(tag_set) == 1 and tag_set.pop() == n.pos:
                    for _, _, morph, freq in paradigms[word]:
                        if n.morph == morph and int(freq) > 1 and _is_token(word, tokenizer):
                            substitutes.append(word)
            if len(substitutes) == 0:
                output.append(n.word)
            else:
                output.append(random.choice(substitutes))
        else:
            output.append(n.word)
    return " ".join(output)


def choose_random_forms(ltm_paradigms, tokenizer, gold_pos, morph, n_samples=10, gold_word=None):
    candidates = set()

    # lemma_tag_pairs = ltm_paradigms.keys()
    # test_lemmas = [l for l, t in lemma_tag_pairs]

    for lemma in ltm_paradigms:
        poses = list(ltm_paradigms[lemma].keys())
        if len(set(poses)) == 1 and poses.pop() == gold_pos:
            form = ltm_paradigms[lemma][gold_pos][morph] if morph in ltm_paradigms[lemma][gold_pos] else ''
            _, morph_alt = alternate_number_morphology(morph)
            form_alt = ltm_paradigms[lemma][gold_pos][morph_alt] if morph_alt in ltm_paradigms[lemma][gold_pos] else ''

            if not _is_good_form(gold_word, form, morph, lemma, gold_pos, tokenizer, ltm_paradigms):
                continue

            candidates.add((lemma, form, form_alt))

    if len(candidates) > n_samples:
        return random.sample(candidates, n_samples)
    else:
        return random.sample(candidates, len(candidates))


def generate_morph_pattern_test(trees, pattern, ltm_paradigms, paradigms, tokenizer, n_sentences=10):

    output = []
    constr_id = 0

    n_vocab_unk = 0
    n_paradigms_unk = 0
    # 'nodes' constitute Y, without X or Z included
    for context, l, r, t, nodes in _grep_morph_pattern(
            trees, pattern.context, [pattern.left_value_1, pattern.left_value_2], pattern.arc_direction):
        # pos_constr = "_".join(n.pos for n in t.nodes[l.index - 1: r.index])

        # filter model sentences with unk and the choice word not in vocab
        if not all(_is_token(n.word, tokenizer) for n in nodes + [l, r]):
            n_vocab_unk += 1
            continue
        if not _is_good_form(r.word, r.word, r.morph, r.lemma, r.pos, tokenizer, ltm_paradigms):
            n_paradigms_unk += 1
            continue

        prefix = ' '.join(n.word for n in t.nodes[:r.index])

        for i in range(n_sentences):
            # sent_id = 0 - original sentence with good lexical items, other sentences are generated
            if i == 0:
                new_context = " ".join(n.word for n in t.nodes)
                form = r.word
                form_alt = get_alternate_number_form(r.lemma, r.pos, r.morph, ltm_paradigms)
                lemma = r.lemma
            else:
                new_context = _generate_context(t.nodes, paradigms, tokenizer)
                random_forms = choose_random_forms(
                    ltm_paradigms, tokenizer, r.pos, r.morph, n_samples=1, gold_word=r.word)
                if len(random_forms) > 0:
                    lemma, form, form_alt = random_forms[0]
                else:
                    # in rare cases, there is no (form, form_alt) both in vocab
                    # original form and its alternation are not found because e.g. one or the other is not in paradigms
                    # (they should anyway be in the vocabulary)
                    lemma, form = r.lemma, r.word
                    form_alt = get_alternate_number_form(r.lemma, r.pos, r.morph, ltm_paradigms)

            number = plurality(r.morph)
            generated_example = GeneratedExample(
                pattern, constr_id, i, r.index - 1, r.pos, r.morph, form, number, form_alt, lemma, l.index - 1, l.pos,
                prefix, new_context)

            output.append(generated_example)

        constr_id += 1

    print("Problematic sentences vocab/paradigms", n_vocab_unk, n_paradigms_unk)
    return output


def alternate_number_morphology(morphology):
    if 'Number=Plur' in morphology:
        morph_alt = morphology.replace('Plur', 'Sing')
        return 'plur', morph_alt
    elif 'Number=Sing' in morphology:
        morph_alt = morphology.replace('Sing', 'Plur')
        return 'sing', morph_alt
    return None, morphology


def plurality(morphology):
    if 'Number=Plur' in morphology:
        return 'plur'
    elif 'Number=Sing' in morphology:
        return 'sing'
    else:
        return None


def _is_token(candidate, tokenizer):
    if candidate is None or len(candidate) == 0:
        return False
    test = tokenizer.tokenize(candidate)
    if any(t == '[UNK]' for t in test):
        return False
    return True


def _is_good_form(gold_form, new_form, gold_morph, lemma, pos, tokenizer, ltm_paradigms):
    _, alt_morph = alternate_number_morphology(gold_morph)
    if not _is_token(new_form, tokenizer):
        return False
    if lemma not in ltm_paradigms or pos not in ltm_paradigms[lemma]:
        return False
    alt_form = ltm_paradigms[lemma][pos][alt_morph] if alt_morph in ltm_paradigms[lemma][pos] else ''
    if not _is_token(alt_form, tokenizer):
        return False
    if gold_form is None:
        print(gold_form, gold_morph)
        return True
    if not _match_features(new_form, gold_form):
        return False
    if not _match_features(alt_form, gold_form):
        return False
    return True


def _is_vowel(c):
    return c in ['a', 'o', 'u', 'e', 'i', 'A', 'O', 'U', 'E', 'I', 'è']


def _match_features(w1, w2):
    return w1[0].isupper() == w2[0].isupper() and _is_vowel(w1[0]) == _is_vowel(w2[0])


def get_alternate_number_form(lemma, pos, morph, ltm_paradigms):
    _, alt_morph = alternate_number_morphology(morph)
    return ltm_paradigms[lemma][pos][alt_morph]
