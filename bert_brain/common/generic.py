import inspect
from itertools import zip_longest
import re


__all__ = ['zip_equal', 'copy_from_properties', 'get_keyword_properties', 'SwitchRemember', 'camel_to_snake',
           'MultiReplace', 'split_with_indices']


def zip_equal(*it):
    """
    Like zip, but raises a ValueError if the iterables are not of equal length
    Args:
        *it: The iterables to zip

    Returns:
        yields a tuple of items, one from each iterable
    """
    # wrap the iterators in an enumerate to guarantee that None is a legitimate sentinel
    iterators = [enumerate(i) for i in it]
    for idx, item in enumerate(zip_longest(*iterators)):
        try:
            result = tuple(part[1] for part in item)
            yield result
        except TypeError:
            culprit = None
            for idx_part, part in enumerate(item):
                if part is None:
                    culprit = idx_part
                    break
            raise ValueError(
                'Unequal number of elements in iterators. Problem occurred at index: {}, iterator_index: {}'.format(
                    idx, culprit))


def copy_from_properties(instance, **kwargs):
    """
    Returns a copy of instance by calling __init__ with keyword arguments matching the properties of instance.
    The values of these keyword arguments are taken from the properties of instance except where overridden by
    kwargs. Thus for a class Foo with properties [a, b, c], copy_from_properties(instance, a=7) is equivalent to
    Foo(a=7, b=instance.b, c=instance.c)
    Notes:
        Now that Python includes dataclasses, using dataclasses is generally preferred to this method.
    Args:
        instance: The instance to use as a template
        **kwargs: The keyword arguments to __init__ that should not come from the current instance's properties

    Returns:
        A copy of instance modified according to kwargs
    """
    property_names = [n for n, v in inspect.getmembers(type(instance), lambda m: isinstance(m, property))]
    init_kwargs = inspect.getfullargspec(type(instance).__init__).args

    def __iterate_key_values():
        for k in init_kwargs[1:]:
            if k in kwargs:
                yield k, kwargs[k]
            elif k in property_names:
                yield k, getattr(instance, k)

    return type(instance)(**dict(__iterate_key_values()))


def get_keyword_properties(instance, just_names=False):
    """
    Related to copy_from_properties, this method gets key-value pairs from an instance and returns them as a list.
    The key-value pairs are returned in the order the keys are specified in the init method for all keys which are
    in the init method and which are also properties of the instance.
    Args:
        instance: The object from which to get key-value pairs
        just_names: If True, then the resulting list contains only the keys and not the values.
    Returns:
        key_value_pairs: A list of 2-tuples containing the keys and values as described.
    """
    property_names = [n for n, v in inspect.getmembers(type(instance), lambda m: isinstance(m, property))]
    init_kwargs = inspect.getfullargspec(type(instance).__init__).args

    if just_names:
        return [k for k in init_kwargs if k in property_names]

    return [(k, getattr(instance, k)) for k in init_kwargs if k in property_names]


class SwitchRemember:

    def __init__(self, var):
        """
        Wraps a value which will be tested for equality so that all values it is compared with are stored.
        Useful for writing a 'switch' statement and raising a descriptive ValueError when no values are matched which
        contains all of the possible legitimate values. E.g.:

        user_input = SwitchRemember(user_input)

        if user_input == 'item1':
            # do something
            ...
        elif user_input == 'item2':
            # do something else
            ...
        elif user_input == 'item3':
            # do a third thing
            ...
        else:
            raise ValueError('Unrecognized value for user_input. Valid values are: {}'.format(user_input.tests)

        Args:
            var: The underlying value to test for equality
        """
        self.var = var
        self._tests = set()

    @property
    def tests(self):
        return list(sorted(self._tests))

    def __eq__(self, test):
        self._tests.add(test)
        return self.var == test


def camel_to_snake(s):
    """
    Converts from CamelCase to snake_case
    Args:
        s: string to convert

    Returns:
        converted string
    """
    # https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def split_with_indices(s):
    """
    Splits on whitespace and returns the indices of each token along with the tokens
    Args:
        s: string to split

    Returns:
        A list of (index, token) pairs
    """
    return [(match.start(), match.group()) for match in re.finditer(r'\S+', s)]


class MultiReplace:
    def __init__(self, replace_dict):
        """
        Utility class which is similar to string.replace, but which matches multiple items simultaneously. Useful
        for cleaning text
        Args:
            replace_dict: A map from patterns to their replacements
        """
        self._replace_dict = dict(replace_dict)
        self._regex = re.compile('|'.join(re.escape(k) for k in self._replace_dict))

    def _get_replacement(self, match):
        return self._replace_dict[match.group(0)]

    def replace(self, text):
        return self._regex.sub(self._get_replacement, text)
