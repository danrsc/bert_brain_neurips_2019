# -*- coding: utf-8 -*-

# https://raw.githubusercontent.com/scjs/tgre/master/tgre/tgre.py

"""Read, write, and modify Praat TextGrid annotations.

Praat is a speech analysis and manipulation program, and TextGrids are
a text annotation format for sound files and speech recordings. The
TextGrid format is documented here:
http://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html

This module provides classes to read and work with TextGrids in Python,
and to write them out into a file format that can be read by Praat.

References
----------
Boersma, Paul & Weenink, David (2016). Praat: doing phonetics by
computer [Computer program]. Retrieved from http://www.praat.org/

Examples
--------
Use `TextGrid.from_file()` to read a TextGrid file.

>>> # noinspection PyUnresolvedReferences
import praat_textgrid
>>> tg = tgre.TextGrid.from_file('test/files/usage-example.TextGrid')
>>> print(tg)
<TextGrid from 0 to 2.5 seconds with 3 tiers>

Access tiers through the `tiers` attribute, which stores a list of tiers.

>>> len(tg.tiers)
3
>>> print(tg.tiers[0])
<IntervalTier "Pat" from 0 to 2.5 seconds with 2 intervals>
>>> tg.tiers[0].name
'Pat'
>>> tg.tiers[0].xmin
0
>>> tg.tiers[0].xmax
2.5
>>> for tier in tg.tiers:
...    print(tier)
<IntervalTier "Pat" from 0 to 2.5 seconds with 2 intervals>
<IntervalTier "Sam" from 0 to 2.5 seconds with 3 intervals>
<TextTier "Metronome" from 0 to 2.5 seconds with 3 points>

Tiers can also be accessed by slicing or iterating through the TextGrid object.

>>> print(tg[0])
<IntervalTier "Pat" from 0 to 2.5 seconds with 2 intervals>
>>> for tier in tg:
...    print(tier)
<IntervalTier "Pat" from 0 to 2.5 seconds with 2 intervals>
<IntervalTier "Sam" from 0 to 2.5 seconds with 3 intervals>
<TextTier "Metronome" from 0 to 2.5 seconds with 3 points>

Access intervals and points by slicing or iterating through the tier object.

>>> print(tg[1][1])
<Interval "ciao" from 1.125 to 1.45>
>>> tg[1][1].text
'ciao'
>>> tg[1][1].xmin
1.125
>>> tg[1][1].xmax
1.45
>>> for interval in tg[1]:
...    print(interval)
<Interval "" from 0 to 1.125>
<Interval "ciao" from 1.125 to 1.45>
<Interval "" from 1.45 to 2.5>

Points have `mark` (the label for a point) and `number` (the timestamp for a
point) rather than `text`, `xmin`, and `xmax`.

>>> print(tg[2][0])
<Point "click" at 0.75>
>>> tg[2][0].mark
'click'
>>> tg[2][0].number
0.75

Convert a TextGrid to a dict.

>>> tg_dict = tg.to_dict()
>>> print(tg_dict)
{'xmin': 0,
 'xmax': 2.5,
 'tiers': [{'name': 'Pat',
            'class': 'IntervalTier',
            'xmin': 0,
            'xmax': 2.5,
            'intervals': [{'xmin': 0,
                           'xmax': 0.65,
                           'text': 'hello'},
                          {'xmin': 0.65,
                           'xmax': 2.5,
                           'text': ''}
                            ]
            },
           {'name': 'Sam',
            'class': 'IntervalTier',
            'xmin': 0,
            'xmax': 2.5,
            'intervals': [{'xmin': 0,
                           'xmax': 1.125,
                           'text': ''},
                          {'xmin': 1.125,
                           'xmax': 1.45,
                           'text': 'ciao'},
                          {'xmin': 1.45,
                           'xmax': 2.5,
                           'text': ''}
                         ]
           },
           {'name': 'Metronome',
            'class': 'TextTier',
            'xmin': 0,
            'xmax': 2.5,
            'points': [{'number': 0.75,
                        'mark': 'click'},
                       {'number': 1.5,
                        'mark': 'click'},
                       {'number': 2.25,
                        'mark': 'click'}
                         ]
           }]
}

Create new intervals.

>>> hi = tgre.Interval(0.4, 0.55, 'hi')
>>> pat = tgre.Interval(0.55, 0.85, 'pat')

Create a new tier. This tier is from 0 to 2.5 seconds, and contains the two
intervals that were just created.

>>> tier = tgre.IntervalTier("words", 0, 2.5, items=[hi, pat])
>>> print(tier)
<IntervalTier "words" from 0 to 2.5 seconds with 2 intervals>
>>> print(tier[0])
<Interval "hi" from 0.4 to 0.55>
>>> print(tier[1])
<Interval "pat" from 0.55 to 0.85>

Add a new interval to an existing tier with the `insert()` method.

>>> tier.insert(0.3, 0.4, 'oh')
>>> print(tier)
<IntervalTier "words" from 0 to 2.5 seconds with 3 intervals>
>>> print(tier[0])
<Interval "oh" from 0.3 to 0.4>
>>> print(tier[1])
<Interval "hi" from 0.4 to 0.55>
>>> print(tier[2])
<Interval "pat" from 0.55 to 0.85>

Remove an interval from a tier with `del`.

>>> del tier[2]
>>> print(tier)
<IntervalTier "words" from 0 to 2.5 seconds with 2 intervals>

Find the interval on a tier at a particular time with the `where()` method.

>>> interval = tier.where(0.48)
>>> print(interval)
<Interval "hi" from 0.4 to 0.55>

Add a tier to a TextGrid by modifying the list in the `tiers` attribute.

>>> tg.tiers.append(tier)
>>> len(tg)
4
>>> for tier in tg:
...    print(tier)
<IntervalTier "Pat" from 0 to 2.5 seconds with 2 intervals>
<IntervalTier "Sam" from 0 to 2.5 seconds with 3 intervals>
<TextTier "Metronome" from 0 to 2.5 seconds with 3 points>
<IntervalTier "words" from 0 to 2.5 seconds with 2 intervals>

Create a new TextGrid.

>>> first_tier = tgre.IntervalTier("segments", 0, 5)
>>> second_tier = tgre.IntervalTier("words", 0, 5)
>>> new_tg = tgre.TextGrid(0, 5, tiers=[first_tier, second_tier])
>>> print(new_tg)
<TextGrid from 0 to 5 seconds with 2 tiers>

Write a TextGrid to a file with the `to_praat()` method.

>>> new_tg.to_praat(path='mytextgrid.TextGrid')

This method creates a TextGrid file with a different format than the ones
that Praat creates, but it will still be read normally by Praat (and also by
this module, if you read it back in with the `TextGrid.from_file()` method).
The `to_praat()` method will raise a ValueError if the result will
not be read correctly by Praat (for example, if there are intervals with
negative duration).

If a TextGrid is encoded in UTF-16, the `TextGrid.from_file()` and
`TextGrid.to_praat()` methods should be called with an optional `encoding`
parameter.

>>> tg = tgre.TextGrid.from_file('test/files/numbers.TextGrid', encoding='utf_16')

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import bisect
import functools
import io
import re


PRAAT_REGEX = re.compile(r"""(
(^|\s)(
        "(?P<string>([^"]|"")*)"|
         (?P<int>\d+)|
         (?P<real>[\d.]+)|
         (?P<flag><exists>)
     )(!.*)?(?=$|\s)|
(!.*)
)""", re.VERBOSE | re.UNICODE)


def praat_reader(text):
    """Yield strings and numbers from text files created by Praat.

    Escaped double-quotes are yielded with the escape character
    removed.

    Parameters
    ----------
    text : str
        Contents of an ooTextFile text file written by Praat, such as
        a TextGrid.

    Yields
    ------
    str or int or float
        Stream of strings and numbers.

    """

    for match in re.finditer(PRAAT_REGEX, text):
        groupdict = match.groupdict()

        if groupdict['string'] is not None:
            yield groupdict['string'].replace('""', '"')

        elif groupdict['int'] is not None:
            yield int(groupdict['int'])

        elif groupdict['real'] is not None:
            yield float(groupdict['real'])

        elif groupdict['flag'] is not None:
            pass


def praat_string(text):
    """Return a string formatted to be recognized by Praat.

    Parameters
    ----------
    text : str
        String to be formatted for Praat.

    Returns
    -------
    str

    """

    return '"' + text.replace('"', '""') + '"'


def tier_from_reader(stream):
    """Return a tier object from a stream of strings and numbers.

    Parameters
    ----------
    stream : iterator of str, int, and float
        Iterator that yields strings and numbers in the order that
        Praat expects to define a tier.

    Returns
    -------
    IntervalTier or TextTier

    """

    tier_class = next(stream)

    if tier_class == 'IntervalTier':
        return IntervalTier.from_reader(stream)

    elif tier_class == 'TextTier':
        return TextTier.from_reader(stream)

    else:
        raise ValueError('Tier type "{}" not recognized'.format(tier_class))


class TextGrid(object):
    """Representation of a Praat TextGrid annotation file.

    Parameters
    ----------
    xmin : int or float
        Start time of the TextGrid, in seconds.

    xmax : int or float
        End time of the TextGrid, in seconds.

    tiers : list of IntervalTier and PointTier
        Annotation tiers that are associated with this TextGrid.

    Attributes
    ----------
    xmin
    xmax
    tiers

    """

    def __init__(self, xmin, xmax, tiers):
        self.xmin = xmin
        self.xmax = xmax
        self.tiers = tiers

    def __repr__(self):
        rep = repr(self.xmin), repr(self.xmax), repr(self.tiers)

        return 'TextGrid({0}, {1}, {2})'.format(*rep)

    def __str__(self):
        return ('<TextGrid from {0.xmin} to {0.xmax} seconds with {1} tiers>'
                .format(self, len(self)))

    def __getitem__(self, i):
        return self.tiers[i]

    def __setitem__(self, key, val):
        self.tiers[key] = val

    def __delitem__(self, key):
        del self.tiers[key]

    def __len__(self):
        return len(self.tiers)

    def __reversed__(self):
        return reversed(self.tiers)

    def __iter__(self):
        return iter(self.tiers)

    @classmethod
    def from_reader(cls, stream):
        """Return a TextGrid from a stream of strings and numbers.

        Parameters
        ----------
        stream : iterator of str, int, and float
            Iterator that yields strings and numbers in the order that
            Praat expects to define a complete TextGrid.

        Returns
        -------
        TextGrid

        """

        xmin = next(stream)
        xmax = next(stream)
        size = next(stream)

        tiers = [tier_from_reader(stream) for i in range(size)]

        last = next(stream, None)

        if last is not None:
            raise ValueError('Unexpected value "{}" found after reading tiers'
                             .format(last))

        return cls(xmin, xmax, tiers)

    @classmethod
    def from_file(cls, path, encoding='utf_8'):
        """Return a TextGrid parsed from a text file created by Praat.

        Parameters
        ----------
        path : str
            Path to a TextGrid file created by Praat.

        encoding : {'utf_8', 'utf_16'}
            Text encoding of the TextGrid file. Default is 'utf_8'.

        Returns
        -------
        TextGrid

        Notes
        -----
        See http://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html
        for documentation on the Praat text file format.

        """

        with io.open(path, encoding=encoding) as textgrid_file:
            elements = praat_reader(textgrid_file.read())

        if next(elements) != 'ooTextFile':
            raise ValueError('Header string "ooTextFile" missing')

        if next(elements) != 'TextGrid':
            raise ValueError('Header string "TextGrid" missing')

        return cls.from_reader(elements)

    def to_dict(self):
        """Return a dict representation of this TextGrid and its tiers.

        Unlike `to_praat()`, this method doesn't verify that the
        TextGrid and its tiers conform to the TextGrid standard. For
        example, it doesn't check that all intervals have positive
        duration, and it doesn't add empty intervals to ensure that an
        IntervalTier has no gaps.

        Returns
        -------
        dict

        """

        tg_dict = vars(self).copy()
        tg_dict['tiers'] = [tier.to_dict() for tier in self.tiers]

        return tg_dict

    def to_praat(self, path=None, encoding='utf_8'):
        """Write this TextGrid to a file readable by Praat.

        Parameters
        ----------
        path : str
            Path to the file where this TextGrid should be written. If
            None, this method returns the TextGrid as a string in a
            format readable by Praat. Default is None.

        encoding : {'utf_8', 'utf_16'}
            Text encoding to use for the file. Default is 'utf_8'.

        Returns
        -------
        None or str
            If `path` is None, return this TextGrid object as a string.

        Raises
        ------
        ValueError
            If the TextGrid doesn't conform to the TextGrid standard.

        """

        for tier in self.tiers:
            if tier.xmin < self.xmin:
                raise ValueError('Tier "{}" starts before TextGrid begins'
                                 .format(tier.name))

            if tier.xmin > self.xmin:
                raise ValueError('Tier "{}" starts after TextGrid begins'
                                 .format(tier.name))

            if tier.xmax > self.xmax:
                raise ValueError('Tier "{}" continues past end of TextGrid'
                                 .format(tier.name))

            if tier.xmax < self.xmax:
                raise ValueError('Tier "{}" ends before end of TextGrid'
                                 .format(tier.name))

        output = ('"ooTextFile"\n"TextGrid"\n'
                  '{0.xmin:.16g} to {0.xmax:.16g} seconds <exists>\n'
                  '{1} tiers\n\n'
                  .format(self, len(self.tiers)))

        output += '\n\n'.join(tier.to_praat() for tier in self.tiers)

        if path is None:
            return output

        with io.open(path, 'w', encoding=encoding) as textgrid_file:
            textgrid_file.write(output)


@functools.total_ordering
class Interval(object):
    """An interval in a TextGrid IntervalTier.

    An interval is a sound file annotation with a duration. For
    example, there might be an interval that marks an [i] vowel
    from 0.35 seconds to 0.42 seconds of the sound file that is
    associated with this TextGrid.

    Parameters
    ----------
    xmin : int or float
        Start time of the interval, in seconds.

    xmax : int or float
        End time of the interval, in seconds.

    text : str
        Label for the interval, possibly an empty string.

    Attributes
    ----------
    xmin
    xmax
    text

    """

    plural = 'intervals'

    def __init__(self, xmin, xmax, text):
        self.xmin = xmin
        self.xmax = xmax
        self.text = text

    def __repr__(self):
        rep = repr(self.xmin), repr(self.xmax), repr(self.text)

        return 'Interval({0}, {1}, {2})'.format(*rep)

    def __str__(self):
        return '<Interval "{0.text}" from {0.xmin} to {0.xmax}>'.format(self)

    def __lt__(self, other):
        return self.xmin < other.xmin

    @classmethod
    def from_reader(cls, stream):
        """Return an Interval from a stream of strings and numbers.

        Parameters
        ----------
        stream : iterator of str, int, and float
            Iterator that yields two numbers and a string, giving the
            xmin, xmax, and text parameters for this Interval.

        Returns
        -------
        Interval

        """

        xmin = next(stream)
        xmax = next(stream)
        text = next(stream)

        return cls(xmin, xmax, text)

    def to_praat(self):
        """Return the Interval as text readable by Praat.

        Returns
        -------
        str

        """

        return (' {0.xmin:23.16g}{0.xmax:24.16g}    {1} '
                .format(self, praat_string(self.text)))


@functools.total_ordering
class Point(object):
    """A point in a TextGrid TextTier.

    A point is a sound file annotation with no duration. For example,
    there might be a point that marks the release of a stop closure
    at 0.65 seconds in the sound file that is associated with this
    TextGrid.

    Parameters
    ----------
    number : int or float
        Time of the point, in seconds.

    mark : str
        Label for the point, possibly an empty string.

    Attributes
    ----------
    number
    mark

    """

    plural = 'points'

    def __init__(self, number, mark):
        self.number = number
        self.mark = mark

    def __repr__(self):
        rep = repr(self.number), repr(self.mark)

        return 'Point({0}, {1})'.format(*rep)

    def __str__(self):
        return '<Point "{0.mark}" at {0.number}>'.format(self)

    def __lt__(self, other):
        return self.number < other.number

    @classmethod
    def from_reader(cls, stream):
        """Return a Point from a stream of strings and numbers.

        Parameters
        ----------
        stream : iterator of str, int, and float
            Iterator that yields one numbers and one string, giving the
            time and mark parameters for this Point.

        Returns
        -------
        Point

        """

        number = next(stream)
        mark = next(stream)

        return cls(number, mark)

    def to_praat(self):
        """Return the Point as text readable by Praat.

        Returns
        -------
        str

        """

        return (' {0:23.16g}    {1} '
                .format(self.number, praat_string(self.mark)))


class Tier(object):
    """Base class for IntervalTier and TextTier.

    Only subclasses should be instantiated. Subclasses should implement:

    1. A class attribute `item` which is a reference to a class like
    `Interval` or `Point`.
    2. A method `check_items` which returns a chronological list of
    items for the tier. The method should deal with invalid items and
    insert empty intervals as needed.

    """

    def __init__(self, name, xmin, xmax, items=None):
        self.name = name
        self.xmin = xmin
        self.xmax = xmax

        if items is not None:
            try:
                self._items = sorted(items)

            except AttributeError:
                raise TypeError('Items cannot be sorted together')

        else:
            self._items = []

    def __repr__(self):
        rep = (self.__class__.__name__, repr(self.name), repr(self.xmin),
               repr(self.xmax), repr(self._items))

        return '{0}({1}, {2}, {3}, {4})'.format(*rep)

    def __str__(self):
        return ('<{0.__class__.__name__} "{0.name}" '
                'from {0.xmin} to {0.xmax} seconds '
                'with {1} {0.item.plural}>'
                .format(self, len(self)))

    def __getitem__(self, key):
        return self._items[key]

    def __delitem__(self, key):
        del self._items[key]

    def __reversed__(self):
        return reversed(self._items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    @classmethod
    def from_reader(cls, stream):
        """Return a TextGrid tier from a stream of strings and numbers.

        Parameters
        ----------
        stream : iterator of str, int, and float
            Strings and numbers that define a tier, in the order
            expected by Praat.

        Returns
        -------
        Tier

        """

        name = next(stream)
        xmin = next(stream)
        xmax = next(stream)
        size = next(stream)

        items = []

        for i in range(size):
            items.append(cls.item.from_reader(stream))

        return cls(name, xmin, xmax, items)

    def insert(self, *args, **kwargs):
        """Add a new item (Interval or Point, as appropriate) to the tier.

        See `help(IntervalTier.item)` or `help(TextTier.item)` for
        parameters.

        """

        bisect.insort(self._items, self.item(*args, **kwargs))

    def to_dict(self):
        """Return a dict representation of this Tier.

        Returns
        -------
        dict

        """

        res = vars(self).copy()
        res['class'] = self.__class__.__name__

        items = [vars(item).copy() for item in sorted(res.pop('_items'))]
        res[self.item.plural] = items

        return res

    def to_praat(self):
        """Return the Tier as text readable by Praat.

        Returns
        -------
        str

        """

        items = self.check_items()

        header = ['{0} named {1} '
                  .format(praat_string(self.__class__.__name__),
                          praat_string(self.name)),

                  'From {0.xmin:.16g} to {0.xmax:.16g} seconds with {1} {0.item.plural}'
                  .format(self, len(items))]

        return '\n'.join(header + [item.to_praat() for item in items])


class IntervalTier(Tier):
    """TextGrid tier containing Interval annotations.

    Intervals can be accessed by slicing or iterating over the tier.
    Intervals can be added to the tier on initialization, or by using
    the `insert()` method.

    Parameters
    ----------
    name : str
        Name of the tier.

    xmin : int or float
        Start time of the tier, in seconds.

    xmax : int or float
        End time of the tier, in seconds.

    items : list of Interval, optional
        List of the intervals that should be included in this tier.
        These will be sorted when they are stored in the IntervalTier.
        Default is None (the tier is initialized with no intervals).

    Attributes
    ----------
    name
    xmin
    xmax

    """

    item = Interval

    def check_items(self):
        """Return a chronological list of the intervals in the IntervalTier.

        If there are gaps in the IntervalTier (i.e., times between the
        tier's start and end times that don't correspond to any interval),
        empty intervals with no text are inserted into the returned list
        of intervals in order to fill out the tier.

        Returns
        -------
        intervals : list of Interval

        Raises
        ------
        ValueError
            If an interval does not have a positive duration, if it is
            outside the bounds (`xmin` and `xmax`) for this tier, or if
            its duration overlaps with another interval on this tier.

        """

        intervals = []
        prev = self.xmin

        for item in sorted(self._items):
            if item.xmin >= item.xmax:
                raise ValueError('Bad interval (xmin >= xmax): {}'
                                 .format(item))

            if item.xmin < prev:
                if prev == self.xmin:
                    raise ValueError('Interval at {} starts before tier begins'
                                     .format(item.xmin))

                raise ValueError('Overlapping intervals at {}, {}'
                                 .format(item.xmin, prev))

            if item.xmax > self.xmax:
                raise ValueError('Interval at {} extends past end of tier'
                                 .format(item.xmin))

            if item.xmin > prev:
                intervals.append(self.item(prev, item.xmin, ''))

            intervals.append(item)
            prev = item.xmax

        if prev < self.xmax:
            intervals.append(self.item(prev, self.xmax, ''))

        return intervals

    def where(self, time):
        """Return the interval in this tier at a given time.

        If there is an interval that overlaps with the given time,
        return a reference to it. Inclusive of `xmin`, so an interval
        is returned if `time` is in `(xmin, xmax]`. If no such interval
        exists, return None.

        Parameters
        ----------
        time : int or float
            Time at which to look for an interval.

        Returns
        -------
        Interval or None

        """

        keys = [item.xmax for item in self._items]
        idx = bisect.bisect(keys, time)

        try:
            item = self._items[idx]

        except IndexError:
            return None

        if item.xmin <= time:
            return self._items[idx]

        return None


class TextTier(Tier):
    """TextGrid tier containing Point annotations.

    Points can be accessed by slicing or iterating over the tier.
    Points can be added to the tier on initialization, or by using
    the `insert()` method.

    Parameters
    ----------
    name : str
        Name of the tier.

    xmin : int or float
        Start time of the tier, in seconds.

    xmax : int or float
        End time of the tier, in seconds.

    items : list of Point, optional
        List of the points that should be included in this tier. These
        will be sorted when they are stored in the TextTier. Default is
        None (the tier is initialized with no points).

    Attributes
    ----------
    name
    xmin
    xmax

    """

    item = Point

    def check_items(self):
        """Return a chronological list of the points in the TextTier.

        Returns
        -------
        points : list of Point

        Raises
        ------
        ValueError
            If a point is outside the bounds (`xmin` and `xmax`) for
            this tier, or if there are two points at the same time.

        """

        numbers = set()

        for item in self._items:
            if item.number in numbers:
                raise ValueError('Multiple points at time {}'
                                 .format(item.number))

            if item.number < self.xmin:
                raise ValueError('Point at time {} occurs before tier begins'
                                 .format(item.number))

            if item.number > self.xmax:
                raise ValueError('Point at time {} occurs after end of tier'
                                 .format(item.number))

            numbers.add(item.number)

        return sorted(self._items)

    def where(self, left, right=None):
        """Return the point(s) in this tier at or between the given time(s).

        If only `left` is specified, return a reference to the point at
        that time. If no such point exists, return None. If `right` is
        also specified, return a list of references to points between
        `left` and `right` (inclusive).

        Parameters
        ----------
        left : int or float
            Time at which to look for a point, or the left boundary
            when looking for points over a range of times.

        right : int or float, optional
            Right boundary when looking for points over a range of
            times.

        Returns
        -------
        Point or None or list of Point

        """

        keys = [item.number for item in self._items]
        left_idx = bisect.bisect_left(keys, left)

        if right is not None:
            right_idx = bisect.bisect(keys, right)
            return self._items[left_idx:right_idx]

        try:
            item = self._items[left_idx]

        except IndexError:
            return None

        if item.number == left:
            return self._items[left_idx]

        return None
