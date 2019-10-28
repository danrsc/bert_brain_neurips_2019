# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""

Classes Node, Arc, DependencyTree providing functionality for syntactic dependency trees
https://raw.githubusercontent.com/facebookresearch/colorlessgreenRNNs/8d41f2a2301d612ce25be90dfc1e96f828f77c85/src/syntactic_testsets/tree_module.py

"""

import re
from queue import Queue
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List

from .conll_reader import ConllRow


__all__ = ['DependencyTree', 'Node', 'Arc']


_root_word = '[ROOT]'


@dataclass
class Node:
    index: Optional[int] = None
    word: str = ''
    lemma: str = ''
    head_id: Optional[int] = None
    pos: str = ''
    dependency_label: str = ''
    morph: Optional[str] = None
    size: Optional[int] = None
    direction: Optional[str] = None

    @property
    def is_root(self):
        return self.word == _root_word and self.pos == _root_word


@dataclass
class Arc:
    DirectionLeft = 'L'
    DirectionRight = 'R'

    head: Node
    direction: str
    child: Node

    @property
    def dependency_label(self):
        return self.child.dependency_label

    def arc_length(self):
        # arcs to ROOT node have length 0
        if self.head.is_root:
            return 0
        else:
            return abs(self.child.index - self.head.index)


class DependencyTree(object):
    def __init__(
            self,
            nodes: List[Node],
            arcs: List[Arc],
            fused_nodes: List[Tuple[int, int, str]],
            text: Optional[str] = None):
        self.nodes: List[Node] = nodes
        self.arcs: List[Arc] = arcs
        self.assign_sizes_to_nodes()
        # for UD annotation to be able to recover original sentence (without split morphemes)
        self.fused_nodes: List[Tuple[int, int, str]] = fused_nodes
        self.text: Optional[str] = text

    @staticmethod
    def from_conll_rows(
            sentence_conll_rows: Sequence[ConllRow],
            root_index: int,
            offset: int,
            text: Optional[str] = None) -> 'DependencyTree':
        nodes = list()
        fused_nodes = list()

        for index_row, row in enumerate(sentence_conll_rows):

            # saving original word segments separated in UD (e.g. Italian darglielo -> dare + gli + lo)
            if re.match(r"[0-9]+-[0-9]+", row.index):
                fused_nodes.append((int(row.index.split("-")[0]), int(row.index.split("-")[1]), row.word))
                continue
            # empty elements (e.g. copula in Russian)
            if re.match(r"[0-9]+\.[0-9]+", row.index):
                continue

            index = int(row.index) if row.index is not None else index_row
            nodes.append(
                Node(index,
                     row.word,
                     row.lemma,
                     int(row.head_id),
                     pos=row.pos,
                     dependency_label=row.dependency_label,
                     morph=row.morph))

        arcs = []
        for idx_node, node in enumerate(nodes):
            if node.head_id == root_index:
                generic_root = Node(root_index, _root_word, _root_word, 0, _root_word, size=0)
                arcs.append(Arc(generic_root, Arc.DirectionLeft, node))
            else:
                head_element = nodes[node.head_id - offset]
                if node.head_id < int(node.index):
                    arcs.append(Arc(head_element, Arc.DirectionRight, node))
                    node.direction = Arc.DirectionRight
                else:
                    arcs.append(Arc(head_element, Arc.DirectionLeft, node))
                    node.direction = Arc.DirectionLeft

        return DependencyTree(nodes, arcs, fused_nodes, text=text)

    def __str__(self):
        return '\n'.join([str(n) for n in self.nodes])

    def __repr__(self):
        return str(self)

    def children(self, head: Node):
        children = []
        for arc in self.arcs:
            if arc.head == head:
                children.append(arc.child)
        return children

    def assign_sizes_to_nodes(self):
        for node in self.nodes:
            node.size = len(self.children(node)) + 1

    def reindex(self, root_index, offset):
        """ After reordering 'nodes' list reflects the final order of nodes, however the indices of node objects
          do not correspond to this order. This function fixes it. """
        new_positions = dict((n.index, i) for i, n in enumerate(self.nodes))
        for i, node in enumerate(self.nodes):
            node.index = i + offset
            if node.head_id != root_index:
                node.head_id = new_positions[node.head_id] + offset

    def remove_node(self, node_x):
        assert len(self.children(node_x)) == 0
        self.nodes.remove(node_x)
        for node in self.nodes:
            if node.head_id > node_x.index:
                node.head_id = node.head_id - 1
            if node.index > node_x.index:
                node.index = node.index - 1

        for i in range(len(self.fused_nodes)):
            start, end, token = self.fused_nodes[i]
            if start > node_x.index:
                start = start - 1
            if end > node_x.index:
                end = end - 1
            self.fused_nodes[i] = (start, end, token)

    def subtree(self, head):
        elements = dict()
        queue = Queue()
        queue.put(head)
        #  head_ = Node(head.index, head.word, head.pos + "X")
        elements[head.index] = head
        visited = set()
        while not queue.empty():
            next_node = queue.get()
            if next_node.index in visited:
                continue
            visited.add(next_node.index)
            for child in self.children(next_node):
                elements[child.index] = child
                queue.put(child)

        return [elements[i] for i in sorted(elements)]

    def is_projective_arc(self, arc):
        st = self.subtree(arc.head)
        # all nodes in subtree of the arc head
        st_idx = [node.index for node in st]
        # span between the child and the head
        indexes = range(arc.child.index + 1, arc.head.index) if arc.child.index < arc.head.index else range(
            arc.head.index + 1, arc.child.index)
        # each node/word between child and head should be part of the subtree
        # if not, than the child-head arc is crossed by some other arc and is non-projective
        for i in indexes:
            if i not in st_idx:
                return False
        return True

    def is_projective(self):
        return all(self.is_projective_arc(arc) for arc in self.arcs)

    def length(self):
        return sum(arc.arc_length() for arc in self.arcs)

    def average_branching_factor(self):
        heads = [node.head_id for node in self.nodes]
        return len(self.nodes)/len(set(heads))

    def remerge_segmented_morphemes(self):
        """
        UD format only: Remove segmented words and morphemes and substitute them by the original word form
        - all children of the segments are attached to the merged word form
        - word form features are assigned heuristically (should work for Italian, not sure about other languages)
            - pos tag and morphology (zero?) comes from the first morpheme
        :return:
        """
        for start, end, token in self.fused_nodes:
            # assert start + 1 == end, t
            self.nodes[start - 1].word = token

            for i in range(end - start):
                # print(i)
                if len(self.children(self.nodes[start])) != 0:
                    for c in self.children(self.nodes[start]):
                        c.head_id = self.nodes[start - 1].index
                        self.arcs.remove(Arc(child=c, head=self.nodes[start], direction=c.direction))
                        self.arcs.append(Arc(child=c, head=self.nodes[start - 1], direction=c.direction))
                assert len(self.children(self.nodes[start])) == 0, (self, start, end, token, i, self.arcs)
                self.remove_node(self.nodes[start])
                # print(t)
                #        print(t)
        self.fused_nodes = []

    def to_conll_rows(self):
        return [ConllRow(
            str(node.index), node.word, node.lemma, str(node.head_id), node.pos, node.dependency_label, node.morph)
            for node in self.nodes]
