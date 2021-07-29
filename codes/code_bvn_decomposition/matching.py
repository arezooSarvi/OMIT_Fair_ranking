# This module uses material from the Wikipedia article Hopcroft--Karp algorithm
# <https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm>, accessed on
# January 3, 2015, which is released under the Creative Commons
# Attribution-Share-Alike License 3.0
# <http://creativecommons.org/licenses/by-sa/3.0/>. That article includes
# pseudocode, which has been translated into the corresponding Python code.
#
# Portions of this module use code from David Eppstein's Python Algorithms and
# Data Structures (PADS) library, which is dedicated to the public domain (for
# proof, see <http://www.ics.uci.edu/~eppstein/PADS/ABOUT-PADS.txt>).
"""Provides functions for computing maximum cardinality matchings and minimum
weight full matchings in a bipartite graph.

If you don't care about the particular implementation of the maximum matching
algorithm, simply use the :func:`maximum_matching`. If you do care, you can
import one of the named maximum matching algorithms directly.

For example, to find a maximum matching in the complete bipartite graph with
two vertices on the left and three vertices on the right:

>>> G = nx.complete_bipartite_graph(2, 3)
>>> left, right = nx.bipartite.sets(G)
>>> list(left)
[0, 1]
>>> list(right)
[2, 3, 4]
>>> nx.bipartite.maximum_matching(G)
{0: 2, 1: 3, 2: 0, 3: 1}

The dictionary returned by :func:`maximum_matching` includes a mapping for
vertices in both the left and right vertex sets.

Similarly, :func:`minimum_weight_full_matching` produces, for a complete
weighted bipartite graph, a matching whose cardinality is the cardinality of
the smaller of the two partitions, and for which the sum of the weights of the
edges included in the matching is minimal.

"""
import collections
import itertools

from networkx.algorithms.bipartite.matrix import biadjacency_matrix
from networkx.algorithms.bipartite import sets as bipartite_sets
import networkx as nx
from outlier_metrics import *


INFINITY = float("inf")


def quality_fn(indices, scores, quality_measure=count_outliers):
    q = quality_measure([scores[i] for i in indices])
    return q


def hopcroft_karp_matching(
    G,
    quality,
    top_nodes=None,
    reverse=False,
    local_search=False,
    top_k=None,
    scores=None,
    right_nodes_counter = None,
):
    """Returns the maximum cardinality matching of the bipartite graph `G`.

    A matching is a set of edges that do not share any nodes. A maximum
    cardinality matching is a matching with the most edges possible. It
    is not always unique. Finding a matching in a bipartite graph can be
    treated as a networkx flow problem.

    The functions ``hopcroft_karp_matching`` and ``maximum_matching``
    are aliases of the same function.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    top_nodes : container of nodes

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed. But if more than one solution exists an exception
      will be raised.

    Returns
    -------
    matches : dictionary

      The matching is returned as a dictionary, `matches`, such that
      ``matches[v] == w`` if node `v` is matched to node `w`. Unmatched
      nodes do not occur as a key in `matches`.

    Raises
    ------
    AmbiguousSolution
      Raised if the input bipartite graph is disconnected and no container
      with all nodes in one bipartite set is provided. When determining
      the nodes in each bipartite set more than one valid solution is
      possible if the input graph is disconnected.

    Notes
    -----
    This function is implemented with the `Hopcroft--Karp matching algorithm
    <https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm>`_ for
    bipartite graphs.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    maximum_matching
    hopcroft_karp_matching
    eppstein_matching

    References
    ----------
    .. [1] John E. Hopcroft and Richard M. Karp. "An n^{5 / 2} Algorithm for
       Maximum Matchings in Bipartite Graphs" In: **SIAM Journal of Computing**
       2.4 (1973), pp. 225--231. <https://doi.org/10.1137/0202019>.

    """
    # First we define some auxiliary search functions.
    #
    # If you are a human reading these auxiliary search functions, the "global"
    # variables `leftmatches`, `rightmatches`, `distances`, etc. are defined
    # below the functions, so that they are initialized close to the initial
    # invocation of the search functions.
    def breadth_first_search():
        for v in left:
            if leftmatches[v] is None:
                distances[v] = 0
                queue.append(v)
            else:
                distances[v] = INFINITY
        distances[None] = INFINITY
        while queue:
            v = queue.popleft()
            if distances[v] < distances[None]:
                for u in G[v]:
                    if len(rightmatches[u]) < right_nodes_counter.get(u, 0):
                        distances[None] = distances[v] + 1
                    else:
                        for rightmatch in rightmatches[u]:
                            if distances[rightmatch] is INFINITY:
                                distances[rightmatch] = distances[v] + 1
                                queue.append(rightmatch)
        return distances[None] is not INFINITY

    def depth_first_search(v):
        if v is not None:
            if type(v) == int:
                v = [v]
            for v_ in v:
                for u in G[v_]:
                    # If u not matched with full count we have a new path.
                    if not rightmatches[u] or len(rightmatches[u]) < right_nodes_counter.get(u, 0):
                        # We need to remove the previous connection of v_
                        if leftmatches[v_]:
                            rightmatches[leftmatches[v_]].remove(v_)
                        leftmatches[v_] = u
                        rightmatches[u].append(v_)
                        return True
                    # If u matched with full count we need to look further for a path
                    else:
                        for r in rightmatches[u]:
                            if distances[r] == distances[v_] + 1:
                                if len(rightmatches[u]) < right_nodes_counter.get(u, 0):
                                    rightmatches[u].append(v_)
                                    leftmatches[v_] = u
                                    return True
                                elif depth_first_search(r):
                                    if leftmatches[v_]:
                                        rightmatches[leftmatches[v_]].remove(v_)
                                    rightmatches[u].append(v_)
                                    leftmatches[v_] = u
                                    return True
                distances[v_] = INFINITY
            return False
        return True

    # Initialize the "global" variables that maintain state during the search.
    left, right = bipartite_sets(G, top_nodes)
    leftmatches = {v: None for v in left}
    rightmatches = {v: [] for v in right}
    distances = {}
    queue = collections.deque()

    if right_nodes_counter is None:
        right_nodes_counter = {r: 1 for r in right}

    # This is vanilla BvN: It is faster since we add each matching found in the breadth first search
    # instead of adding just one matching with the lowest quality an repeating the breadth first search
    # to find new potential candidates.
    if quality is None:
        num_matched_pairs = 0
        while breadth_first_search():
            for v in left:
                if leftmatches[v] is None:
                    if depth_first_search(v):
                        num_matched_pairs += 1

    else:
        while breadth_first_search():
            indices = [v for v in left if leftmatches[v] is not None]
            candidates = []
            for v in left:
                if leftmatches[v] is None:
                    indices.append(v)
                    item_quality = quality(indices)
                    candidates.append((v, item_quality))
                    indices.pop()

            # sort the candidates wrt their quality.
            candidates.sort(key=lambda pair: pair[1], reverse=reverse)

            for v, _ in candidates:
                if depth_first_search(v):
                    break
    # TODO: see what we can do with remembering previous examples to improve performance
    prev_matchings = []
    top_k_indices = sorted(list(right))[:top_k]
    if local_search:
        # if quality_fn([rightmatches[v] for v in top_k_indices], scores=scores) > 0:
        # print("starting local search")
        while quality_fn([rightmatches[v] for v in top_k_indices], scores=scores) > 0:
            if leftmatches in prev_matchings:
                break  # We do not want to repeat the same search over and over again
            prev_matchings.append(leftmatches)
            # remove all matchings after the top_k
            for i in sorted(list(right))[top_k:]:
                leftmatches[rightmatches[i]] = None
                rightmatches[i] = None
            # calculate outlieriness of each item
            max_outlier = (-1, -1)
            top_k_list = [scores[i] for i in [rightmatches[v] for v in top_k_indices]]
            std = np.std(top_k_list)
            mean = np.mean(top_k_list)
            for i in top_k_indices:
                outlierness = abs((rightmatches[i] - mean) / std)
                if outlierness > max_outlier[0]:
                    max_outlier = (outlierness, i)

            # remove item that most likely causes the outlier
            leftmatches[rightmatches[max_outlier[1]]] = None
            rightmatches[max_outlier[1]] = None

            # go back to breadth_first_search
            # TODO: This is copied code
            while breadth_first_search():
                indices = [v for v in left if leftmatches[v] is not None]
                candidates = []
                for v in left:
                    if leftmatches[v] is None:
                        indices.append(v)
                        item_quality = quality(indices)
                        candidates.append((v, item_quality))
                        indices.pop()

                # sort the candidates wrt their quality.
                candidates.sort(key=lambda pair: pair[1], reverse=reverse)

                for v, _ in candidates:
                    if depth_first_search(v):
                        break

    # Strip the entries matched to `None`.
    leftmatches = {k: v for k, v in leftmatches.items() if v is not None}
    rightmatches = {k: v for k, v in rightmatches.items() if v is not None}

    # At this point, the left matches and the right matches are inverses of one
    # another. In other words,
    #
    #     leftmatches == {v, k for k, v in rightmatches.items()}
    #
    # Finally, we combine both the left matches and right matches.
    return dict(itertools.chain(leftmatches.items(), rightmatches.items()))
