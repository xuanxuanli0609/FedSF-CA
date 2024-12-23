# built-ins
import itertools as it
import sys
import argparse
import logging
import json
from copy import deepcopy

# libraries
from numpy import (array, mean, zeros, zeros_like, unique,
    newaxis, nonzero, median, float, ones, arange, inf, isnan,
    flatnonzero, unravel_index, bincount)
from tqdm import tqdm
import numpy as np
from scipy.stats import sem
from scipy.sparse import lil_matrix
from scipy.special import comb as nchoosek
from scipy.ndimage.measurements import label
import networkx as nx
from networkx import Graph, biconnected_components
from networkx.algorithms.traversal.depth_first_search import dfs_preorder_nodes
from skimage.segmentation import relabel_sequential

from viridis import tree

# local modules
from . import agglo2
from . import morpho
from . import sparselol as lol
from . import iterprogress as ip
from . import optimized as opt
from .ncut import ncutW
from .mergequeue import MergeQueue
from .evaluate import merge_contingency_table, split_vi, xlogx
from . import evaluate as ev
from . import classify
from .classify import get_classifier, \
    unique_learning_data_elements, concatenate_data_elements
from .dtypes import label_dtype


arguments = argparse.ArgumentParser(add_help=False)
arggroup = arguments.add_argument_group('Agglomeration options')
arggroup.add_argument('-t', '--thresholds', nargs='+', default=[128],
    type=float, metavar='FLOAT',
    help='''The agglomeration thresholds. One output file will be written
        for each threshold.'''
)
arggroup.add_argument('-l', '--ladder', type=int, metavar='SIZE',
    help='Merge any bodies smaller than SIZE.'
)
arggroup.add_argument('-p', '--pre-ladder', action='store_true', default=True,
    help='Run ladder before normal agglomeration (default).'
)
arggroup.add_argument('-L', '--post-ladder',
    action='store_false', dest='pre_ladder',
    help='Run ladder after normal agglomeration instead of before (SLOW).'
)
arggroup.add_argument('-s', '--strict-ladder', type=int, metavar='INT',
    default=1,
    help='''Specify the strictness of the ladder agglomeration. Level 1
        (default): merge anything smaller than the ladder threshold as
        long as it's not on the volume border. Level 2: only merge smaller
        bodies to larger ones. Level 3: only merge when the border is
        larger than or equal to 2 pixels.'''
)
arggroup.add_argument('-M', '--low-memory', action='store_true',
    help='''Use less memory at a slight speed cost. Note that the phrase
        'low memory' is relative.'''
)
arggroup.add_argument('--disallow-shared-boundaries', action='store_false',
    dest='allow_shared_boundaries',
    help='''Watershed pixels that are shared between more than 2 labels are
        not counted as edges.'''
)
arggroup.add_argument('--allow-shared-boundaries', action='store_true',
    default=True,
    help='''Count every watershed pixel in every edge in which it participates
        (default: True).'''
)


def conditional_countdown(seq, start=1, pred=bool):
    """Count down from 'start' each time pred(elem) is true for elem in seq.

    Used to know how many elements of a sequence remain that satisfy a
    predicate.

    Parameters
    ----------
    seq : iterable
        Any sequence.
    start : int, optional
        The starting element.
    pred : function, type(next(seq)) -> bool
        A predicate acting on the elements of `seq`.

    Examples
    --------
    >>> seq = range(10)
    >>> cc = conditional_countdown(seq, start=5, pred=lambda x: x % 2 == 1)
    >>> next(cc)
    5
    >>> next(cc)
    4
    >>> next(cc)
    4
    >>> next(cc)
    3
    """
    remaining = start
    for elem in seq:
        if pred(elem):
            remaining -= 1
        yield remaining


############################
# Merge priority functions #
############################


def batchify(func):
    """Convert classical (g, n1, n2) -> f policy to batch (g, [e]) -> [f]

    This is meant for policies that wouldn't gain much from batch evaluation
    or that aren't used very much.

    Parameters
    ----------
    func : function
        A merge priority function with signature (g, n1, n2) -> f.

    Returns
    -------
    batch_func : function
        A batch merge priority function with signature (g, [(n1, n2)]) -> [f].
    """
    def batch_func(g, edges):
        result = []
        for n1, n2 in edges:
            result.append(func(g, n1, n2))
        return result
    return batch_func


@batchify
def oriented_boundary_mean(g, n1, n2):
    return mean(g.oriented_probabilities_r[g.boundary(n1, n2)])


@batchify
def boundary_mean(g, n1, n2):
    return mean(g.probabilities_r[g.boundary(n1, n2)])


@batchify
def boundary_median(g, n1, n2):
    return median(g.probabilities_r[g.boundary(n1, n2)])


@batchify
def approximate_boundary_mean(g, n1, n2):
    """Return the boundary mean as computed by a MomentsFeatureManager.

    The feature manager is assumed to have been set up for g at construction.
    """
    return g.feature_manager.compute_edge_features(g, n1, n2)[1]


def make_ladder(priority_function, threshold, strictness=1):
    """Convert priority function to merge small segments first.

    Small segments tend to mess with other segmentation metrics, so we
    merge them early so that more sophisticated function can work on big
    segments. This is particularly useful for bad fragment generation
    methods that generate lots of tiny fragments.

    Parameters
    ----------
    priority_function : function (g, [e]) -> [f]
        The merge priority function to convert.
    threshold : int or float
        The minimum size to be considered for merging.
    strictness : int in {1, 2, 3}
        How hard to check for segment size:
          - 1: only merge small nodes that are not at the volume boundary.
          - 2: only merge small nodes not at the volume boundary, *but not
               to each other.*
          - 3: conditions 1 and 2 but also ensure that the boundary shared
               between segments is bigger than 2 voxels.

    Returns
    -------
    ladder_priority_function : function (g, [e]) -> [f]
        Same as priority function but only for small segments, otherwise
        returns infinity.
    """
    def ladder_function(g, edges):
        edges = np.array(edges)
        pass_ladder = np.empty(len(edges), dtype=bool)
        for i, (n1, n2) in enumerate(edges):
            s1 = g.nodes[n1]['size']
            s2 = g.nodes[n2]['size']
            ladder_condition = \
                    (s1 < threshold and not g.at_volume_boundary(n1)) or \
                    (s2 < threshold and not g.at_volume_boundary(n2))
            if strictness >= 2:
                ladder_condition &= ((s1 < threshold) != (s2 < threshold))
            if strictness >= 3:
                ladder_condition &= len(g.boundary(n1, n2)) > 2
            pass_ladder[i] = ladder_condition
        priority = np.empty(len(edges), dtype=float)
        priority[pass_ladder] = priority_function(g, edges[pass_ladder])
        priority[~pass_ladder] = np.inf
        return priority
    return ladder_function


def no_mito_merge(priority_function):
    """Convert priority function to avoid merging mitochondria.

    Mitochondria are super annoying in segmentation. This uses pre-
    -computed mitochondrion labels for the segments to avoid merging
    anything that looks like a mitochondrion, in the beginning. These
    can be dealt with later when the bulk of the segmentation is
    correct.

    Parameters
    ----------
    priority_function : function (g, [e]) -> [f]
        The merge priority function to convert.

    Returns
    -------
    mito_priority_function : function (g, [e]) -> [f]
        Same as priority function, but avoids merging frozen nodes/edges.
        Freezing can be defined using any property, not just mitochondria!

    See Also
    --------
    mito_merge
    """
    def predict(g, edges):
        priorities = priority_function(g, edges)
        for i, (n1, n2) in enumerate(edges):
            frozen = (n1 in g.frozen_nodes or
                      n2 in g.frozen_nodes or
                      (n1, n2) in g.frozen_edges)
            if frozen:
                priorities[i] = np.inf
        return priorities
    return predict


@batchify
def mito_merge(g, n1, n2):
    """Simple priority funct to merge segments previously labeled as mito."""
    if n1 in g.frozen_nodes and n2 in g.frozen_nodes:
        return np.inf
    elif (n1, n2) in g.frozen_edges:
        return np.inf
    elif n1 not in g.frozen_nodes and n2 not in g.frozen_nodes:
        return np.inf
    else:
        if n1 in g.frozen_nodes:
            mito = n1
            cyto = n2
        else:
            mito = n2
            cyto = n1
        if g.nodes[mito]['size'] > g.nodes[cyto]['size']:
            return np.inf
        else:
            return 1.0 - (float(len(g.boundary(mito, cyto)))/
            sum([len(g.boundary(mito, x)) for x in g.neighbors(mito)]))


def classifier_probability(feature_map, classifier):
    def predict(g, edges):
        edges = np.atleast_2d(edges)
        boundary = np.sum(edges == g.boundary_body, axis=1).astype(bool)
        result = np.empty(len(edges))
        result[boundary] = np.inf
        features = np.atleast_2d([feature_map(g, n1, n2)
                                  for n1, n2 in edges[~boundary]])
        if features.size > 0:
            prediction = classifier.predict_proba(features)[:, 1]
        else:
            prediction = np.array([])
        result[~boundary] = prediction
        return result
    return predict


def ordered_priority(edges):
    d = {}
    n = len(edges)
    for i, (n1, n2) in enumerate(edges):
        score = float(i)/n
        d[(n1, n2)] = score
        d[(n2, n1)] = score

    def ord(g, edges):
        return [d.get(e, inf) for e in edges]
    return ord


def expected_change_vi(feature_map, classifier, alpha=1.0, beta=1.0):
    prob_func = classifier_probability(feature_map, classifier)
    def predict(g, edges):
        p = prob_func(g, edges)  # Prediction from the classifier
        # Calculate change in VI if n1 and n2 should not be merged
        n1_sizes = np.fromiter((g.node[n1]['size'] for n1, n2 in edges),
                               dtype=float, count=len(edges))
        n2_sizes = np.fromiter((g.node[n2]['size'] for n1, n2 in edges),
                               dtype=float, count=len(edges))
        v = compute_local_vi_change(n1_sizes, n2_sizes, g.volume_size)
        # Return expected change
        return  p*alpha*v - (1-p)*beta*v
    return predict


def compute_local_vi_change(s1, s2, n):
    """Compute change in VI if we merge disjoint sizes s1,s2 in a volume n."""
    py1 = s1 / n
    py2 = s2 / n
    py = py1 + py2
    return -(py1*np.log2(py1) + py2*np.log2(py2) - py*np.log2(py))


def compute_true_delta_vi(ctable, n1, n2):
    p1 = ctable[n1].sum()
    p2 = ctable[n2].sum()
    p3 = p1+p2
    p1g_log_p1g = xlogx(ctable[n1]).sum()
    p2g_log_p2g = xlogx(ctable[n2]).sum()
    p3g_log_p3g = xlogx(ctable[n1]+ctable[n2]).sum()
    return p3*np.log2(p3) - p1*np.log2(p1) - p2*np.log2(p2) - \
                                2*(p3g_log_p3g - p1g_log_p1g - p2g_log_p2g)


def expected_change_rand(feature_map, classifier, alpha=1.0, beta=1.0):
    prob_func = classifier_probability(feature_map, classifier)
    def predict(g, edges):
        p = prob_func(g, edges) # Prediction from the classifier
        n1_sizes = np.fromiter((g.node[n1]['size'] for n1, n2 in edges),
                               dtype=float, count=len(edges))
        n2_sizes = np.fromiter((g.node[n2]['size'] for n1, n2 in edges),
                               dtype=float, count=len(edges))
        v = compute_local_rand_change(n1_sizes, n2_sizes, g.volume_size)
        return p*v*alpha - (1-p)*beta*v
    return predict


def compute_local_rand_change(s1, s2, n):
    """Compute change in rand if we merge disjoint sizes s1,s2 in volume n."""
    return s1 * s2 / nchoosek(n, 2)


def compute_true_delta_rand(ctable, n1, n2, n):
    """Compute change in RI obtained by merging rows n1 and n2.

    This function assumes ctable is normalized to sum to 1.
    """
    localct = n * ctable[(n1, n2), :]
    total = localct.data.sum()
    sqtotal = (localct.data ** 2).sum()
    delta_sxy = 1. / 2 * ((np.array(localct.sum(axis=0)) ** 2).sum() -
                          sqtotal)
    delta_sx = 1. / 2 * (total ** 2 -
                         (np.array(localct.sum(axis=1)) ** 2).sum())
    return (2 * delta_sxy - delta_sx) / nchoosek(n, 2)


def boundary_mean_ladder(g, edges, threshold, strictness=1):
    f = make_ladder(boundary_mean, threshold, strictness)
    return f(g, edges)


def boundary_mean_plus_sem(g, edges, alpha=-6):
    bvals = [g.probabilities_r[g.boundary(n1, n2)] for n1, n2 in edges]
    means = np.fromiter(map(mean, bvals), dtype=float, count=len(edges))
    sems = np.fromiter(map(sem, bvals), dtype=float, count=len(edges))
    return means + alpha*sems


def random_priority(g, edges):
    edges = np.atleast_2d(edges)
    result = np.random.rand(len(edges))
    result[np.sum(edges == g.boundary_body, axis=1).astype(bool)] = np.inf
    return result


class Rag(Graph):
    """Region adjacency graph for segmentation of nD volumes.

    Parameters
    ----------
    watershed : array of int, shape (M, N, ..., P)
        The labeled regions of the image. Note: this is called
        `watershed` for historical reasons, but could refer to a
        superpixel map of any origin.
    probabilities : array of float, shape (M, N, ..., P[, Q])
        The probability of each pixel of belonging to a particular
        class. Typically, this has the same shape as `watershed`
        and represents the probability that the pixel is part of a
        region boundary, but it can also have an additional
        dimension for probabilities of belonging to other classes,
        such as mitochondria (in biological images) or specific
        textures (in natural images).
    merge_priority_function : callable function, optional
        This function must take exactly three arguments as input
        (a Rag object and two node IDs) and return a single float.
    feature_manager : ``features.base.Null`` object, optional
        A feature manager object that controls feature computation
        and feature caching.
    mask : array of bool, shape (M, N, ..., P)
        A mask of the same shape as `watershed`, `True` in the
        positions to be processed when making a RAG, `False` in the
        positions to ignore.
    show_progress : bool, optional
        Whether to display an ASCII progress bar during long-
        -running graph operations.
    connectivity : int in {1, ..., `watershed.ndim`}
        When determining adjacency, allow neighbors along
        `connectivity` dimensions.
    channel_is_oriented : array-like of bool, shape (Q,), optional
        For multi-channel images, some channels, for example some
        edge detectors, have a specific orientation. In conjunction
        with the `orientation_map` argument, specify which channels
        have an orientation associated with them.
    orientation_map : array-like of float, shape (Q,)
        Specify the orientation of the corresponding channel. (2D
        images only)
    normalize_probabilities : bool, optional
        Divide the input `probabilities` by their maximum to ensure
        a range in [0, 1].
    exclusions : array-like of int, shape (M, N, ..., P), optional
        Volume of same shape as `watershed`. Mark points in the
        volume with the same label (>0) to prevent them from being
        merged during agglomeration. For example, if
        `exclusions[45, 92] == exclusions[51, 105] == 1`, then
        segments `watershed[45, 92]` and `watershed[51, 105]` will
        never be merged, regardless of the merge priority function.
    isfrozennode : function, optional
        Function taking in a Rag object and a node id and returning
        a bool. If the function returns ``True``, the node will not
        be merged, regardless of the merge priority function.
    isfrozenedge : function, optional
        As `isfrozennode`, but the function should take the graph
        and *two* nodes, to specify an edge that cannot be merged.
    use_slow : bool, optional
        Use the slow Python machinery to build the RAG edges. This is
        mainly for historical reasons, and for applications that
        require pixel-perfect RAGs. (The fast version depends on
        morphological operations that lose information when a single
        pixel has more than one unique neighbor.)
    update_unchanged_edges : bool, optional
        If True, recompute merge probabilities when merging two nodes,
        even though only the node information has changed. This option
        is present for historical reasons, but should make little
        difference to merge probabilites, at least when strong edge
        features are used.
    """

    def __init__(self, watershed=array([], label_dtype),
                 probabilities=array([]),
                 merge_priority_function=boundary_mean, gt_vol=None,
                 feature_manager=gala.features.base.Null(), mask=None,
                 show_progress=False, connectivity=1,
                 channel_is_oriented=None, orientation_map=array([]),
                 normalize_probabilities=False, exclusions=array([]),
                 isfrozennode=None, isfrozenedge=None, use_slow=False,
                 update_unchanged_edges=False):

        super(Rag, self).__init__(weighted=False)
        self.show_progress = show_progress
        self.connectivity = connectivity
        self.pbar = (ip.StandardProgressBar() if self.show_progress
                     else ip.NoProgressBar())
        self.set_watershed(watershed, connectivity)
        self.set_probabilities(probabilities, normalize_probabilities)
        self.set_orientations(orientation_map, channel_is_oriented)
        self.merge_priority_function = merge_priority_function
        self.max_merge_score = -inf
        if mask is None:
            self.mask = np.broadcast_to([True], self.watershed_r.shape)
            self.is_masked = False
        else:
            self.mask = morpho.pad(mask, True).ravel()
            self.is_masked = True
        self.use_slow = use_slow
        self.build_graph_from_watershed()
        self.set_feature_manager(feature_manager)
        self.set_ground_truth(gt_vol)
        self.set_exclusions(exclusions)
        self.merge_queue = MergeQueue()
        self.tree = tree.Ultrametric(self.nodes())
        self.frozen_nodes = set()
        if isfrozennode is not None:
            for node in self.nodes():
                if isfrozennode(self, node):
                    self.frozen_nodes.add(node)
        self.frozen_edges = set()
        if isfrozenedge is not None:
            for n1, n2 in self.edges():
                if isfrozenedge(self, n1, n2):
                    self.frozen_edges.add((n1,n2))
        self.update_unchanged_edges = update_unchanged_edges
        if update_unchanged_edges:
            self.move_edge = self.merge_edge_properties
        self.fast_edges = not use_slow


    def __copy__(self):
        """Return a copy of the object and attributes.
        """
        mask = (
            morpho.juicy_center(self.mask.reshape(self.probabilities.shape))
            if self.is_masked
            else None
        )
        return Rag(
            watershed=morpho.juicy_center(self.watershed),
            probabilities=morpho.juicy_center(self.probabilities),
            merge_priority_function=self.merge_priority_function,
            gt_vol=morpho.juicy_center(self.gt) if self.gt else None,
            feature_manager=self.feature_manager,
            mask=mask,
            show_progress=self.show_progress,
            connectivity=self.connectivity,
            use_slow=self.use_slow,
            update_unchanged_edges=self.update_unchanged_edges,
        )


    def copy(self):
        """Return a copy of the object and attributes.
        """
        return self.__copy__()


    def extent(self, nodeid):
        try:
            ext = self.extents
            full_ext = [ext[f] for f in self.nodes[nodeid]['fragments']]
            return np.concatenate(full_ext).astype(np.intp)
        except AttributeError:
            extent_array = opt.flood_fill(self.watershed,
                               np.array(self.nodes[nodeid]['entrypoint']),
                               np.fromiter(self.nodes[nodeid]['fragments'],
                                           dtype=int))
            if len(extent_array) != self.nodes[nodeid]['size']:
                sys.stderr.write('Flood fill fail - found %d voxels but size'
                                 'expected %d\n' % (len(extent_array),
                                                    self.nodes[nodeid]['size']))
            raveled_indices = np.ravel_multi_index(extent_array.T,
                                                   self.watershed.shape)
            return set(raveled_indices)

    def boundary(self, u, v):
        edge_dict = self.edges[u, v]
        try:
            boundary_ids = edge_dict['boundary-ids']
        except KeyError:  # RAG built using old, slow method
            bound = edge_dict['boundary']
        else:
            all_bounds = [self.boundaries[i] for i in boundary_ids]
            bound = np.concatenate(all_bounds).astype(np.intp)
        return bound


    def real_edges(self, *args, **kwargs):
        """Return edges internal to the volume.

        The RAG actually includes edges to a "virtual" region that
        envelops the entire volume. This function returns the list of
        edges that are internal to the volume.

        Parameters
        ----------
        *args, **kwargs : arbitrary types
            Arguments and keyword arguments are passed through to the
            ``edges()`` function of the ``networkx.Graph`` class.

        Returns
        -------
        edge_list : list of tuples
            A list of pairs of node IDs, which are typically integers.

        See Also
        --------
        real_edges_iter, networkx.Graph.edges
        """
        return [e for e in super(Rag, self).edges(*args, **kwargs) if
                                            self.boundary_body not in e[:2]]

    def real_edges_iter(self, *args, **kwargs):
        """Return iterator of edges internal to the volume.

        The RAG actually includes edges to a "virtual" region that
        envelops the entire volume. This function returns the list of
        edges that are internal to the volume.

        Parameters
        ----------
        *args, **kwargs : arbitrary types
            Arguments and keyword arguments are passed through to the
            ``edges()`` function of the ``networkx.Graph`` class.

        Returns
        -------
        edges_iter : iterator of tuples
            An iterator over pairs of node IDs, which are typically
            integers.
        """
        return (e for e in super(Rag, self).edges_iter(*args, **kwargs) if
                                            self.boundary_body not in e[:2])

    def build_graph_from_watershed(self, idxs=None):
        """Build the graph object from the region labels.

        The region labels should have been set ahead of time using
        ``set_watershed()``.

        Parameters
        ----------
        idxs : array-like of int, optional
            Linear indices into raveled volume array. If provided, the
            graph is built only for these indices.
        """
        if self.watershed.size == 0:
            return # stop processing for empty graphs
        idxs_is_none = idxs is None
        if idxs_is_none:
            idxs = arange(self.watershed.size, dtype=self.steps.dtype)
        if self.is_masked:
            idxs = idxs[self.mask[idxs]]  # use only masked idxs
        self.build_nodes(idxs)
        if self.is_masked or not idxs_is_none or self.use_slow:
            self.fast_edges = False
            inner_idxs = idxs[self.watershed_r[idxs] != self.boundary_body]
            self.build_edges_slow(inner_idxs)
        else:
            self.fast_edges = True
            self.build_edges_fast()

    def build_nodes(self, idxs):
        self.add_node(self.boundary_body)
        labels = np.unique(self.watershed_r[idxs])
        sizes = np.bincount(self.watershed_r)
        if not hasattr(self, 'extents'):
            self.extents = lol.SparseLOL(lol.extents(self.watershed))
        for nodeid in labels:
            self.add_node(nodeid)
            node = self.nodes[nodeid]
            node['size'] = sizes[nodeid]
            node['fragments'] = {nodeid}  # set literal
            node['entrypoint'] = (
                np.array(np.unravel_index(self.extent(nodeid)[0],
                                          self.watershed.shape)))

    def build_edges_slow(self, idxs):
        if self.show_progress:
            idxs = ip.with_progress(idxs, title='Graph ', pbar=self.pbar)
        for idx in idxs:
            nodeid = self.watershed_r[idx]
            ns = idx + self.steps
            ns = ns[self.mask[ns]]
            adj = self.watershed_r[ns]
            adj = set(adj)
            for v in adj:
                if v == nodeid:
                    continue
                if self.has_edge(nodeid, v):
                    self.edges[nodeid, v]['boundary'].append(idx)
                else:
                    self.add_edge(nodeid, v, boundary=[idx])

    def build_edges_fast(self):
        """Build the graph edges using agglo2's sparse graph functions.
        """
        edges_coo = agglo2.edge_matrix(self.watershed, self.connectivity)
        edge_map, self.boundaries = agglo2.sparse_boundaries(edges_coo)
        coo = edge_map.tocoo()
        edges_iter = ((i, j, {'boundary-ids': {edge_map[i, j]}})
                      for i in range(edge_map.shape[0])
                      for j in edge_map.indices[edge_map.indptr[i]:
                                                edge_map.indptr[i+1]])
        self.add_edges_from(edges_iter)

    def set_feature_manager(self, feature_manager):
        """Set the feature manager and ensure feature caches are computed.

        Parameters
        ----------
        feature_manager : ``features.base.Null`` object
            The feature manager to be used by this RAG.

        Returns
        -------
        None
        """
        self.feature_manager = feature_manager
        self.compute_feature_caches()


    def compute_feature_caches(self):
        """Use the feature manager to compute node and edge feature caches.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for n in ip.with_progress(
                    self.nodes(), title='Node caches ', pbar=self.pbar):
            self.nodes[n]['feature-cache'] = \
                            self.feature_manager.create_node_cache(self, n)
        for n1, n2 in ip.with_progress(
                    self.edges(), title='Edge caches ', pbar=self.pbar):
            self.edges[n1, n2]['feature-cache'] = \
                            self.feature_manager.create_edge_cache(self, n1, n2)


    def set_probabilities(self, probs=array([]), normalize=False):
        """Set the `probabilities` attributes of the RAG.

        For various reasons, including removing the need for bounds
        checking when looking for neighboring pixels, the volume of
        pixel-level probabilities is padded on all faces. In addition,
        this function adds an attribute `probabilities_r`, a raveled
        view of the padded probabilities array for quick access to
        individual voxels using linear indices.

        Parameters
        ----------
        probs : array
            The input probabilities array.
        normalize : bool, optional
            If ``True``, the values in the array are scaled to be in
            [0, 1].

        Returns
        -------
        None
        """
        if len(probs) == 0:
            self.probabilities = zeros_like(self.watershed)
            self.probabilities_r = self.probabilities.ravel()
        probs = probs.astype('float')
        if normalize and len(probs) > 1:
            probs -= probs.min() # ensure probs.min() == 0
            probs /= probs.max() # ensure probs.max() == 1
        sp = probs.shape
        sw = tuple(array(self.watershed.shape, dtype=int)-\
                    2*self.pad_thickness*ones(self.watershed.ndim, dtype=int))
        p_ndim = probs.ndim
        w_ndim = self.watershed.ndim
        padding = [inf]+(self.pad_thickness-1)*[0]
        if p_ndim == w_ndim:
            self.probabilities = morpho.pad(probs, padding)
            self.probabilities_r = self.probabilities.ravel()[:,newaxis]
        elif p_ndim == w_ndim+1:
            axes = list(range(p_ndim-1))
            self.probabilities = morpho.pad(probs, padding, axes)
            self.probabilities_r = self.probabilities.reshape(
                                                (self.watershed.size, -1))


    def set_orientations(self, orientation_map, channel_is_oriented):
        """Set the orientation map of the probability image.

        Parameters
        ----------
        orientation_map : array of float
            A map of angles of the same shape as the superpixel map.
        channel_is_oriented : 1D array-like of bool
            A vector having length the number of channels in the
            probability map.

        Returns
        -------
        None
        """
        if len(orientation_map) == 0:
            self.orientation_map = zeros_like(self.watershed)
            self.orientation_map_r = self.orientation_map.ravel()
        padding = [0]+(self.pad_thickness-1)*[0]
        self.orientation_map = morpho.pad(orientation_map, padding).astype(int)
        self.orientation_map_r = self.orientation_map.ravel()
        if channel_is_oriented is None:
            nchannels = 1 if self.probabilities.ndim==self.watershed.ndim \
                else self.probabilities.shape[-1]
            self.channel_is_oriented = array([False]*nchannels)
            self.max_probabilities_r = zeros_like(self.probabilities_r)
            self.oriented_probabilities_r = zeros_like(self.probabilities_r)
            self.non_oriented_probabilities_r = self.probabilities_r
        else:
            self.channel_is_oriented = channel_is_oriented
            self.max_probabilities_r = \
                self.probabilities_r[:, self.channel_is_oriented].max(axis=1)
            self.oriented_probabilities_r = \
                self.probabilities_r[:, self.channel_is_oriented]
            self.oriented_probabilities_r = \
                self.oriented_probabilities_r[
                    list(range(len(self.oriented_probabilities_r))),
                    self.orientation_map_r]
            self.non_oriented_probabilities_r = \
                self.probabilities_r[:, ~self.channel_is_oriented]


    def set_watershed(self, ws=array([], label_dtype), connectivity=1):
        """Set the initial segmentation volume (watershed).

        The initial segmentation is called `watershed` for historical
        reasons only.

        Parameters
        ----------
        ws : array of int
            The initial segmentation.
        connectivity : int in {1, ..., `ws.ndim`}, optional
            The pixel neighborhood.

        Returns
        -------
        None
        """
        ws = ws.astype(label_dtype)
        try:
            self.boundary_body = np.max(ws) + 1
        except ValueError: # empty watershed given
            self.boundary_body = 1
        self.volume_size = ws.size
        if ws.size > 0:
            ws, fw, inv = relabel_sequential(ws)
            self.inverse_watershed_map = inv  # translates to original labels
            self.forward_map = fw
        self.watershed = morpho.pad(ws, self.boundary_body)
        self.watershed_r = self.watershed.ravel()
        self.pad_thickness = 1
        self.steps = morpho.raveled_steps_to_neighbors(self.watershed.shape,
                                                       connectivity)

    def __contains__(self, value):
        new_value = self.forward_map(np.asarray(value))
        return super().__contains__(new_value)


    def set_ground_truth(self, gt=None):
        """Set the ground truth volume.

        This is useful for tracking segmentation accuracy over time.

        Parameters
        ----------
        gt : array of int
            A ground truth segmentation of the same volume passed to
            ``set_watershed``.

        Returns
        -------
        None
        """
        if gt is not None:
            gtm = gt.max()+1
            gt_ignore = [0, gtm] if (gt==0).any() else [gtm]
            seg_ignore = [0, self.boundary_body] if \
                        (self.watershed==0).any() else [self.boundary_body]
            self.gt = morpho.pad(gt, gtm)
            self.rig = merge_contingency_table(self.watershed, self.gt,
                                               ignore_seg=seg_ignore,
                                               ignore_gt=gt_ignore)
        else:
            self.gt = None
            # null pattern to transparently allow merging of nodes.
            # Bonus feature: counts how many sp's went into a single node.
            try:
                self.rig = ones(2 * self.watershed.max() + 1)
            except ValueError:
                self.rig = ones(2 * self.number_of_nodes() + 1)


    def set_exclusions(self, excl):
        """Set an exclusion volume, forbidding certain merges.

        Parameters
        ----------
        excl : array of int
            Exclusions work as follows: the volume `excl` is the same
            shape as the initial segmentation (see ``set_watershed``),
            and consists of mostly 0s. Any voxels with *the same*
            non-zero label will not be allowed to merge during
            agglomeration (provided they were not merged in the initial
            segmentation).

            This allows manual separation *a priori* of difficult-to-
            -segment regions.

        Returns
        -------
        None
        """
        if excl.size != 0:
            excl = morpho.pad(excl, [0] * self.pad_thickness)
        for n in self.nodes():
            if excl.size != 0:
                eids = unique(excl.ravel()[self.extent(n)])
                eids = eids[flatnonzero(eids)]
                self.nodes[n]['exclusions'] = set(list(eids))
            else:
                self.nodes[n]['exclusions'] = set()


    def build_merge_queue(self):
        """Build a queue of node pairs to be merged in a specific priority.

        Returns
        -------
        mq : MergeQueue object
            A MergeQueue is a Python ``deque`` with a specific element
            structure: a list of length 4 containing:
                 - the merge priority (any ordered type)
                 - a 'valid' flag
                 - and the two nodes in arbitrary order
            The valid flag allows one to "remove" elements from the
            queue in O(1) time by setting the flag to ``False``. Then,
            one checks the flag when popping elements and ignores those
            marked as invalid.

            One other specific feature is that there are back-links from
            edges to their corresponding queue items so that when nodes
            are merged, affected edges can be invalidated and reinserted
            in the queue with a new priority.
        """
        edges = self.real_edges()
        if edges:
            weights = self.merge_priority_function(self, edges)
        else:
            weights = []
        queue_items = []
        for w, (l1, l2) in zip(weights, edges):
            qitem = [w, True, l1, l2]
            queue_items.append(qitem)
            self.edges[l1, l2]['qlink'] = qitem
            self.edges[l1, l2]['weight'] = w
        return MergeQueue(queue_items, with_progress=self.show_progress)


    def rebuild_merge_queue(self):
        """Build a merge queue from scratch and assign to self.merge_queue.

        See Also
        --------
        build_merge_queue
        """
        self.merge_queue = self.build_merge_queue()


    def agglomerate(self, threshold=0.5, save_history=False):
        """Merge nodes hierarchically until given edge confidence threshold.

        This is the main workhorse of the ``agglo`` module!

        Parameters
        ----------
        threshold : float, optional
            The edge priority at which to stop merging.
        save_history : bool, optional
            Whether to save and return a history of all the merges made.

        Returns
        -------
        history : list of tuple of int, optional
            The ordered history of node pairs merged.
        scores : list of float, optional
            The list of merge scores corresponding to the `history`.
        evaluation : list of tuple, optional
            The split VI after each merge. This is only meaningful if
            a ground truth volume was provided at build time.

        Notes
        -----
            This function returns ``None`` when `save_history` is
            ``False``.
        """
        if self.merge_queue.is_empty():
            self.merge_queue = self.build_merge_queue()
        history, scores, evaluation = [], [], []
        # total merges is number of nodes minus boundary_node minus one.
        progress = tqdm(total=self.number_of_nodes() - 2)
        while len(self.merge_queue) > 0 and \
                                        self.merge_queue.peek()[0] < threshold:
            merge_priority, _, n1, n2 = self.merge_queue.pop()
            self.update_frozen_sets(n1, n2)
            self.merge_nodes(n1, n2, merge_priority)
            if save_history:
                history.append((n1,n2))
                scores.append(merge_priority)
                evaluation.append(
                    (self.number_of_nodes()-1, self.split_vi())
                )
            progress.update(1)
        progress.close()
        if save_history:
            return history, scores, evaluation


    def agglomerate_count(self, stepsize=100, save_history=False):
        """Agglomerate until 'stepsize' merges have been made.

        This function is like ``agglomerate``, but rather than to a
        certain threshold, a certain number of merges are made,
        regardless of threshold.

        Parameters
        ----------
        stepsize : int, optional
            The number of merges to make.
        save_history : bool, optional
            Whether to save and return a history of all the merges made.

        Returns
        -------
        history : list of tuple of int, optional
            The ordered history of node pairs merged.
        scores : list of float, optional
            The list of merge scores corresponding to the `history`.
        evaluation : list of tuple, optional
            The split VI after each merge. This is only meaningful if
            a ground truth volume was provided at build time.

        Notes
        -----
            This function returns ``None`` when `save_history` is
            ``False``.

        See Also
        --------
        agglomerate
        """
        if self.merge_queue.is_empty():
            self.merge_queue = self.build_merge_queue()
        history, evaluation = [], []
        i = 0
        for i in range(stepsize):
            if len(self.merge_queue) == 0:
                break
            merge_priority, _, n1, n2 = self.merge_queue.pop()
            i += 1
            self.merge_nodes(n1, n2, merge_priority)
            if save_history:
                history.append((n1, n2))
                evaluation.append(
                    (self.number_of_nodes()-1, self.split_vi())
                )
        if save_history:
            return history, evaluation


    def agglomerate_ladder(self, min_size=1000, strictness=2):
        """Merge sequentially all nodes smaller than `min_size`.

        Parameters
        ----------
        min_size : int, optional
            The smallest allowable segment after ladder completion.
        strictness : {1, 2, 3}, optional
            `strictness == 1`: all nodes smaller than `min_size` are
            merged according to the merge priority function.
            `strictness == 2`: in addition to `1`, small nodes can only
            be merged to big nodes.
            `strictness == 3`: in addition to `2`, nodes sharing less
            than one pixel of boundary are not agglomerated.

        Returns
        -------
        None

        Notes
        -----
        Nodes that are on the volume boundary are not agglomerated.
        """
        original_merge_priority_function = self.merge_priority_function
        self.merge_priority_function = make_ladder(
            self.merge_priority_function, min_size, strictness
        )
        self.rebuild_merge_queue()
        self.agglomerate(inf)
        self.merge_priority_function = original_merge_priority_function
        self.merge_queue.finish()
        self.rebuild_merge_queue()
        max_score = max([qitem[0] for qitem in self.merge_queue.q])
        for n in self.tree.nodes():
            self.tree.nodes[n]['w'] -= max_score


    def learn_agglomerate(self, gts, feature_map,
                          min_num_samples=1,
                          learn_flat=True,
                          learning_mode='strict',
                          labeling_mode='assignment',
                          priority_mode='active',
                          memory=True,
                          unique=True,
                          random_state=None,
                          max_num_epochs=10,
                          min_num_epochs=2,
                          max_num_samples=np.inf,
                          classifier='random forest',
                          active_function=classifier_probability,
                          mpf=boundary_mean):
        """Agglomerate while comparing to ground truth & classifying merges.

        Parameters
        ----------
        gts : array of int or list thereof
            The ground truth volume(s) corresponding to the current
            probability map.
        feature_map : function (Rag, node, node) -> array of float
            The map from node pairs to a feature vector. This must
            consist either of uncached features or of the cache used
            when building the graph.
        min_num_samples : int, optional
            Continue training until this many training examples have
            been collected.
        learn_flat : bool, optional
            Do a flat learning on the static graph with no
            agglomeration.
        learning_mode : {'strict', 'loose'}, optional
            In 'strict' mode, if a "don't merge" edge is encountered,
            it is added to the training set but the merge is not
            executed. In 'loose' mode, the merge is allowed to proceed.
        labeling_mode : {'assignment', 'vi-sign', 'rand-sign'}, optional
            How to decide whether two nodes should be merged based on
            the ground truth segmentations. ``'assignment'`` means the
            nodes are assigned to the ground truth node with which they
            share the highest overlap. ``'vi-sign'`` means the the VI
            change of the switch is used (negative is better).
            ``'rand-sign'`` means the change in Rand index is used
            (positive is better).
        priority_mode : string, optional
            One of:
                ``'active'``: Train a priority function with the data
                              from previous epochs to obtain the next.
                ``'random'``: Merge edges at random.
                ``'mixed'``: Alternate between epochs of ``'active'``
                             and ``'random'``.
                ``'mean'``: Use the mean boundary value. (In this case,
                            training is limited to 1 or 2 epochs.)
                ``'custom'``: Use the function provided by `mpf`.
        memory : bool, optional
            Keep the training data from all epochs (rather than just
            the most recent one).
        unique : bool, optional
            Remove duplicate feature vectors.
        random_state : int, optional
            If provided, this parameter is passed to `get_classifier`
            to set the random state and allow consistent results across
            tests.
        max_num_epochs : int, optional
            Do not train for longer than this (this argument *may*
            override the `min_num_samples` argument).
        min_num_epochs : int, optional
            Train for no fewer than this number of epochs.
        max_num_samples : int, optional
            Train for no more than this number of samples.
        classifier : string, optional
            Any valid classifier descriptor. See
            ``gala.classify.get_classifier()``
        active_function : function (feat. map, classifier) -> function, optional
            Use this to create the next priority function after an
            epoch.
        mpf : function (Rag, node, node) -> float
            A merge priority function to use when ``priority_mode`` is
            ``'custom'``.

        Returns
        -------
        data : list of array
            Four arrays containing:
                - the feature vectors, shape ``(n_samples, n_features)``.
                - the labels, shape ``(n_samples, 3)``. A value of `-1`
                  means "should merge", while `1` means "should
                  not merge". The columns correspond to the three
                  labeling methods: assignment, VI sign, or RI sign.
                - the VI and RI change of each merge, ``(n_edges, 2)``.
                - the list of merged edges ``(n_edges, 2)``.
        alldata : list of list of array
            A list of lists like `data` above: one list for each epoch.

        Notes
        -----
        The gala algorithm [1] uses the default parameters. For the
        LASH algorithm [2], use:
            - `learning_mode`: ``'loose'``
            - `labeling_mode`: ``'rand-sign'``
            - `memory`: ``False``

        References
        ----------
        .. [1] Nunez-Iglesias et al, Machine learning of hierarchical
               clustering to segment 2D and 3D images, PLOS ONE, 2013.
        .. [2] Jain et al, Learning to agglomerate superpixel
               hierarchies, NIPS, 2011.

        See Also
        --------
        Rag
        """
        learning_mode = learning_mode.lower()
        labeling_mode = labeling_mode.lower()
        priority_mode = priority_mode.lower()
        if priority_mode == 'mean' and unique:
            max_num_epochs = 2 if learn_flat else 1
        if priority_mode in ['random', 'mean'] and not memory:
            max_num_epochs = 1
        label_type_keys = {'assignment':0, 'vi-sign':1, 'rand-sign':2}
        if type(gts) != list:
            gts = [gts] # allow using single ground truth as input
        master_ctables = [merge_contingency_table(self.get_segmentation(), gt)
                          for gt in gts]
        alldata = []
        data = [[],[],[],[]]
        for num_epochs in range(max_num_epochs):
            ctables = deepcopy(master_ctables)
            if len(data[0]) > min_num_samples and num_epochs >= min_num_epochs:
                break
            if learn_flat and num_epochs == 0:
                alldata.append(self.learn_flat(gts, feature_map))
                data = unique_learning_data_elements(alldata) if memory \
                    else alldata[-1]
                continue
            g = self.copy()
            if priority_mode == 'mean':
                g.merge_priority_function = boundary_mean
            elif num_epochs > 0 and priority_mode == 'active' or \
                num_epochs % 2 == 1 and priority_mode == 'mixed':
                cl = get_classifier(classifier, random_state=random_state)
                feat, lab = classify.sample_training_data(
                    data[0], data[1][:, label_type_keys[labeling_mode]],
                    max_num_samples)
                cl = cl.fit(feat, lab)
                g.merge_priority_function = active_function(feature_map, cl)
            elif priority_mode == 'random' or \
                (priority_mode == 'active' and num_epochs == 0):
                g.merge_priority_function = random_priority
            elif priority_mode == 'custom':
                g.merge_priority_function = mpf
            g.show_progress = False # bug in MergeQueue usage causes
                                    # progressbar crash.
            g.rebuild_merge_queue()
            alldata.append(g.learn_epoch(ctables, feature_map,
                                         learning_mode=learning_mode,
                                         labeling_mode=labeling_mode))
            if memory:
                if unique:
                    data = unique_learning_data_elements(alldata)
                else:
                    data = concatenate_data_elements(alldata)
            else:
                data = alldata[-1]
            logging.debug('data size %d at epoch %d'%(len(data[0]), num_epochs))
        return data, alldata


    def learn_flat(self, gts, feature_map):
        """Learn all edges on the graph, but don't agglomerate.

        Parameters
        ----------
        gts : array of int or list thereof
            The ground truth volume(s) corresponding to the current
            probability map.
        feature_map : function (Rag, node, node) -> array of float
            The map from node pairs to a feature vector. This must
            consist either of uncached features or of the cache used
            when building the graph.

        Returns
        -------
        data : list of array
            Four arrays containing:
                - the feature vectors, shape ``(n_samples, n_features)``.
                - the labels, shape ``(n_samples, 3)``. A value of `-1`
                  means "should merge", while `1` means "should
                  not merge". The columns correspond to the three
                  labeling methods: assignment, VI sign, or RI sign.
                - the VI and RI change of each merge, ``(n_edges, 2)``.
                - the list of merged edges ``(n_edges, 2)``.

        See Also
        --------
        learn_agglomerate
        """
        if type(gts) != list:
            gts = [gts] # allow using single ground truth as input
        ctables = [merge_contingency_table(self.get_segmentation(), gt)
                   for gt in gts]
        assignments = [ev.assignment_table(ct) for ct in ctables]
        return list(map(array, zip(*[
                self.learn_edge(e, ctables, assignments, feature_map)
                for e in self.real_edges()])))


    def learn_edge(self, edge, ctables, assignments, feature_map):
        """Determine whether an edge should be merged based on ground truth.

        Parameters
        ----------
        edge : (int, int) tuple
            An edge in the graph.
        ctables : list of array
            A list of contingency tables determining overlap between the
            current segmentation and the ground truth.
        assignments : list of array
            Similar to the contingency tables, but each row is thresholded
            so each segment corresponds to exactly one ground truth segment.
        feature_map : function (Rag, node, node) -> array of float
            The map from node pairs to a feature vector.

        Returns
        -------
        features : 1D array of float
            The feature vector for that edge.
        labels : 1D array of float, length 3
            The labels determining whether the edge should be merged.
            A value of `-1` means "should merge", while `1` means "should
            not merge". The columns correspond to the three labeling
            methods: assignment, VI sign, or RI sign.
        weights : 1D array of float, length 2
            The VI and RI change of the merge.
        nodes : tuple of int
            The given edge.
        """
        n1, n2 = edge
        features = feature_map(self, n1, n2).ravel()
        # Calculate weights for weighting data points
        s1, s2 = [self.nodes[n]['size'] for n in [n1, n2]]
        weights = \
            compute_local_vi_change(s1, s2, self.volume_size), \
            compute_local_rand_change(s1, s2, self.volume_size)
        # Get the fraction of times that n1 and n2 assigned to
        # same segment in the ground truths
        cont_labels = [
            [(-1) ** (np.all(ev.nzcol(a, n1) == ev.nzcol(a, n2)))
             for a in assignments],
            [compute_true_delta_vi(ctable, n1, n2) for ctable in ctables],
            [-compute_true_delta_rand(ctable, n1, n2, self.volume_size)
                                                    for ctable in ctables]
        ]
        labels = [np.sign(mean(cont_label)) for cont_label in cont_labels]
        if any(map(isnan, labels)) or any([label == 0 for label in labels]):
            logging.debug('NaN or 0 labels found. ' +
                                    ' '.join(map(str, [labels, (n1, n2)])))
        labels = [1 if i==0 or isnan(i) or n1 in self.frozen_nodes or
            n2 in self.frozen_nodes or (n1, n2) in self.frozen_edges else
            i for i in labels]
        return features, labels, weights, (n1, n2)


    def learn_epoch(self, ctables, feature_map,
                    learning_mode='permissive', labeling_mode='assignment'):
        """Learn the agglomeration process using various strategies.

        Parameters
        ----------
        ctables : array of float or list thereof
            One or more contingency tables between own segments and gold
            standard segmentations
        feature_map : function (Rag, node, node) -> array of float
            The map from node pairs to a feature vector. This must
            consist either of uncached features or of the cache used
            when building the graph.
        learning_mode : {'strict', 'permissive'}, optional
            If ``'strict'``, don't proceed with a merge when it goes against
            the ground truth. For historical reasons, 'loose' is allowed as
            a synonym for 'strict'.
        labeling_mode : {'assignment', 'vi-sign', 'rand-sign'}, optional
            Which label to use for `learning_mode`. Note that all labels
            are saved in the end.

        Returns
        -------
        data : list of array
            Four arrays containing:
                - the feature vectors, shape ``(n_samples, n_features)``.
                - the labels, shape ``(n_samples, 3)``. A value of `-1`
                  means "should merge", while `1` means "should
                  not merge". The columns correspond to the three
                  labeling methods: assignment, VI sign, or RI sign.
                - the VI and RI change of each merge, ``(n_edges, 2)``.
                - the list of merged edges ``(n_edges, 2)``.
        """
        label_type_keys = {'assignment':0, 'vi-sign':1, 'rand-sign':2}
        assignments = [ev.csrRowExpandableCSR(asst)
                       for asst in map(ev.assignment_table, ctables)]
        g = self
        data = []
        while len(g.merge_queue) > 0:
            merge_priority, _, n1, n2 = g.merge_queue.pop()
            if g.boundary_body in (n1, n2):
                continue
            dat = g.learn_edge((n1,n2), ctables, assignments, feature_map)
            data.append(dat)
            label = dat[1][label_type_keys[labeling_mode]]
            if learning_mode != 'strict' or label < 0:
                node_id = g.merge_nodes(n1, n2, merge_priority)
                for ctable, assignment in zip(ctables, assignments):
                    ctable[node_id] = ctable[n1] + ctable[n2]
                    ctable[n1] = 0
                    ctable[n2] = 0
                    assignment[node_id] = (ctable[node_id] ==
                                           ctable[node_id].max())
                    assignment[n1] = 0
                    assignment[n2] = 0
        return list(map(array, zip(*data)))


    def replay_merge_history(self, merge_seq, labels=None, num_errors=1):
        """Agglomerate according to a merge sequence, optionally labeled.

        Parameters
        ----------
        merge_seq : iterable of pair of int
            The sequence of node IDs to be merged.
        labels : iterable of int in {-1, 0, 1}, optional
            A sequence matching `merge_seq` specifying whether a merge
            should take place or not. -1 or 0 mean "should merge", 1
            otherwise.

        Returns
        -------
        n : int
            Number of elements consumed from `merge_seq`
        e : (int, int)
            Last merge pair observed.

        Notes
        -----
        The merge sequence and labels *must* be generators if you don't want
        to manually keep track of how much has been consumed. The merging
        continues until `num_errors` false merges have been encountered, or
        until the sequence is fully consumed.
        """
        if labels is None:
            labels1 = it.repeat(False)
            labels2 = it.repeat(False)
        else:
            labels1 = (label > 0 for label in labels)
            labels2 = (label > 0 for label in labels)
        counter = it.count()
        errors_remaining = conditional_countdown(labels2, num_errors)
        nodes = None
        for nodes, label, errs, count in \
                        zip(merge_seq, labels1, errors_remaining, counter):
            n1, n2 = nodes
            if not label:
                self.merge_nodes(n1, n2)
            elif errs == 0:
                break
        return next(counter), nodes


    def rename_node(self, old, new):
        """Rename node `old` to `new`, updating edges and weights.

        Parameters
        ----------
        old : int
            The node being renamed.
        new : int
            The new node id.
        """
        self.add_node(new, **self.nodes[old])
        self.add_edges_from(
            [(new, v, self.edges[old, v]) for v in self.neighbors(old)])
        for v in self.neighbors(new):
            qitem = self.edges[new, v].get('qlink', None)
            if qitem is not None:
                if qitem[2] == old:
                    qitem[2] = new
                else:
                    qitem[3] = new
        self.remove_node(old)


    def merge_nodes(self, n1, n2, merge_priority=0.0):
        """Merge two nodes, while updating the necessary edges.

        Parameters
        ----------
        n1, n2 : int
            Nodes determining the edge for which to update the UCM.
        merge_priority : float, optional
            The merge priority of the merge.

        Returns
        -------
        node_id : int
            The id of the node resulting from the merge.

        Notes
        -----
        Additionally, the RIG (region intersection graph), the
        contingency matrix to the ground truth (if provided) is
        updated.
        """
        if len(self.nodes[n1]['exclusions']
                & self.nodes[n2]['exclusions']) > 0:
            return
        else:
            self.nodes[n1]['exclusions'].update(self.nodes[n2]['exclusions'])
        w = self.edges[n1, n2].get('weight', merge_priority)
        self.nodes[n1]['size'] += self.nodes[n2]['size']
        self.nodes[n1]['fragments'].update(self.nodes[n2]['fragments'])

        self.feature_manager.update_node_cache(self, n1, n2,
                self.nodes[n1]['feature-cache'], self.nodes[n2]['feature-cache'])
        nn1 = list(self.neighbors(n1))
        nn2 = list(self.neighbors(n2))
        common_neighbors = np.intersect1d(
                np.array(nn1), np.array(nn2), assume_unique=True
                )
        edges_to_update = []
        for n in common_neighbors:
            self.merge_edge_properties((n2, n), (n1, n))
            edges_to_update.append((n1, n))
        new_neighbors = np.setdiff1d(
                nn2,
                np.concatenate((common_neighbors, [n1])),
                assume_unique=True,
                )
        for n in new_neighbors:
            self.move_edge((n2, n), (n1, n))
            if self.update_unchanged_edges:
                edges_to_update.append((n1, n))
        try:
            self.merge_queue.invalidate(self.edges[n1, n2]['qlink'])
        except KeyError:  # no edge or no queue link
            pass
        self.update_merge_queue(edges_to_update)
        node_id = self.tree.merge(n1, n2, w)
        self.remove_node(n2)
        self.rename_node(n1, node_id)
        self.rig[node_id] = self.rig[n1] + self.rig[n2]
        self.rig[n1] = 0
        self.rig[n2] = 0
        return node_id

    def merge_subgraph(self, subgraph=None, source=None):
        """Merge a (typically) connected set of nodes together.

        Parameters
        ----------
        subgraph : agglo.Rag, networkx.Graph, or list of int (node id)
            A subgraph to merge.
        source : int (node id), optional
            Merge the subgraph to this node.
        """
        if type(subgraph) not in [Rag, Graph]: # input is node list
            subgraph = self.subgraph(subgraph)
        if len(subgraph) == 0:
            return
        for subsubgraph in (
                self.subgraph(list(c))
                for c in list(nx.connected_components(subgraph))
                ):
            node_dfs = list(dfs_preorder_nodes(subsubgraph, source))
            # dfs_preorder_nodes returns iter, convert to list
            source_node, other_nodes = node_dfs[0], node_dfs[1:]
            for current_node in other_nodes:
                source_node = self.merge_nodes(source_node, current_node)

    def split_node(self, u, n=2, **kwargs):
        """Use normalized cuts [1] to split a node/segment.

        Parameters
        ----------
        u : int (node id)
            Which node to split.
        n : int, optional
            How many segments to split it into.

        Returns
        -------
        None

        References
        ----------
        .. [1] Shi, J., and Malik, J. (2000). Normalized cuts and image
               segmentation. Pattern Analysis and Machine Intelligence.
        """
        node_extent = self.extent(u)
        labels = unique(self.watershed_r[node_extent])
        self.remove_node(u)
        self.build_graph_from_watershed(idxs=node_extent)
        self.ncut(num_clusters=n, nodes=labels, **kwargs)

    def separate_fragments(self, f0, f1):
        """Ensure fragments (watersheds) f0 and f1 are in different nodes.

        If f0 and f1 are the same segment, split that segment at the
        lowest common ancestor of f0 and f1 in the merge tree, then add an
        exclusion. Otherwise, simply add an exclusion.

        Parameters
        ----------
        f0, f1 : int
            The fragments to be separated.

        Returns
        -------
        s0, s1 : int
            The separated segments resulting from the break. If the
            fragments were already in separate segments, return the
            highest ancestor of each fragment on the merge tree.
        """
        lca = tree.lowest_common_ancestor(self.tree, f0, f1)
        if lca is not None:
            s0, s1 = self.tree.children(lca)
            self.delete_merge(lca)
        else:
            s0 = self.tree.highest_ancestor(f0)
            s1 = self.tree.highest_ancestor(f1)
        return s0, s1

    def delete_merge(self, tree_node):
        """Delete the merge represented by `tree_node`.

        Parameters
        ----------
        tree_node : int
            A node that may not be currently in the graph, but was at
            some point in its history.
        """
        highest = self.tree.highest_ancestor(tree_node)
        if highest != tree_node:
            leaves = self.tree.leaves(tree_node)
            # the graph doesn't keep nodes in the history, only the
            # most recent nodes. So, we only need to find that one and
            # update its fragment list.
            self.nodes[highest]['fragments'].difference_update(leaves)
        self.tree.remove_node(tree_node)

    def move_edge(self, src, dst):
        """Move edge `src` to `dst`. If `dst` exists it is clobbered.

        Parameters
        ----------
        src, dst : (int, int)
            The edges being merged.
        """
        u, v = dst
        w, x = src
        self.add_edge(u, v, **self.edges[w, x])
        if 'qlink' in self.edges[u, v]:
            qelem = self.edges[u, v]['qlink']
            qelem[2:] = u, v

    def merge_edge_properties(self, src, dst):
        """Merge the properties of edge src into edge dst.

        Parameters
        ----------
        src, dst : (int, int)
            Edges being merged.

        Returns
        -------
        None
        """
        u, v = dst
        w, x = src
        if not self.has_edge(u,v):
            self.add_edge(u, v, **self.edges[w, x])
        else:
            if self.fast_edges:
                self.edges[u, v]['boundary-ids'].update(
                        self.edges[w, x]['boundary-ids']
                        )
            else:
                self.edges[u, v]['boundary'].extend(
                        self.edges[w, x]['boundary']
                        )
            self.feature_manager.update_edge_cache(
                    self,
                    (u, v),
                    (w, x),
                    self.edges[u, v]['feature-cache'],
                    self.edges[w, x]['feature-cache'],
                    )
        try:
            self.merge_queue.invalidate(self.edges[w, x]['qlink'])
        except KeyError:
            pass


    def update_merge_queue(self, edges):
        """Update the merge queue item for edge (u, v). Add new by default.

        Parameters
        ----------
        u, v : int (node id)
            Edge being updated.

        Returns
        -------
        None
        """
        edges = [e for e in edges if self.boundary_body not in e]
        if not self.merge_queue.is_null_queue and edges:
            weights = self.merge_priority_function(self, edges)
            for w, (u, v) in zip(weights, edges):
                if 'qlink' in self.edges[u, v]:
                    self.merge_queue.invalidate(self.edges[u, v]['qlink'])
                new_qitem = [w, True, u, v]
                self.edges[u, v]['qlink'] = new_qitem
                self.edges[u, v]['weight'] = w
                self.merge_queue.push(new_qitem)


    def get_segmentation(self, threshold=None):
        """Return the unpadded segmentation represented by the graph.

        Remember that the segmentation volume is padded with an
        "artificial" segment that envelops the volume. This function
        simply removes the wrapping and returns a segmented volume.

        Parameters
        ----------
        threshold : float, optional
            Get the segmentation at the given threshold. If no
            threshold is given, return the segmentation at the current
            level of agglomeration.

        Returns
        -------
        seg : array of int
            The segmentation of the volume presently represented by the
            graph.
        """
        if threshold is None:
            # a threshold of np.inf is the same as no threshold on the
            # tree when getting the map (see below). Thus, using a
            # threshold of `None` (the default), we get the segmentation
            # implied by the current merge tree.
            threshold = np.inf
        elif threshold > self.max_merge_score:
            # If a higher threshold is required than has been merged, we
            # continue the agglomeration until that threshold is hit.
            self.agglomerate(threshold)
        m = self.tree.get_map(threshold)
        seg = m[self.watershed]
        if self.pad_thickness > 1: # volume has zero-boundaries
            seg = morpho.remove_merged_boundaries(seg, self.connectivity)
        return morpho.juicy_center(seg, self.pad_thickness)


    def build_volume(self, nbunch=None):
        """Return the segmentation induced by the graph.

        Parameters
        ----------
        nbunch : iterable of int (node id), optional
            A list of nodes for which to build the volume. All nodes
            are used if this is not provided.

        Returns
        -------
        seg : array of int
            The segmentation implied by the graph.

        Notes
        -----
        This function is very similar to ``get_segmentation``, but it
        builds the segmentation from the bottom up, rather than using
        the currently-stored segmentation.
        """
        v = zeros_like(self.watershed)
        vr = v.ravel()
        if nbunch is None:
            nbunch = self.nodes()
        for n in nbunch:
            vr[self.extent(n)] = n
        return morpho.juicy_center(v, self.pad_thickness)


    def build_boundary_map(self, ebunch=None):
        """Return a map of the current merge priority.

        Parameters
        ----------
        ebunch : iterable of (int, int), optional
            The list of edges for which to build a map. Use all edges
            if not provided.

        Returns
        -------
        bm : array of float
            The image of the edge weights.
        """
        if len(self.merge_queue) == 0:
            self.rebuild_merge_queue()
        m = zeros(self.watershed.shape, 'float')
        mr = m.ravel()
        if ebunch is None:
            ebunch = self.real_edges_iter()
        ebunch = sorted(
                [(self.edges[u, v]['weight'], u, v) for u, v in ebunch]
                )
        for w, u, v in ebunch:
            b = self.boundary(u, v)
            mr[b] = w
        if hasattr(self, 'ignored_boundary'):
            m[self.ignored_boundary] = inf
        return morpho.juicy_center(m, self.pad_thickness)


    def remove_obvious_inclusions(self):
        """Merge any nodes with only one edge to their neighbors."""
        for n in self.nodes():
            if self.degree(n) == 1:
                self.merge_nodes(self.neighbors(n)[0], n)


    def remove_inclusions(self):
        """Merge any segments fully contained within other segments.

        In 3D EM images, inclusions are not biologically plausible, so
        this function can be used to remove them.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        bcc = list(biconnected_components(self))
        if len(bcc) > 1:
            container = [i for i, s in enumerate(bcc) if
                         self.boundary_body in s][0]
            del bcc[container] # remove the main graph
            bcc = list(map(list, bcc))
            for cc in bcc:
                cc.sort(key=lambda x: self.nodes[x]['size'], reverse=True)
            bcc.sort(key=lambda x: self.nodes[x[0]]['size'])
            for cc in bcc:
                self.merge_subgraph(cc, cc[0])


    def orphans(self):
        """List all the nodes that do not touch the volume boundary.

        Parameters
        ----------
        None

        Returns
        -------
        orphans : list of int (node id)
            A list of node ids.

        Notes
        -----
        "Orphans" are not biologically plausible in EM data, so we can
        flag them with this function for further scrutiny.
        """
        return [n for n in self.nodes() if not self.at_volume_boundary(n)]


    def compute_orphans(self):
        """Find all the segments that do not touch the volume boundary.

        Parameters
        ----------
        None

        Returns
        -------
        orphans : list of int (node id)
            A list of node ids.

        Notes
        -----
        This function differs from ``orphans`` in that it does not use
        the graph, but rather computes orphans directly from the
        segmentation.
        """
        return morpho.orphans(self.get_segmentation())


    def is_traversed_by_node(self, n):
        """Determine whether a body traverses the volume.

        This is defined as touching the volume boundary at two distinct
        locations.

        Parameters
        ----------
        n : int (node id)
            The node being inspected.

        Returns
        -------
        tr : bool
            Whether the segment "traverses" the volume being segmented.
        """
        if not self.at_volume_boundary(n) or n == self.boundary_body:
            return False
        v = zeros(self.watershed.shape, 'uint8')
        v.ravel()[self.boundary(n, self.boundary_body)] = 1
        _, n = label(v, ones([3]*v.ndim))
        return n > 1


    def traversing_bodies(self):
        """List all bodies that traverse the volume."""
        return [n for n in self.nodes() if self.is_traversed_by_node(n)]


    def non_traversing_bodies(self):
        """List bodies that are not orphans and do not traverse the volume."""
        return [n for n in self.nodes() if self.at_volume_boundary(n) and
                not self.is_traversed_by_node(n) and n != self.boundary_body]


    def raveler_body_annotations(self, traverse=False):
        """Return JSON-compatible dict formatted for Raveler annotations."""
        orphans = self.compute_orphans()
        non_traversing_bodies = self.compute_non_traversing_bodies() \
                                if traverse else []
        data = \
            [{'status':'not sure', 'comment':'orphan', 'body ID':int(o)}
                for o in orphans] +\
            [{'status':'not sure', 'comment':'does not traverse',
                'body ID':int(n)} for n in non_traversing_bodies]
        metadata = {'description':'body annotations', 'file version':2}
        return {'data':data, 'metadata':metadata}


    def at_volume_boundary(self, n):
        """Return True if node n touches the volume boundary."""
        return self.has_edge(n, self.boundary_body) or n == self.boundary_body


    def should_merge(self, n1, n2):
        return self.rig[n1].argmax() == self.rig[n2].argmax()


    def get_pixel_label(self, n1, n2):
        boundary = self.boundary(n1, n2)
        min_idx = boundary[self.probabilities_r[boundary,0].argmin()]
        if self.should_merge(n1, n2):
            return min_idx, 2
        else:
            return min_idx, 1


    def pixel_labels_array(self, false_splits_only=False):
        ar = zeros_like(self.watershed_r)
        labels = [self.get_pixel_label(*e) for e in self.real_edges()]
        if false_splits_only:
            labels = [l for l in labels if l[1] == 2]
        ids, ls = list(map(array,zip(*labels)))
        ar[ids] = ls.astype(ar.dtype)
        return ar.reshape(self.watershed.shape)


    def split_vi(self, gt=None):
        if self.gt is None and gt is None:
            return array([0,0])
        elif self.gt is not None:
            return split_vi(self.rig)
        else:
            return split_vi(self.get_segmentation(), gt, [0], [0])

    def get_edge_coordinates(self, n1, n2, arbitrary=False):
        """Find where in the segmentation the edge (n1, n2) is most visible."""
        return get_edge_coordinates(self, n1, n2, arbitrary)


    def write(self, fout, output_format='GraphML'):
        if output_format == 'Plaza JSON':
            self.write_plaza_json(fout)
        else:
            raise ValueError('Unsupported output format for agglo.Rag: %s'
                % output_format)


    def write_plaza_json(self, fout, synapsejson=None, offsetz=0):
        """Write graph to Steve Plaza's JSON spec."""
        json_vals = {}
        if synapsejson is not None:
            synapse_file = open(synapsejson)
            json_vals1 = json.load(synapse_file)
            body_count = {}

            for item in json_vals1["data"]:
                bodyid = ((item["T-bar"])["body ID"])
                if bodyid in body_count:
                    body_count[bodyid] += 1
                else:
                    body_count[bodyid] = 1
                for psd in item["partners"]:
                    bodyid = psd["body ID"]
                    if bodyid in body_count:
                        body_count[bodyid] += 1
                    else:
                        body_count[bodyid] = 1

            json_vals["synapse_bodies"] = []
            for body, count in body_count.items():
                temp = [body, count]
                json_vals["synapse_bodies"].append(temp)

        edge_list = [
            {'location': list(map(int, self.get_edge_coordinates(i, j)[-1::-1])),
            'node1': int(i), 'node2': int(j),
            'edge_size': len(self.boundary(i, j)),
            'size1': self.nodes[i]['size'],
            'size2': self.nodes[j]['size'],
            'weight': float(self.edges[i, j]['weight'])}
            for i, j in self.real_edges()
        ]
        json_vals['edge_list'] = edge_list

        with open(fout, 'w') as f:
            json.dump(json_vals, f, indent=4)


    def ncut(self, num_clusters=10, kmeans_iters=5, sigma=255.0*20, nodes=None,
            **kwargs):
        """Run normalized cuts on the current set of fragments.

        Parameters
        ----------
        num_clusters : int, optional
            The desired number of clusters
        kmeans_iters : int, optional
            The maximum number of iterations for the kmeans clustering
            of the Laplacian eigenvectors.
        sigma : float, optional
            The damping factor on the edge weights. The higher this value,
            the closer to 1 (the maximum) edges with large weights will be.
        nodes : collection of int, optional
            Restrict the ncut to the listed nodes.
        """
        if nodes is None:
            nodes = self.nodes()
        # Compute weight matrix
        W = self.compute_W(self.merge_priority_function, nodes=nodes)
        # Run normalized cut
        labels, eigvec, eigval = ncutW(W, num_clusters, kmeans_iters, **kwargs)
        # Merge nodes that are in same cluster
        self.cluster_by_labels(labels, nodes)


    def cluster_by_labels(self, labels, nodes=None):
        """Merge all superpixels with the same label (1 label per 1 sp)"""
        if nodes is None:
            nodes = array(self.nodes())
        if not (len(labels) == len(nodes)):
            raise ValueError('Number of labels should be %d but is %d.',
                self.number_of_nodes(), len(labels))
        for l in unique(labels):
            inds = nonzero(labels==l)[0]
            nodes_to_merge = nodes[inds]
            node1 = nodes_to_merge[0]
            for node in nodes_to_merge[1:]:
                self.merge_nodes(node1, node)


    def compute_W(self, distance_function, sigma=255.0*20, nodes=None):
        """Compute the weight matrix for n-cut clustering.

        See `ncut` for parameters.
        """
        if nodes is None:
            nodes = array(self.nodes())
        n = len(nodes)
        nodes2ind = dict(zip(nodes, range(n)))
        W = lil_matrix((n, n))
        for u, v in self.real_edges(nodes):
            try:
                i, j = nodes2ind[u], nodes2ind[v]
            except KeyError:
                continue
            w = distance_function(self, (u, v))
            W[i, j] = W[j, i] = np.exp(-w**2 / sigma)
        return W.tocsr()


    def update_frozen_sets(self, n1, n2):
        self.frozen_nodes.discard(n1)
        self.frozen_nodes.discard(n2)
        for x, y in self.frozen_edges.copy():
            if n2 in [x, y]:
                self.frozen_edges.discard((x, y))
            if x == n2:
                self.frozen_edges.add((n1, y))
            if y == n2:
                self.frozen_edges.add((x, n1))


def get_edge_coordinates(g, n1, n2, arbitrary=False):
    """Find where in the segmentation the edge (n1, n2) is most visible."""
    boundary = g.boundary(n1, n2)
    if arbitrary:
        # quickly get an arbitrary point on the boundary
        idx = boundary.pop(); boundary.append(idx)
        coords = unravel_index(idx, g.watershed.shape)
    else:
        boundary_idxs = unravel_index(boundary, g.watershed.shape)
        coords = [bincount(dimcoords).argmax() for dimcoords in boundary_idxs]
    return array(coords) - g.pad_thickness


def is_mito_boundary(g, n1, n2, channel=2, threshold=0.5):
    return max(np.mean(g.probabilities_r[g.boundary(n1, n2), c])
               for c in channel) > threshold


def is_mito(g, n, channel=2, threshold=0.5):
    return max(np.mean(g.probabilities_r[g.extent(n), c])
               for c in channel) > threshold


def best_possible_segmentation(ws, gt):
    """Build the best possible segmentation given a superpixel map."""
    ws = Rag(ws)
    assignment = ev.assignment_table(ws.get_segmentation(), gt).tocsc()
    for gt_node in range(assignment.shape[1]):
        i, j = assignment.indptr[gt_node : gt_node+2]
        ws.merge_subgraph(assignment.indices[i:j])
    return ws.get_segmentation()
