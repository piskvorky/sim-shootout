#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>

"""
USAGE: %(program)s INPUT_DIRECTORY

Compare speed and accuracy of several similarity retrieval methods, using the corpus prepared by prepare_shootout.py.

Example: ./shootout.py ~/data/wiki/shootout

"""

import os
import sys
import time
import logging
import itertools
from functools import wraps

import numpy

import gensim

MAX_DOCS = 10000000  # clip the dataset at this many docs, if larger (=use a wiki subset)
TOP_N = 10  # how many similars to ask for
ACC = 'exact'  # what accuracy are we aiming for
NUM_QUERIES = 100  # query with this many different documents, as a single experiment
REPEATS = 3  # run all queries this many times, take the best timing


FLANN_995 = {
    'log_level': 'info',
    'target_precision': 0.995,
    'algorithm': 'autotuned',
}

FLANN_99 = {  # autotuned on wiki corpus with target_precision=0.99
    'iterations': 5,
    'multi_probe_level_': 2L,
    'cb_index': 0.20000000298023224,
    'centers_init': 'default',
    'log_level': 'info',
    'build_weight': 0.009999999776482582,
    'leaf_max_size': 4,
    'memory_weight': 0.0,
    'sample_fraction': 0.10000000149011612,
    'checks': 12288,
    'max_neighbors': -1,
    'random_seed': 215924497,
    'trees': 1,
    'target_precision': 0.99,
    'table_number_': 12L,
    'sorted': 1,
    'branching': 32,
    'algorithm': 'kmeans',
    'key_size_': 20L,
    'eps': 0.0,
    'cores': 0
}

FLANN_98 = {
    'iterations': 5,
    'multi_probe_level_': 2L,
    'cb_index': 0.20000000298023224,
    'centers_init': 'default',
    'log_level': 'info',
    'build_weight': 0.009999999776482582,
    'leaf_max_size': 4,
    'memory_weight': 0.0,
    'sample_fraction': 0.10000000149011612,
    'checks': 5072,
    'max_neighbors': -1,
    'random_seed': 638707089,
    'trees': 1,
    'target_precision': 0.98,
    'table_number_': 12L,
    'sorted': 1,
    'branching': 16,
    'algorithm': 'kmeans',
    'key_size_': 20L,
    'eps': 0.0,
    'cores': 0,
}

FLANN_95 = {
    'iterations': 5,
    'multi_probe_level_': 2L,
    'cb_index': 0.20000000298023224,
    'centers_init': 'default',
    'log_level': 'info',
    'build_weight': 0.009999999776482582,
    'leaf_max_size': 4,
    'memory_weight': 0.0,
    'sample_fraction': 0.10000000149011612,
    'checks': 3072,
    'max_neighbors': -1,
    'random_seed': 638707089,
    'trees': 1,
    'target_precision': 0.949999988079071,
    'table_number_': 12L,
    'sorted': 1,
    'branching': 16,
    'algorithm': 'kmeans',
    'key_size_': 20L,
    'eps': 0.0,
    'cores': 0,
}

FLANN_9 = {
    'algorithm': 'kmeans',
    'branching': 32,
    'build_weight': 0.009999999776482582,
    'cb_index': 0.20000000298023224,
    'centers_init': 'default',
    'checks': 2688,
    'cores': 0,
    'eps': 0.0,
    'iterations': 1,
    'key_size_': 20L,
    'leaf_max_size': 4,
    'log_level': 'info',
    'max_neighbors': -1,
    'memory_weight': 0.0,
    'multi_probe_level_': 2L,
    'random_seed': 354901449,
    'sample_fraction': 0.10000000149011612,
    'sorted': 1,
    'table_number_': 12L,
    'target_precision': 0.9,
    'trees': 1
}

FLANN_7 = {
    'iterations': 5,
    'multi_probe_level_': 2L,
    'cb_index': 0.0,
    'centers_init': 'default',
    'log_level': 'info',
    'build_weight': 0.009999999776482582,
    'leaf_max_size': 4,
    'memory_weight': 0.0,
    'sample_fraction': 0.10000000149011612,
    'checks': 684,
    'max_neighbors': -1,
    'random_seed': 746157721,
    'trees': 4,
    'target_precision': 0.699999988079071,
    'table_number_': 12L,
    'sorted': 1,
    'branching': 32,
    'algorithm': 'default',
    'key_size_': 20L,
    'eps': 0.0,
    'cores': 0
}


ACC_SETTINGS = {
    'flann': {'7': FLANN_7, '9': FLANN_9, '95': FLANN_95, '99': FLANN_99, '995': FLANN_995},
    'annoy': {'1': 1, '10': 10, '50': 50, '100': 100, '500': 500},
    'lsh': {'low': {'k': 10, 'l': 10, 'w': float('inf')}, 'high': {'k': 10, 'l': 10, 'w': float('inf')}},
}


logger = logging.getLogger('shootout')

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        times = []
        logger.info("benchmarking %s at k=%s acc=%s" % (fn.__name__, TOP_N, ACC))
        for _ in xrange(REPEATS):  # try running it three times, report the best time
            start = time.time()
            ret = fn(*args, **kwargs)
            times.append(time.time() - start)
        logger.info("%s took %.3fms/query" % (fn.__name__, 1000.0 * min(times) / NUM_QUERIES))
        logger.info("%s raw timings: %s" % (fn.__name__, times))
        return ret

    return with_profiling


@profile
def gensim_1by1(index, queries):
    for query in queries:
        _ = index[query]

@profile
def gensim_at_once(index, queries):
    _ = index[queries]

@profile
def flann_1by1(index, queries):
    for query in queries:
        _ = index.nn_index(query, TOP_N)

@profile
def flann_at_once(index, queries):
    _ = index.nn_index(queries, TOP_N)

@profile
def sklearn_1by1(index, queries):
    for query in queries:
        _ = index.kneighbors(query, n_neighbors=TOP_N)

@profile
def sklearn_at_once(index, queries):
    _ = index.kneighbors(queries, n_neighbors=TOP_N)

@profile
def kgraph_1by1(index, queries, dataset):
    for query in queries:
        _ = index.search(dataset, query[None, :], K=TOP_N, threads=1)

@profile
def kgraph_at_once(index, queries, dataset):
    _ = index.search(dataset, queries, K=TOP_N, threads=1)

@profile
def annoy_1by1(index, queries):
    for query in queries:
        _ = index.get_nns_by_vector(list(query.astype(float)), TOP_N)

@profile
def lsh_1by1(index, queries):
    for query in queries:
        _ = index.Find(query[:, None])[:TOP_N]


def flann_predictions(index, queries):
    if TOP_N == 1:
        # flann returns differently shaped arrays when asked for only 1 nearest neighbour
        return [index.nn_index(query, TOP_N)[0] for query in queries]
    else:
        return [index.nn_index(query, TOP_N)[0][0] for query in queries]


def sklearn_predictions(index, queries):
    return [list(index.kneighbors(query, TOP_N)[1].ravel()) for query in queries]


def annoy_predictions(index, queries):
    return [index.get_nns_by_vector(list(query.astype(float)), TOP_N) for query in queries]


def lsh_predictions(index, queries):
    return [[pos for pos, _ in index_lsh.Find(query[:, None])[:TOP_N]] for query in queries]


def gensim_predictions(index, queries):
    return [[pos for pos, _ in index[query]] for query in queries]


def kgraph_predictions(index, queries):
    global dataset
    return index.search(dataset, queries, K=TOP_N, threads=1)


def get_accuracy(predicted_ids, queries, gensim_index, expecteds=None):
    """Return precision (=percentage of overlapping ids) and average similarity difference."""
    logger.info("computing ground truth")
    correct, diffs = 0.0, []
    for query_no, (predicted, query) in enumerate(zip(predicted_ids, queries)):
        expected_ids, expected_sims = zip(*gensim_index[query]) if expecteds is None else expecteds[query_no]
        correct += len(set(expected_ids).intersection(predicted))
        predicted_sims = [numpy.dot(gensim_index.vector_by_id(id1), query) for id1 in predicted]
        # if we got less than TOP_N results, assume zero similarity for the missing ids (LSH)
        predicted_sims.extend([0.0] * (TOP_N - len(predicted_sims)))
        diffs.extend(-numpy.array(predicted_sims) + expected_sims)
    return correct / (TOP_N * len(queries)), numpy.mean(diffs), numpy.std(diffs), max(diffs)


def log_precision(method, index, queries, gensim_index, expecteds=None):
    logger.info("computing accuracy of %s over %s queries at k=%s, acc=%s" % (method.__name__, NUM_QUERIES, TOP_N, ACC))
    acc, avg_diff, std_diff, max_diff = get_accuracy(method(index, queries), queries, gensim_index, expecteds)
    logger.info("%s precision=%.3f, avg diff=%.3f, std diff=%.5f, max diff=%.3f" % (method.__name__, acc, avg_diff, std_diff, max_diff))


def print_similar(title, index_gensim, id2title, title2id):
    """Print out the most similar Wikipedia articles, given an article title=query"""
    pos = title2id[title.lower()]  # throws if title not found
    for pos2, sim in index_gensim[index_gensim.vector_by_id(pos)]:
        print pos2, `id2title[pos2]`, sim


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    indir = sys.argv[1]
    if len(sys.argv) > 2:
        TOP_N = int(sys.argv[2])
    if len(sys.argv) > 3:
        ACC = sys.argv[3]
    lsi_vectors = os.path.join(indir, 'lsi_vectors.mm.gz')
    logger.info("testing k=%s and acc=%s" % (TOP_N, ACC))

    mm = gensim.corpora.MmCorpus(gensim.utils.smart_open(lsi_vectors))
    num_features, num_docs = mm.num_terms, min(mm.num_docs, MAX_DOCS)
    sim_prefix = os.path.join(indir, 'index%s' % num_docs)

    # some libs (flann, sklearn) expect the entire input as a full matrix, all at once (no streaming)
    if os.path.exists(sim_prefix + "_clipped.npy"):
        logger.info("loading dense corpus (need for flann, scikit-learn)")
        clipped = numpy.load(sim_prefix + "_clipped.npy", mmap_mode='r')
    else:
        logger.info("creating dense corpus of %i documents under %s" % (num_docs, sim_prefix + "_clipped.npy"))
        clipped = numpy.empty((num_docs, num_features), dtype=numpy.float32)
        for docno, doc in enumerate(itertools.islice(mm, num_docs)):
            if docno % 100000 == 0:
                logger.info("at document #%i/%i" % (docno + 1, num_docs))
            clipped[docno] = gensim.matutils.sparse2full(doc, num_features)
        numpy.save(sim_prefix + "_clipped.npy", clipped)
    clipped_corpus = gensim.matutils.Dense2Corpus(clipped, documents_columns=False)  # same as islice(mm, num_docs)

    logger.info("selecting %s documents, to act as top-%s queries" % (NUM_QUERIES, TOP_N))
    queries = clipped[:NUM_QUERIES]

    if os.path.exists(sim_prefix + "_gensim"):
        logger.info("loading gensim index")
        index_gensim = gensim.similarities.Similarity.load(sim_prefix + "_gensim")
        index_gensim.output_prefix = sim_prefix
        index_gensim.check_moved()  # update shard locations in case the files were copied somewhere else
    else:
        logger.info("building gensim index")
        index_gensim = gensim.similarities.Similarity(sim_prefix, clipped_corpus, num_best=TOP_N, num_features=num_features, shardsize=100000)
        index_gensim.save(sim_prefix + "_gensim")
    index_gensim.num_best = TOP_N
    logger.info("finished gensim index %s" % index_gensim)

    logger.info("loading mapping between article titles and ids")
    id2title = gensim.utils.unpickle(os.path.join(indir, 'id2title'))
    title2id = dict((title.lower(), pos) for pos, title in enumerate(id2title))
    # print_similar('Anarchism', index_gensim, id2title, title2id)

    if 'gensim' in program:
        # log_precision(gensim_predictions, index_gensim, queries, index_gensim)
        gensim_at_once(index_gensim, queries)
        gensim_1by1(index_gensim, queries)

    if 'flann' in program:
        import pyflann
        pyflann.set_distance_type('euclidean')
        index_flann = pyflann.FLANN()
        flann_fname = sim_prefix + "_flann_%s" % ACC
        if os.path.exists(flann_fname):
            logger.info("loading flann index")
            index_flann.load_index(flann_fname, clipped)
        else:
            logger.info("building FLANN index")
            # flann expects index vectors as a 2d numpy array, features = columns
            params = index_flann.build_index(clipped, **ACC_SETTINGS['flann'][ACC])
            logger.info("built flann index with %s" % params)
            index_flann.save_index(flann_fname)
        logger.info("finished FLANN index")

        log_precision(flann_predictions, index_flann, queries, index_gensim)
        flann_1by1(index_flann, queries)
        flann_at_once(index_flann, queries)

    if 'annoy' in program:
        import annoy
        index_annoy = annoy.AnnoyIndex(num_features, metric='angular')
        annoy_fname = sim_prefix + "_annoy_%s" % ACC
        if os.path.exists(annoy_fname):
            logger.info("loading annoy index")
            index_annoy.load(annoy_fname)
        else:
            logger.info("building annoy index")
            # annoy expects index vectors as lists of Python floats
            for i, vec in enumerate(clipped_corpus):
                index_annoy.add_item(i, list(gensim.matutils.sparse2full(vec, num_features).astype(float)))
            index_annoy.build(ACC_SETTINGS['annoy'][ACC])
            index_annoy.save(annoy_fname)
            logger.info("built annoy index")

        log_precision(annoy_predictions, index_annoy, queries, index_gensim)
        annoy_1by1(index_annoy, queries)

    if 'lsh' in program:
        import lsh
        if os.path.exists(sim_prefix + "_lsh"):
            logger.info("loading lsh index")
            index_lsh = gensim.utils.unpickle(sim_prefix + "_lsh")
        else:
            logger.info("building lsh index")
            index_lsh = lsh.index(**ACC_SETTINGS['lsh'][ACC])
            # lsh expects input as D x 1 numpy arrays
            for vecno, vec in enumerate(clipped_corpus):
                index_lsh.InsertIntoTable(vecno, gensim.matutils.sparse2full(vec)[:, None])
            gensim.utils.pickle(index_lsh, sim_prefix + '_lsh')
        logger.info("finished lsh index")

        log_precision(lsh_predictions, index_lsh, queries, index_gensim)
        lsh_1by1(index_lsh, queries)

    if 'sklearn' in program:
        from sklearn.neighbors import NearestNeighbors
        if os.path.exists(sim_prefix + "_sklearn"):
            logger.info("loading sklearn index")
            index_sklearn = gensim.utils.unpickle(sim_prefix + "_sklearn")
        else:
            logger.info("building sklearn index")
            index_sklearn = NearestNeighbors(n_neighbors=TOP_N, algorithm='auto').fit(clipped)
            logger.info("built sklearn index %s" % index_sklearn._fit_method)
            gensim.utils.pickle(index_sklearn, sim_prefix + '_sklearn')  # 32GB RAM not enough to store the scikit-learn model...
        logger.info("finished sklearn index")

        log_precision(sklearn_predictions, index_sklearn, queries, index_gensim)
        sklearn_1by1(index_sklearn, queries)
        sklearn_at_once(index_sklearn, queries)

    if 'kgraph' in program:
        import pykgraph
        index_kgraph = pykgraph.KGraph()
        if os.path.exists(sim_prefix + "_kgraph"):
            logger.info("loading kgraph index")
            index_kgraph.load(sim_prefix + "_kgraph")
        else:
            logger.info("building kgraph index")
            index_kgraph.build(clipped)
            logger.info("built kgraph index")
            index_kgraph.save(sim_prefix + "_kgraph")
        logger.info("finished kgraph index")

        global dataset
        dataset = clipped
        log_precision(kgraph_predictions, index_kgraph, queries, index_gensim)
        kgraph_1by1(index_kgraph, queries, clipped)
        kgraph_at_once(index_kgraph, queries, clipped)

    logger.info("finished running %s" % program)
