#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>

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
import random
import cPickle
from functools import wraps

import numpy

import gensim

MAX_DOCS = 1000000  # clip the dataset to be indexed at this many docs, if larger
NUM_QUERIES = 100
TOP_N = 50  # how many similars to ask for

logger = logging.getLogger('shootout')

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        times = []
        logger.info("benchmarking %s" % fn.__name__)
        for _ in xrange(3):  # try running it three times, report the best time
            start = time.time()
            ret = fn(*args, **kwargs)
            times.append(time.time() - start)
        logger.info("%s took %.3fs" % (fn.__name__, min(times)))
        logger.debug("%s raw timings: %s" % (fn.__name__, times))
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
def annoy_1by1(index, queries):
    for query in queries:
        _ = index.get_nns_by_vector(list(query.astype(float)), TOP_N)


def flann_precision(index_gensim, index_flann, queries):
    correct, diffs = 0, []
    for query in queries:
        expected_ids, expected_sims = zip(*index_gensim[query])
        predicted = index_flann.nn_index(query, TOP_N)
        correct += len(set(expected_ids).intersection(predicted[0][0]))  # how many top-n did flann get right?
        predicted_sims = 1 - predicted[1][0] / 2  # convert flann output to cossim score
        diffs.extend(-predicted_sims + expected_sims)  # how far was flann from the correct values?
    logger.info("flann precision=%.3f, avg diff=%.3f" %
        (1.0 * correct / (TOP_N * len(queries)), 1.0 * sum(diffs) / len(diffs)))

def annoy_precision(index_gensim, index_annoy, queries):
    correct, diffs = 0, []
    for query in queries:
        expected_ids, expected_sims = zip(*index_gensim[query])
        predicted = index_annoy.get_nns_by_vector(list(query.astype(float)), TOP_N)
        correct += len(set(expected_ids).intersection(predicted))  # how many top-n did annoyy get right?
        predicted_sims = [numpy.dot(index_gensim.vector_by_id(i), query) for i in predicted]
        diffs.extend(-numpy.array(predicted_sims) + expected_sims)  # how far was flann from the correct values?
    logger.info("annoy precision=%.3f, avg diff=%.3f" %
        (1.0 * correct / (TOP_N * len(queries)), 1.0 * sum(diffs) / len(diffs)))


def gensim_precision(index_gensim, queries):
    correct, diffs = 0, []
    for query in queries:
        expected_ids, expected_sims = zip(*index_gensim[query])
        predicted, predicted_sims = expected_ids, expected_sims
        correct += len(set(expected_ids).intersection(predicted))  # how many top-n did annoyy get right?
        diffs.extend(-numpy.array(predicted_sims) + expected_sims)  # how far was flann from the correct values?
    logger.info("gensim precision=%.3f, avg diff=%.3f" %
        (1.0 * correct / (TOP_N * len(queries)), 1.0 * sum(diffs) / len(diffs)))


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
    lsi_vectors = os.path.join(indir, 'lsi_vectors.mm.gz')
    sim_prefix = os.path.join(indir, 'index%s' % MAX_DOCS)

    mm = gensim.corpora.MmCorpus(gensim.utils.smart_open(lsi_vectors))
    num_features = mm.num_terms
    if os.path.exists(sim_prefix + "_clipped.npy"):
        logger.info("loading dense corpus")
        clipped = numpy.load(sim_prefix + "_clipped.npy", mmap_mode='r')
    else:
        # precompute the entire corpus as a dense matrix in RAM -- FLANN needs this anyway
        logger.info("creating dense corpus for FLANN")
        clipped = gensim.matutils.corpus2dense(itertools.islice(mm, MAX_DOCS), num_features).astype(numpy.float32, order='C').T
        numpy.save(sim_prefix + "_clipped.npy", clipped)
    clipped_corpus = gensim.matutils.Dense2Corpus(clipped, documents_columns=False)  # same as the islice(MAX_DOCS) above

    logger.info("selecting %s random documents, to act as top-%s queries" % (NUM_QUERIES, TOP_N))
    query_docs = random.sample(range(clipped.shape[0]), NUM_QUERIES)
    queries = clipped[query_docs]

    if os.path.exists(sim_prefix + "_gensim"):
        logger.info("loading gensim index")
        index_gensim = gensim.similarities.Similarity.load(sim_prefix + "_gensim")
        index_gensim.num_best = TOP_N
    else:
        logger.info("building gensim index")
        index_gensim = gensim.similarities.Similarity(sim_prefix, clipped_corpus, num_best=TOP_N, num_features=num_features, shardsize=100000)
        index_gensim.save(sim_prefix + "_gensim")
        logger.info("built gensim index %s" % index_gensim)

    gensim_precision(index_gensim, queries)  # sanity check & prewarm mmap memory
    gensim_1by1(index_gensim, queries)
    gensim_at_once(index_gensim, queries)

    if 'flann' in program:
        import pyflann

        pyflann.set_distance_type('euclidean')
        index_flann = pyflann.FLANN()
        if os.path.exists(sim_prefix + "_flann"):
            logger.info("loading flann index")
            index_flann.load_index(sim_prefix + "_flann", clipped)
        else:
            logger.info("building FLANN index")
            # flann; expects index vectors as a 2d numpy array, features = columns
            params = index_flann.build_index(clipped, algorithm="autotuned", target_precision=0.95, log_level="info")
            index_flann.save_index(sim_prefix + "_flann")
            logger.info("built FLANN index %s" % params)

        flann_precision(index_gensim, index_flann, queries)
        flann_1by1(index_flann, queries)
        flann_at_once(index_flann, queries)

    if 'annoy' in program:
        import annoy

        index_annoy = annoy.AnnoyIndex(num_features, metric='euclidean')
        if os.path.exists(sim_prefix + "_annoy"):
            logger.info("loading annoy index")
            index_annoy.load(sim_prefix + "_annoy")
        else:
            logger.info("building annoy index")
            # annoy; expects index vectors as lists of Python floats
            for i, vec in enumerate(clipped_corpus):
                index_annoy.add_item(i, list(gensim.matutils.sparse2full(vec, num_features).astype(float)))
            index_annoy.build(50)
            index_annoy.save(sim_prefix + "_annoy")
            logger.info("built annoy index")

        annoy_precision(index_gensim, index_annoy, queries)
        annoy_1by1(index_annoy, queries)

    logger.info("finished running %s" % program)
