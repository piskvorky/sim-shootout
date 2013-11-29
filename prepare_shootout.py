#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>

"""
USAGE: %(program)s enwiki-latest-pages-articles.xml.bz2 OUTPUT_DIRECTORY

Parse all articles from a raw bz2 Wikipedia dump => train a latent semantic model on the articles => store resulting files into OUTPUT_DIRECTORY:

* title_tokens.txt.gz: raw article titles and tokens, one article per line, "article_title[TAB]space_separated_tokens[NEWLINE]"
* dictionary: pickled mapping between word<=>word_id
* dictionary.txt: same as `dictionary` but in plain text format
* tfidf.model: pickled TF-IDF model
* lsi.model: pickled model for latent semantic analysis model, trained on TF-IDF'ed wiki dump
* lsi_vectors.mm: wikipedia articles stored as vectors in LSI space, in MatrixMarket format

The wikipedia dump can be downloaded from http://dumps.wikimedia.org/enwiki/latest/

Example:
    ./prepare_shootout.py ~/data/wiki/enwiki-latest-pages-articles.xml.bz2 ~/data/wiki/shootout
"""

import logging
import os
import sys
import multiprocessing
import bz2

import gensim

logger = logging.getLogger('prepare_shootout')

PROCESSES = max(1, multiprocessing.cpu_count() - 1)  # parallelize parsing using this many processes

MIN_WORDS = 50  # ignore articles with fewer tokens (redirects, stubs etc)

NUM_TOPICS = 500  # number of latent factors for LSA


def process_article((title, text)):
    """Parse a wikipedia article, returning its content as `(title, list of tokens)`, all utf8."""
    text = gensim.corpora.wikicorpus.filter_wiki(text)  # remove markup etc, get plain text
    return title.encode('utf8').replace('\t', ''), gensim.utils.simple_preprocess(text)  # tokenize


def convert_wiki(infile):
    """
    Yield articles from a bz2 Wikipedia dump `infile` as (title, tokens) 2-tuples.

    Only articles of sufficient length are returned (short articles & redirects
    etc are ignored).

    """
    logger.info("extracting articles from %s using %i processes" % (infile, PROCESSES))
    articles, articles_all = 0, 0
    positions, positions_all = 0, 0

    pool = multiprocessing.Pool(PROCESSES)
    # process the corpus in smaller chunks of docs, because multiprocessing.Pool
    # is dumb and would try to load the entire dump into RAM...
    texts = gensim.corpora.wikicorpus._extract_pages(bz2.BZ2File(infile))  # generator
    for group in gensim.utils.chunkize(texts, chunksize=10 * PROCESSES):
        for title, tokens in pool.imap(process_article, group):
            if articles_all % 100000 == 0:
                logger.info("PROGRESS: at article #%i: '%s'; accepted %i articles with %i total tokens" %
                    (articles_all, title, articles, positions))
            articles_all += 1
            positions_all += len(tokens)
            if len(tokens) >= MIN_WORDS and not title.startswith('Wikipedia:'): # article redirects and short stubs are pruned here
                articles += 1
                positions += len(tokens)
                yield title, tokens
    pool.terminate()

    logger.info("finished iterating over Wikipedia corpus of %i documents with %i positions"
        " (total %i articles, %i positions before pruning articles shorter than %i words)" %
        (articles, positions, articles_all, positions_all, MIN_WORDS))


class ShootoutCorpus(gensim.corpora.TextCorpus):
    def get_texts(self):
        lines = gensim.corpora.textcorpus.getstream(self.input)  # open file/reset stream to its start
        for lineno, line in enumerate(lines):
            yield line.split('\t')[1].split()
        self.length = lineno + 1


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    infile, outdir = sys.argv[1:3]
    outfile = lambda fname: os.path.join(outdir, fname)

    preprocessed_file = outfile('title_tokens.txt.gz')
    if not os.path.exists(preprocessed_file):
        with gensim.utils.smart_open(preprocessed_file, 'wb') as fout:
            for docno, (title, tokens) in enumerate(convert_wiki(infile)):
                try:
                    line = "%s\t%s" % (title.replace('\t', ''), ' '.join(tokens))
                    unicode(line, 'utf8')  # make sure everything's convertible to utf8
                    fout.write("%s\n" % line)
                except:
                    logger.info("invalid line at title %s" % title)

    dict_file = outfile('dictionary')
    if os.path.exists(dict_file):
        corpus = ShootoutCorpus()
        corpus.input = gensim.utils.smart_open(preprocessed_file)
        corpus.dictionary = gensim.corpora.Dictionary.load(dict_file)
    else:
        corpus = ShootoutCorpus(gensim.utils.smart_open(preprocessed_file))
        corpus.dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=50000)
        corpus.dictionary.save(dict_file)
        corpus.dictionary.save_as_text(dict_file + '.txt')

    tfidf_file = outfile('tfidf.model')
    if os.path.exists(tfidf_file):
        tfidf = gensim.models.TfidfModel.load(tfidf_file)
    else:
        tfidf = gensim.models.TfidfModel(corpus)
        tfidf.save(tfidf_file)

    lsi_file = outfile('lsi.model')
    if os.path.exists(lsi_file):
        lsi = gensim.models.LsiModel.load(lsi_file)
    else:
        lsi = gensim.models.LsiModel(tfidf[corpus], id2word=corpus.dictionary, num_topics=NUM_TOPICS, chunksize=10000)
        lsi.save(lsi_file)

    vectors_file = os.path.join(outdir, 'lsi_vectors.mm')
    gensim.corpora.MmCorpus.serialize(vectors_file, (gensim.matutils.unitvec(vec) for vec in lsi[tfidf[corpus]]))

    logger.info("finished running %s" % program)
