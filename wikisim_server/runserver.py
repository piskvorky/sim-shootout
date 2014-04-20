#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <me@radimrehurek.com>

"""
USAGE: %(program)s CONFIG

    Start the wiki-sim server for http://radimrehurek.com/2014/01/performance-shootout-of-nearest-neighbours-querying/#wikisim
and leave it running (ctrl+c to quit).

Ports/interfaces for the server are specified in the config file.

Example:
    ./runserver.py hetzner.conf

"""

from __future__ import with_statement

import os
import sys
from functools import wraps
import time
import logging

import cherrypy
from cherrypy.process.plugins import DropPrivileges, PIDFile
import annoy
import gensim


logger = logging.getLogger(__name__)


def server_exception_wrap(func):
    """
    Method decorator to return nicer JSON responses: handle internal server errors & request timings.

    """
    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        try:
            # append "success=1" and time taken in milliseconds to the response, on success
            logger.debug("calling server method '%s'" % (func.func_name))
            cherrypy.response.timeout = 3600 * 24 * 7  # [s]

            # include json data in kwargs; HACK: should be a separate decorator really, but whatever
            if getattr(cherrypy.request, 'json', None):
                kwargs.update(cherrypy.request.json)

            start = time.time()
            result = func(self, *args, **kwargs)
            if result is None:
                result = {}
            result['success'] = 1
            result['taken'] = time.time() - start
            logger.info("method '%s' succeeded in %ss" % (func.func_name, result['taken']))
            return result
        except Exception, e:
            logger.exception("exception serving request")
            result = {
                'error': repr(e),
                'success': 0,
            }
            cherrypy.response.status = 500
            return result
    return _wrapper


class Server(object):
    def __init__(self, basedir, k):
        self.basedir = basedir
        self.k = k

        self.index_annoy = annoy.AnnoyIndex(500, metric='angular')
        self.index_annoy.load(os.path.join(basedir, 'index3507620_annoy_100'))
        self.id2title = gensim.utils.unpickle(os.path.join(basedir, 'id2title'))
        self.title2id = dict((gensim.utils.to_unicode(title).lower(), pos) for pos, title in enumerate(self.id2title))

    @server_exception_wrap
    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def similars(self, *args, **kwargs):
        """
        For a Wiki article named `title`, return the top-k most similar Wikipedia articles.

        """
        title = gensim.utils.to_unicode(kwargs.pop('title', u'')).lower()
        if title in self.title2id:
            logger.info("finding similars for %s" % title)
            query = self.title2id[title]  # convert query from article name (string) to index id (int)
            nn = self.index_annoy.get_nns_by_item(query, self.k)
            result = [self.id2title[pos2] for pos2 in nn]  # convert top10 from ids back to article names
            logger.info("similars to %s: %s" % (title, result))
        else:
            result = []
        return {'nn': result, 'num_articles': len(self.id2title)}

    @server_exception_wrap
    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def status(self, *args, **kwargs):
        """
        Return the server status.

        """
        result = {
            'basedir': self.basedir,
            'k': self.k,
            'num_articles': len(self.id2title)
        }
        return result
    ping = status
#endclass Server


class Config(object):
    def __init__(self, **d):
        self.__dict__.update(d)

    def __getattr__(self, name):
        return None # unset config values will default to None

    def __getitem__(self, name):
        return self.__dict__[name]


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(module)s:%(lineno)d : %(funcName)s(%(threadName)s) : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logging.info("running %s" % ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) < 2:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    conf_file = sys.argv[1]
    config_srv = Config(**cherrypy.lib.reprconf.Config(conf_file).get('global'))
    config = Config(**cherrypy.lib.reprconf.Config(conf_file).get('wiki_sim'))

    if config_srv.pid_file:
        PIDFile(cherrypy.engine, config_srv.pid_file).subscribe()
    if config_srv.run_user and config_srv.run_group:
        logging.info("dropping priviledges to %s:%s" % (config_srv.run_user, config_srv.run_group))
        DropPrivileges(cherrypy.engine, gid=config_srv.run_group, uid=config_srv.run_user).subscribe()

    cherrypy.quickstart(Server(config.BASE_DIR, config.TOPN), config=conf_file)

    logging.info("finished running %s" % program)
