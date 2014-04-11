###############################################################################
#Copyright (C) 2013  Michael O. Vertolli michaelvertolli@gmail.com
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see http://www.gnu.org/licenses/
###############################################################################

"""
Masters thesis project that iteratively creates contextually 'coherent' sets
of labels from a co-occurrence matrix. The matrix is built from a training set
of labelled images from Peekaboom (http://dl.acm.org/citation.cfm?id=1124782).
This is an approach described by Paul Thagard (2000) with some minor
modifications.

Classes:
SemNet(path) -- algorithms for generating coherent sets given query

"""

import numpy, random, pickle
from Thesis_fix3 import Comparer, Coherencer

class SemNet(object):
  """Generates a coherent selection of labels from the database given a query.

  The class uses a simple artificial neural network approach similar to a
  semantic network.

  Public functions:
  ***Argument names are designed to be illustrative and may be different but
  similar in the actual function***
  loadDB(path)
  reset()
  setParams(queries, numTermsPerQuery, decay, initA, cycleLimit, activationMin,
            activationMax, threshold, reloadWeightMatrix)
  buildTermsToProc(numberOfTermsToTest)
  ask(default)
  askCycle(numTermsPerQuery, threshold, cycleLimit, numberOfTermsToTest, remRef,
           seedWithTop)
  
  """
  def __init__(self, pathToDatabase='D:\\pkb_matrix_sept2013-2-Sem',
               pathToIndices='D:\\pkb_k_index_nov2013-Sem.npy'):
    """Initializes the necessary components.

    Keyword arguments:
    pathToDatabase (str) -- the path to the database ***Windows ONLY***
    pathToIndices (str) -- path to label indices for entire database
                          ***Windows ONLY***

    db (dict of dicts) -- original Peekaboom database of co-occurrence
                          probabilities
    wdb (numpy matrix) -- co-occurrence between all pairs of labels
                          
    neg (float) -- base negative co-occurrence probability for labels with
                   no co-occurrence
    kindex (dict) -- label indices for entire database

    reset() initializes remaining base components:
    
    queries (list) -- list of terms that search is based off of
    numTerms (int) -- number of terms to get COUNTING the QUERY
    decay (float) -- rate of decay of node activation
    initA (float) -- base initialization of all nodes minus QUERY
    cycleLimit (int) -- max number of activation update cycles
    aMin(float) -- minimum activation allowed for a node
    aMax (float) -- maximum activation allowed for a node
    threshold (float) -- necessary average co-occurrence probability for
                         coherence

    """
    self.db = self.loadDB(pathToDatabase)
    self.neg = -0.14878295850321488
    self.kindex = self.loadDB(pathToWeightDatabase)
    self.wdb = numpy.ones([8372, 8372])*(self.neg)
    for key in self.kindex.items():
      for key2 in self.db[key[0]]:
        self.wdb[key[1]][self.kindex[key2]] = self.db[key[0]][key2]
    self.reset()

  def loadDB(self, path):
    """Loads nested dictionary of terms from path defaulting to my location.

    Keyword arguments:
    path (str) -- the path to the database ***Windows ONLY***

    """
    with open(path, 'r') as f:
      db = pickle.load(f)
    return db

  def loadW(self, index_):
    """Makes smaller numpy matrix of cooccurring terms with the query and query.

    Keyword arguments:
    index_ (list) -- list of cooccurring terms and query;
                     associates label with the index in the matrix

    """
    indices = [self.kindex[x] for x in index_]
    w = [[self.wdb[key][key2] for key2 in indices] for key in indices]
    w = numpy.array(w)
    return w

  def reset(self):
    """Resets relevant internal variables to start values."""
    self.queries = []
    self.numTerms = 0
    self.decay = 0.0
    self.initA = 0.0
    self.cycleLimit = 0
    self.aMin = 0.0
    self.aMax = 0.0
    self.threshold = 0.0

  def buildPool(self, queries):
    """Returns pool of co-occurring terms that will be searched in.

    Keyword Arguments:
    queries (list) -- the strings that should be queried

    """
    pool = set(queries)
    #iterating the algorithm below increases the edge depth for search
    pool |= set([x for y in pool for x in self.db[y]])
    #pool |= set([x for y in pool for x in self.db[y].keys()])
    return list(pool)


##Negative value is set to -0.14878295850321488 for 2-Sem
  def setParams(self, queries=['party'], numTerms=5, decay=0.1, initA=0.01,
                cycleLimit=500, aMin=-1.0, aMax=1.0,
                threshold=0.005, reloadW=True):
    """Sets up all the information needed to make a query.

    Keyword arguments:
    queries (list) -- list of terms that search is based off of
    numTerms (int) -- number of terms to get COUNTING the QUERY
    decay (float) -- rate of decay of node activation
    initA (float) -- base initialization of all nodes minus QUERY
    cycleLimit (int) -- max number of activation update cycles
    aMin(float) -- minimum activation allowed for a node
    aMax (float) -- maximum activation allowed for a node
    threshold (float) -- necessary average co-occurrence probability for
                         coherence
    index (list) -- list of cooccurring terms and query;
                    associates label with the index in the matrix
    ii (func) -- quick call to list index function to identify position of label
    qIndex (list) -- indices of all queries in self.index
    nodeLen (int) -- number of co-occurring terms with this query
    w (numpy matrix) -- smaller numpy matrix of weights for co-occurring terms
    aCur (numpy array) -- array of current activations
    aNex (numpy array) -- array of updated activations
    diff (numpy array) -- array of last ten changes in activation for all nodes
    
    """
    self.queries = queries
    self.numTerms = numTerms
    self.decay = decay
    self.initA = initA
    self.cycleLimit = cycleLimit
    self.aMin = aMin
    self.aMax = aMax
    self.threshold = threshold
    self.index = self.buildPool(self.queries)
    ii = self.index.index
    self.qIndex = [ii(q) for q in self.queries]
    self.nodeLen = len(self.index)
    if reloadW:
      self.w = self.loadW(self.index, self.neg, self.nodeLen)
    self.aCur = numpy.ones([self.nodeLen,])*self.initA
    self.aNex = numpy.array([])
    self.diff = numpy.array([])

  def buildTermsToProc(self, num):
    """Selects a random list of terms for processing.

    Keyword Arguments:
    num (int) -- number of terms to select

    """
    self.termsToProc = [x for x in random.sample(self.db.keys(), num)]

  def f(self, net, a, aMin, aMax):
    """Produces sigmoid change in activation based on current activation.

    Approximation of a sigmoid function.

    Keyword Arguments:
    net (numpy array) -- net input activation for each node
    a (numpy array) -- current activation of each node
    aMin (float) -- minimum activation allowed for a node
    aMax (float) -- maximum activation allowed for a node

    """
    return numpy.array([net[i]*(aMax-x) if net[i] > 0 else net[i]*(x-aMin) for
                        i, x in enumerate(a)])

  def update(self, a, decay, aMin, aMax):
    """Updates activation of each node based on input and decay.

    Keyword Arguments:
    a (numpy array) -- current activation of each node
    decay (float) -- rate of decay of node activation
    aMin (float) -- minimum activation allowed for a node
    aMax (float) -- maximum activation allowed for a node

    """
    return a*(1.0-decay)+self.f(numpy.dot(a, self.w), a, aMin, aMax)

  def adjust(self, a, qIndex, aMin, aMax):
    """Fixes over/underflow and changes in the query node activation.

    Keyword Arguments:
    a (numpy array) -- current activation of each node
    qIndex (list) -- indices of all queries in self.index
    aMin (float) -- minimum activation allowed for a node
    aMax (float) -- maximum activation allowed for a node

    """
    for ind in qIndex:
      a[ind] = 1.0
    temp = [aMax if x > aMax else x for x in a]
    temp = [aMin if x < aMin else x for x in temp]
    return numpy.array(temp)

  def ask(self, default=True):
    """Determines a coherent set by updating all nodes in parallel.

    The function returns when the cycleLimit is passed or the average change
    in activation over the past 10 updates is less than threshold.

    Keyword arguments:
    default (bool) -- set to False except when testing; causes defaults to run
    
    """
    if default:
      self.reset()
      self.setParams()
    self.aCur = self.adjust(self.aCur, self.qIndex, self.aMin, self.aMax)
    if len(self.aCur) <= 5:
      return self.index
    for x in xrange(self.cycleLimit):
      self.aNex = self.update(self.aCur, self.decay, self.aMin, self.aMax)
      self.aNex = self.adjust(self.aNex, self.qIndex, self.aMin, self.aMax)
      diff = sum([abs(val) for val in (self.aCur-self.aNex)])
##      print 'Iteration: ', x, ' Diff: ', diff, ' Avg Activ: ', \
##            sum(self.aNex)/len(self.aNex)
      self.diff = numpy.append(self.diff, diff)
      if len(self.diff) > 10:
        #Haven't tested but structure is simple.
        self.diff = self.diff[1:]
        if numpy.mean(abs(numpy.array(self.diff[:9]) -
                      numpy.array(self.diff[1:]))) <= self.threshold:
          return self.sortActive()
      self.aCur = numpy.copy(self.aNex)
    return self.sortActive()

  def sortActive(self):
    """Orders the labels based on activation.

    Nodes with equal activation are equally likely.
    
    """
    n = self.numTerms
    if len(self.index) <= n:
      return self.index
    sel = zip(self.index, self.aNex)
    #Randomizes labels with equal activiation.
    random.shuffle(sel)
    sel = [x for x in sel if x[0] not in self.queries]
    sel = sorted(sel, key=lambda x: x[1], reverse=True)
    qLen = len(self.queries)
    if len(sel) > n-qLen:
      sel = sel[:n-qLen]
    sel.extend([(x, 1.0) for x in self.queries])
    return [x[0] for x in sel]

  def topN(self, terms, numTerms):
    """Calculates top co-occurring terms and returns them.

    Keyword arguments:
    term (str) -- query or term that's being associated to
    numTerms (int) -- number of associated terms INCLUDING query
    
    """
    store = []
    for term in terms:
      temp = self.db[term]
      ls = sorted(temp, key=lambda label: temp[label],
                  reverse=True)[:numTerms-1]
      store.append(ls)
    if len(store) > 1:
      #Tries to take common terms and if that fails adds with random selection
      #Might be better to go with topN of highest common co-occurring terms
      #Better to sum co-occurrence with each query for each coTerm
      temp = set(store[0])
      for x in store[1:]:
        temp &= set(x)
      temp |= set(random.sample([x for y in store for x in y],
                                numTerms-len(temp)))
      store[0] = [x for x in temp]
    return store[0]

  def askCycle(self, nTerms=5, threshold=0.005, cycleLimit=500, num=False,
               seed=False):
    """Performs 'num' cycles of ask() and returns results.

    Keyword arguments:
    nTerms (int) -- number of associated terms in result INCLUDING query
    threshold (float) -- value at which a set of co-occurring terms are
                         considered coherent
    
    cycleLimit (int) -- max number of activation update cycles
    num (int) -- number of cycles; if False, terms are selected elsewhere
    seed (bool) -- indicates if the model should be seeded with the top-n
                   co-occurring terms

    """
    if num:
      self.buildTermsToProc(num)
    results = []
    for term in self.termsToProc:
      self.reset()
      self.setParams([term], numTerms=nTerms, threshold=threshold,
                     cycleLimit=cycleLimit)
      if seed:
        sample = self.topN(self.queries, self.numTerms)
        sampleInd = [self.index.index(x) for x in sample]
        for i in sampleInd:
          self.aCur[i] = 0.5
      results.append(self.ask(False))
    return results

 
    
