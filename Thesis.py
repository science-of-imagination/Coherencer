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
This marks an alternate approach for dealing with context in the computational
generation of images and 3D scenes.

Classes:
Coherencer() -- algorithms for generating coherent sets given query
Comparer()   -- test algorithms
Main()       -- faster use case of other two classes


"""

import random, pickle, numpy

class Coherencer(object):
  """Generates a coherent selection of labels from the database given a query.

  The class accounts for coherence by setting a threshold that the average
  co-occurrence for the selections must pass. It is an iterative approach.
  It also contains algorithms for Top N search.

  Public functions:
  ***Argument names are designed to be illustrative and may be different but
  similar in the actual function***
  loadDB(path)
  reset()
  setParams(queries, numTermsPerQuery, threshold, seedTrue)
  ask(default)
  runLoop(numOfTermsToTest, query, threshold)
  buildTermsToProc(numberOfTermsToTest)
  askCycle(numCycles, numTermsPerQuery, threshold, seedWithTop)
  calcTopN(numCycles, numTermsPerQuery)
  
  """
  def __init__(self, path='D:\\pkb_matrix_sept2013-2'):
    """Initializes the necessary components.

    Keyword arguments:
    path (str) -- the path to the database ***Windows ONLY***

    db (dict of dicts) -- original Peekaboom database of co-occurrence
                          probabilities
    termsToProc (list) -- random list of queries

    reset() initializes remaining components:
    queries (list) -- list of terms that search is based off of
    numTerms (int) -- number of terms to get COUNTING the QUERY
    threshold (float) -- necessary average co-occurrence probability for
                         coherence
    coTerms (list) -- terms that co-occur with the query
    selTerms (list) --  terms that were selected from the coTerms
    totals (numpy array) -- average co-occurrence probability for row and column
    best (tuple) -- the selection with the highest co-occurrence probability so
                    far

    """
    self.db = self.loadDB(path)
    self.reset()
    self.termsToProc = []

  def loadDB(self, path):
    """Loads nested dictionary of terms from path defaulting to my location.

    Keyword arguments:
    path (str) -- the path to the database ***Windows ONLY***

    """
    with open(path, 'r') as f:
      db = pickle.load(f)
    return db

  def reset(self):
    """Resets relevant internal variables to start values."""
    
    self.queries = []
    self.numTerms = 0
    self.threshold = 0.0
    self.coTerms = []
    self.selTerms = []
    self.totals = []
    self.best = (0, [])

  def buildPool(self, queries):
    """Returns pool of co-occurring terms that will be searched in.

    Keyword Arguments:
    queries (list) -- the strings that should be queried

    """
    pool = set(queries)
    #iterating the algorithm below increases the edge depth for search
    pool |= set([x for y in pool for x in self.db[y].keys()])
    pool = [x for x in pool if x not in queries]
    return pool

  def selectTerms(self, seed):
    """Gets seeded or random selection of terms from pool and returns list.

    Keyword Arguments:
    seed (bool) -- indicates if the model should be seeded with the top-n
                   co-occurring terms

    """
    if seed:
      sample = self.topN(self.queries, self.numTerms)
    else:
      sample = random.sample(self.coTerms, self.numTerms-len(self.queries))
    self.coTerms = [x for x in self.coTerms if x not in sample]
    return sample



  def setParams(self, queries=['party'], numTerms=5, threshold=0.23, seed=True):
    """Sets up all the information needed to make a query.

    Keyword arguments:
    queries (list) -- the strings that should be queried
    numTerms (int) -- number of terms to gather including queries
    threshold (float) -- value at which a set of co-occurring terms are
                         considered coherent
    seed (bool) -- indicates if the model should be seeded with the top-n
                   co-occurring terms
    
    """
    self.queries = queries
    self.selTerms.extend(queries)
    self.numTerms = numTerms
    self.threshold = threshold
    self.coTerms = self.buildPool(self.queries)
    self.selTerms.extend(self.selectTerms(seed))
    self.fill()


  def fill(self):
    """Returns a co-occurrence matrix with the diagonal masked.

    This builds a numpy matrix of the co-occurrence probabilities. The diagonal
    is masked to ignore term co-occurrence with itself.
    
    """
    self.matrix = numpy.zeros([self.numTerms, self.numTerms])
    for i, term in enumerate(self.selTerms):
      for i2, term2 in enumerate(self.selTerms):
        try:
          val = self.db[term][term2]
        except KeyError:
          pass
        else:
          self.matrix[i][i2] = val
        try:
          val = self.db[term2][term]
        except KeyError:
          pass
        else:
          self.matrix[i2][i] = val
    self.matrix = numpy.ma.masked_equal(self.matrix, -1)

  #generic of above function:
  def mat(self, terms):
    """Generic version of fill().

    In this version, the relevant terms must be input directly. Not used in the
    current implementation

    Keyword Arguments:
    terms (list) -- the current selected terms
    
    """
    matrix = numpy.zeros([len(terms), len(terms)])
    for i, term in enumerate(terms):
      for i2, term2 in enumerate(terms):
        try:
          val = self.db[term][term2]
        except KeyError:
          pass
        else:
          matrix[i][i2] = val
        try:
          val = self.db[term2][term]
        except KeyError:
          pass
        else:
          matrix[i2][i] = val
    matrix = numpy.ma.masked_equal(matrix, -1)
    return matrix

  def assess(self):
    """Returns true if mean co-occurrence of matrix is greater than threshold.

    The algorithm tests the mean co-occurrence probability of all pairs of terms
    against the threshold. It also checks whether this set is better than the
    current best set and stores it as such if it is.
    
    """
    self.totals = numpy.sum(self.matrix, 0) + numpy.sum(self.matrix, 1)
    total = numpy.mean(self.matrix)
    if total > self.best[0]:
      self.best = (total, self.selTerms)
    if total > self.threshold:
      return True
    else:
      return False



  def filter_(self):
    """Determines term with the lowest row and column co-occurrence and removes.

    This algorithm removes the most unrelated term and then gets another term
    from the pool. This term is then removed from the pool and stored in
    the list of selected terms. The function then returns True. If there are no
    further terms in the pool, it returns False.
    
    """
    vals = [x for x in self.totals]
    r = False
    vals = zip(self.selTerms, vals)
    vals = sorted(vals, key=lambda x: x[1], reverse=r)
    self.selTerms.remove(vals[0][0])
    try:
      sel = random.choice(self.coTerms)
    except IndexError:
      return False
    else:
      self.coTerms.remove(sel)
      self.selTerms.append(sel)
      return True

  def ask(self, default=True):
    """Determines a coherent set of terms or the most coherent set then returns.

    The term with the most co-occurring terms has less than 3000 so the for
    loop effectively makes sure that the entire pool is searched.

    Keyword arguments:
    default (bool) -- set to False except when testing; causes defaults to run
    
    """
    if default:
      self.reset()
      self.setParams()
    for x in xrange(5000):
      if self.assess():
        return self.selTerms
      else:
        if self.filter_():
          self.fill()
        else:
          return self.best[1]

  def runLoop(num, query, t=0.23):
    """Runs the model once with the given parameters.

    This function is really for the SOILIE model.

    Keyword Arguments:
    num (int) -- the number of terms to be selected
    query (string) -- a single string query
    t (float) -- the co-occurrence threshold

    """
    self.reset()
    self.setParams([query], num, t)
    terms = self.ask(False)
    return terms

  def buildTermsToProc(self, num):
    """Selects a random list of terms for processing.

    Keyword Arguments:
    num (int) -- number of terms to select

    """
    self.termsToProc = [x for x in random.sample(self.db.keys(), num)]

  def askCycle(self, numTerms=5, threshold=0.23, #sThreshold=10,
               seed=True, num=False):
    """Performs 'num' cycles of ask() and returns results.

    Keyword arguments:
    numTerms (int) -- number of associated terms in result INCLUDING query
    threshold (float) -- value at which a set of co-occurring terms are
                         considered coherent
    seed (bool) -- indicates if the model should be seeded with the top-n
                   co-occurring terms
    num (int) -- number of cycles; if False, terms are selected elsewhere

    """
    if num:
      self.buildTermsToProc(num)
    results = []
    for term in self.termsToProc:
      if len(self.db[term].keys()) < numTerms-1:
        temp = [term]
        temp.extend(self.db[term].keys())
        results.append(temp)
      else:
        self.setParams([term], numTerms, threshold, seed)
        results.append(self.ask(False))
      self.reset()
    return results

  def topN(self, terms, numTerms):
    """Calculates top co-occurring terms and returns them.

    Keyword arguments:
    terms (list) -- queries or terms that are being associated to
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

  def calcTopN(self, numTerms=5, num=False):
    """Performs 'num' cycles of top co-occurrence search and returns results.

    Keyword arguments:
    numTerms (int) -- number of associated terms in result INCLUDING query
    num (int) -- number of cycles; if False, terms are selected elsewhere
    
    """
    if num:
      self.buildTermsToProc(num)
    results2 = []
    for term in self.termsToProc:
      ls = self.topN([term], numTerms)
      ls.append(term)
      results2.append(ls)
    return results2

class Comparer(object):
  """Tests if image in 2nd database has matching terms to coherencer output.

  Public functions:
  ***Argument names are designed to be illustrative and may be different but
  similar in the actual function***
  loadDB(path)
  test(listOfResults)
  compare(singleResult)

  """

  def __init__(self):
    """Loads the database and stores the terms for quick access.

    """
    self.db = self.loadDB()
    self.terms = self.db.keys()

  def loadDB(self, path="D:\\pkb_by_lbl_filter5_sept2013"):
    """Loads database from the specified path defaulting to my location.

    """
    with open(path, 'r') as f:
      db = pickle.load(f)
    return db

  def test(self, results):
    """Takes output and returns total number of sets with corresponding image.

    Keyword arguments:
    results (list of lists) -- list of 'coherent' terms from one of the
                               functions
    
    """
    c = self.compare
    total = 0
    for comb in results:
      if c(comb):
        total +=1
    return total

  def compare(self, comb):
    """Checks if there is are images that have this 'comb'ination of terms

    Keyword arguments:
    comb (list) -- a result set of terms for a particular query

    """
    if comb == False or comb == None:
      return False
    theSet = []
    for term in comb:
      if term not in self.terms:
        return False
    theSet.extend(self.db[comb[0]])
    comb = comb[1:]
    for term in comb:
      imgs = self.db[term]
      theSet = [t for t in theSet if t in imgs]
    if len(theSet) > 0:
      return True
    else:
      return False


