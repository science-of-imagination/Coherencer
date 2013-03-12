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
Comparer() -- test algorithms
Main() -- faster use case of other two classes

"""

import os
import random
import pickle

class Coherencer(object):
  """Designed to select co-occurring terms from the database given a query.

  The class accounts for coherence by setting a threshold that the average
  co-occurrence for the selections must pass. It is an iterative approach.
  It also contains algorithms for Top N search and two random searches:
  global (full random) and local (random of terms that co-occur).

  Public functions:
  ***Argument names are designed to be illustrative and may be different but
  similar in the actual function***
  reset()
  loadDB(path)
  buildTermsToProc(numberOfTermsToTest)
  setParams(queries, numTermsPerQuery, threshold, matrixDegree)
  ask(default)
  askCycle(numCycles, numTermsPerQuery, threshold, seedWithTop)
  calcTopN(numCycles, numTermsPerQuery)
  rndmN(numCycles, numTermsPerQuery, fullRandom)
  
  """

  def __init__(self):
    """Initializes all the necessary components.

    db (dict of dicts) -- original Peekaboom database of frequency counts
    queries (list) -- list of terms that co-occurrence search is based off of
    numTerms (int) -- number of terms to get COUNTING THE QUERY
    threshold (float) -- necessary average correlation value for co-occurrence
    set acceptance
    matrix (dict of dicts) -- correlation matrix for the selected terms and the
    query
    matrixDegree (int) -- maximum degree of the matrix needed for the coherence
    check (i.e., 2D, 3D, 4D matrix, etc.) - Not used as yet.
    coTerms (list) -- terms that co-occur with the query
    selTerms (list) --  terms that were selected from the coTerms
    remTerms (list) --  terms that were removed in order to raise the average
    correlation value and increase coherence
    termTotals (dict of lists) -- average correlation for the row and column
    corresponding to each term in a list [row, column]
    totalAvg (float) --  total average correlation for the current selection
    termsToProc (list) -- random list of queries
    
    """
    self.db = self.loadDB()
    self.reset()
    self.termsToProc = []

  def reset(self):
    """Resets all internal variables to start values."""
    self.queries = []
    self.numTerms = 0
    self.threshold = 0.0
    self.matrix = {}
    self.matrixDegree = 0
    self.coTerms = []
    self.selTerms = []
    self.remTerms = []
    self.termTotals = {}
    self.totalAvg = 0.0
    
  def loadDB(self, path="D:\\pkb_matrix_train_split1"):
    """Loads nested dictionary of terms from path defaulting to my location.

    Keyword arguments:
    path (str) -- the path to the database ***Windows ONLY***

    """
    with open(path, 'r') as f:
      db = pickle.load(f)
    return db

  def buildTermsToProc(self, num):
    """Selects a random list of terms for processing of size 'num'."""
    self.termsToProc = []
    keys = self.db.keys()
    for x in range(num):
      self.termsToProc.append(random.choice(keys))
    
  def setParams(self, queries=['party'], numTerms=5, threshold=0.27,
                matrixDegree=2):
    """Sets up all the information needed to make a query.

    Keyword arguments:
    queries (list) -- the strings that should be queried
    numTerms (int) -- number of terms to gather including queries
    threshold (float) -- value at which a set of co-occurring terms are
    considered coherent; 0.27 was tested to be best with Peekaboom
    matrixDegree -- the depth of the database and corresponding comparison;
    leave at 2; designed for future functionality (triplet or higher testing)
    
    """
    self.queries = queries
    self.selTerms.extend(queries)
    self.numTerms = numTerms
    self.threshold = threshold
    self.matrixDegree = matrixDegree
    for query in queries:
      self.coTerms.extend(self.db[query].keys())

  def selectTerms(self):
    """Randomly selects a number of terms from co-occurring terms.

    ***Currently only used for replacement; not initial seed.***
    Limits number of terms selected for speed.
    Another variable that can be manipulate for future paper.

    """
    #I use xrange in order to optimize for queries with coTerms < numTerms
    #and/or queries with low cooccurrence average and hence require multiple
    #iterations of selectTerms(). The subtraction is so that this function can
    #be reused for selecting a new term when filtering.
    for x in xrange((self.numTerms-len(self.selTerms))):
      if len(self.coTerms) <= 0:
        return False
      sel = random.choice(self.coTerms)
      self.selTerms.append(sel)
      self.coTerms.remove(sel)
    return True

  def setupTermTotals(self):
    """Sets up the dictionary for the averages of each term.

    Each term has a list whose indices correspond to the average the term gets
    in each corresponding position in a matrix call.
    e.g., if a value call is self.matrix[term1][term2][term3]
    then, for a particular term, its zero index would have the average it got
    of all values that occur when it is in the place of term1 its first index
    would have the average for all the values in the place of term2 and its
    second index would have the average for term3.
    
    """
    d = self.matrixDegree
    self.termTotals = dict((term, [0 for x in range(d)])
                           for term in self.selTerms)

  def testTuple(self, val):
    """Ensures you only get relevant value.

    This is an artifact of the way I made the original database:
    I kept both the frequency of the co-occurrence as well as the average in a
    tuple. The thought was that the base frequency might be another filtering
    point: A frequency of 0.1 is slightly different if it occurred 1 / 10
    versus 100 / 1000.

    Keyword arguments:
    val (tuple or int) -- the value corresponding to a database query
    
    """
    try:
      return val[1]
    except TypeError:
      return val

  def buildMatrix(self, theDict, val, degree):
    """Builds the matrix of correlations for the current selected terms.

    The co-occurrence of a term with itself is set to 0 and ignored in further
    calculations.

    Keyword arguments:
    theDict (dict) -- holds a dictionary at various depths of the matrix in
    order to build the new matrix. The starting value is self.matrix = {}
    val (dict) --  holds a dictionary at various depths of the original
    database in order to get the corresponding value. The starting value is
    self.db
    degree (int) -- holds the remaining number of degrees lower than the
    current degree of the matrix. The degree must correspond with the given
    database or all of the values will come up as 0. The starting value is
    self.matrixDegree
    """
    for term in self.selTerms:
      #This is how I am testing to see if there is another layer to the
      #database. If there is, the current contents of val will be a dict.
      if type(val) == dict:
        #Use get() because not all combinations exist in the database.
        newVal = val.get(term, 0)
      else:
        newVal = val
      #Checks to see if the matrix being built has another level of depth
      #necessary.
      if degree > 1:
        #In order to provide another level, the current term is made a
        #dictionary key.
        theDict[term] = {}
        #And, the function is recursively called with this new key, the new
        #value that corresponds in the database (either a dictionary if the
        #terms co-occur or 0), and with the remaining degrees of the matrix
        #left to be built.
        self.buildMatrix(theDict[term], newVal, (degree-1))
      else:
        #If there are no more degrees left to be built, assign the value that
        #was grabbed from the database.
        theDict[term] = self.testTuple(newVal)

  def totalTermVals(self, theDict, degree):
    """Recursive function that adds all values for terms in matrix.

    Keyword arguments:
    theDict (dict of dicts) -- increasingly small pieces of the matrix starting
    with the whole thing
    degree (int) -- the depth of the matrix current in; starts equal to
    matrixDegree
    
    """
    #This makes sure that the correct cell is filled in the term's list of
    #averages.
    avgsCell = self.matrixDegree-degree
    total = 0
    if degree > 1:
      for term in self.selTerms:
        #This makes the recursive call that will both get the value and add it
        #to each corresponding level for each term the value corresponds to in
        #the multidimensional matrix.
        val = self.totalTermVals(theDict[term], (degree-1))
        self.termTotals[term][avgsCell] = self.termTotals[term][avgsCell]+val
        total += val
      return total
    else:
      for term in self.selTerms:
        #This adds the co-occurrence value to last term in the recursive
        #dictionary call.
        self.termTotals[term][avgsCell] = \
         self.termTotals[term][avgsCell]+theDict[term]
        total += theDict[term]
      return total

  def setTotalAvg(self):
    """Computes the total average of termTotals.

    """
    num = self.numTerms
    #Multiply by the matrix degree in order to accommodate using the term in
    #each dimension e.g., columns and rows in 2D
    #num-1 because a 4x4 matrix, ignoring diagonals is equivalent to a 4x3
    num = num*(num-1)*self.matrixDegree
    self.totalAvg = sum([x for term in self.selTerms
                         for x in self.termTotals[term]])
    self.totalAvg /= num

  def filterMatrix(self, regulatoryFn):
    """Determines outlier term (currently lowest) and removes it.

    It then sets up the data for a new cycle by resetting the matrix and total
    variables.

    Keyword arguments:
    regulatoryFn (function) -- takes a function that returns a term from a list
    for removal such that the matrix moves towards the threshold in the
    appropriate fashion.
    
    """
    rvrsdAvgs = {}
    #I believe this method will naturally overwrite any terms with the same
    #probability. This is advantageous because we only get one term, but sloppy
    #because it's unpredictable which term is kept (the last one?). There is a
    #way to get around this if I don't collapse the row and col term averages
    #and, instead, turn them into a tuple. The tuple could then be used as a
    #dictionary key that is a lot less likely to occur more than once.
    for term in self.termTotals:
      rvrsdAvgs[((self.termTotals[term][0]+self.termTotals[term][1])
                 / 2)] = term
    remAvg = regulatoryFn(rvrsdAvgs.keys())
    remTerm = rvrsdAvgs[remAvg]
    for term in self.selTerms:
      try:
        del self.matrix[term][remTerm]
      except KeyError:
        pass
    del self.matrix[remTerm]
    self.selTerms.remove(remTerm)
    self.remTerms.append(remTerm)
    self.matrix = {}
    self.termTotals = {}
    self.totalAvg = 0
    
    

  def ask(self, default=True):
    """Determines coherent set of terms and returns or False if impossible.

    All of the functions, besides the initial setup, are strung together here
    and looped in order to find the selection of terms that meet the threshold.

    Keyword arguments:
    default (bool) -- set to False except in testing; causes defaults to run
    """
    if default:
      self.reset()
      self.setParams()
    #I switched to a for statement with lazy sequence generation over a tail
    #call to ask() in order to optimize and protect from stack overflow.
    for x in xrange(5000):
      if self.selectTerms():
        self.buildMatrix(self.matrix, self.db, self.matrixDegree)
        self.setupTermTotals()
        self.totalTermVals(self.matrix, self.matrixDegree)
        self.setTotalAvg()
        if self.totalAvg >= self.threshold:
          return self.selTerms
        else:
          #Currently, the class only regulates by removing minimum values.
          self.filterMatrix(min)
      else:
        break
    return False
    #print "There is no combination of terms that satisfies this threshold."
    
  def askCycle(self, nTerms, t, seed, num=False):
    """Performs 'num' cycles of ask() and returns results.

    Keyword arguments:
    num (int) -- number of cycles; if False, terms are selected elsewhere
    nTerms (int) -- number of associated terms in result INCLUDING query
    t (float) -- threshold
    seed (bool) -- determines whether to seed initial results with top
    co-occurrences of query

    """
    if num:
      self.buildTermsToProc(num)
    results = []
    #avgs = []
    for term in self.termsToProc:
      self.setParams([term], numTerms=nTerms, threshold=t)
      #To seed with topN
      if seed:
        self.selTerms.extend(self.topN(term, nTerms))
      results.append(self.ask(False))
      #avgs.append(self.totalAvg)
      self.reset()
    return results#, avgs

  def calcTopN(self, nTerms, num=False):
    """Performs 'num' cycles of top co-occurrence search and returns results.

    Keyword arguments:
    num (int) -- number of cycles; if False, terms are selected elsewhere
    nTerms (int) -- number of associated terms in result EXCLUDING query
    
    """
    if num:
      self.buildTermsToProc(num)
    results = []
    for term in self.termsToProc:
      ls = self.topN(term, nTerms)
      ls.append(term)
      results.append(ls)
    return results

  def topN(self, term, nTerms):
    """Calculates top co-occurring terms and returns them.

    Keyword arguments:
    term (str) -- query or term that's being associated to
    nTerms (int) -- number of associated terms EXCLUDING query
    
    """
    temp = self.db[term]
    ls = sorted(temp, key=lambda label: temp[label], reverse=True)[:nTerms]
    return ls

  def rndmN(self, nTerms, fullRandom, num=False):
    """Performs 'num' cycles of random search and returns results.

    Keyword arguments:
    num (int) -- number of cycles; if False, terms are selected elsewhere
    nTerms (int) -- number of associated terms in result EXCLUDING query
    fullRandom (bool) -- make it True for all possible terms and False for
    co-occurring terms

    """
    if num:
      self.buildTermsToProc(num)
    results = []
    for term in self.termsToProc:
      k = self.db[term].keys()
      ls = []
      if fullRandom:
        sample = self.db.keys()
        sample.remove(term)
      else:
        sample = self.db[term].keys()
      if len(k) > nTerms:
        ls = random.sample(sample, nTerms)
      else:
        ls = random.sample(sample, len(k))
      ls.append(term)
      results.append(ls)
    return results


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

  def loadDB(self, path=os.path.join("D:\\", "pkb_by_lbl_test_split1")):
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

class Main(object):
  """Runs all coherence algorithms and prints output.

  Public functions:
  run(numterms, numQueries, numTimes)

  """
  def __init__(self):
    """Initializes both classes.

    """
    self.coh = Coherencer()
    self.com = Comparer()

  def run(self, numTerms, numQueries, numTimes):
    """Runs each of the algorithms.

    Keyword arguments:
    numTerms (int) -- number of associated terms per query
    numQueries (int) -- number of individual queries per round
    numTimes (int) -- number of rounds

    """
    f = '{:<15}{:<15}{:<15}{:<15}{:<15}'
    print f.format('Full Random', 'Part Random', 'Top N', 'New', 'New Top N')
    for x in range(numTimes):
      self.coh.buildTermsToProc(numQueries)

      resRnd1 = self.coh.rndmN(numTerms, True)
      resRnd2 = self.coh.rndmN(numTerms, False)
      resTopN = self.coh.calcTopN(numTerms)
      resNew1 = self.coh.askCycle(numTerms+1, 0.37, False) 
      resNew2 = self.coh.askCycle(numTerms+1, 0.37, True) #0.36 for n = 4

      totRnd1 = self.com.test(resRnd1)
      totRnd2 = self.com.test(resRnd2)
      totTopN = self.com.test(resTopN)
      totNew1 = self.com.test(resNew1)
      totNew2 = self.com.test(resNew2)

      self.coh.reset()

      print f.format(totRnd1, totRnd2, totTopN, totNew1, totNew2)

if __name__ == '__main__':
  m = Main()
#  m.run(6, 1000, 20)

##########
##For Testing purposes

  t = 0.25
  while t < 0.45:
    a = []
    for x in range(100):
      r = m.coh.askCycle(4, t, True, 1000)
      a.append(m.com.test(r))
    print t, sum(a)/100, max(a), min(a)
    t += 0.01
##########
