# START GP INTEGRATION CODE*************************************************************************************************************************************
import deap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.tools import crossover

import operator
import math
import random
import string
import copy
import ast
from collections import defaultdict
from functools import partial, wraps
from exstracs_classifier import Classifier
from exstracs_constants import *
from exstracs_pareto import *


# -----------------------------------------------------------------------
# Set up DEAP stuff
# -----------------------------------------------------------------------


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


# Change the evaluate function as needed.
# This takes in one parameter.
def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = self.toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x ** 4 - x ** 3 - x ** 2 - x) ** 2 for x in points)
    return math.fsum(sqerrors) / len(points),


# STOP GP INTEGRATION CODE*************************************************************************************************************************************


# START GP INTEGRATION CODE*************************************************************************************************************************************
class Tree:
    def __init__(self, tree=None, exploreIter=None):

        # Initialize DEAP GP environment for generating a single tree---------------------------------------------------------------
        # addPrimitive adds different operators (operator, num. of arguments)
        # Written to approximate x**4 - x**3 - x**2 - x)**2

        # PrimitiveSet constructor takes in num. arguments - default named ARGx where x is index
        pset = gp.PrimitiveSet("MAIN", int(cons.env.formatData.numAttributes))
        # pset = gp.PrimitiveSet("MAIN", 11)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(max, 2)
        pset.addPrimitive(operator.lt, 2)
        pset.addPrimitive(operator.gt, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protectedDiv, 2)
        pset.addPrimitive(operator.neg, 1)
        pset.addPrimitive(math.cos, 1)
        pset.addPrimitive(math.sin, 1)
        # addEphemeralConstant(name, function)
        randomName = str(exploreIter) + str(time.time()) + str(random.randint(-10000, 10000))
        # print('New Tree: '+str(randomName))
        pset.addEphemeralConstant(randomName, lambda: random.randint(-5, 5))
        # pset.addEphemeralConstant(randomName, lambda: random.randint(-5,5))
        # pset.addEphemeralConstant("rand101", lambda: random.randint(-5,5))

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        # an expression that generates either full trees (all leaves on the same level between 2 and 5)
        # or grown trees with leaves on different levels.
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)

        # generating test points of values from -1 to 1, with 0.1 intervals
        self.toolbox.register("evaluate", evalSymbReg, points=[x / 10. for x in range(-10, 10)])
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=2, max_=4)
        self.toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        # methods used to reference terminal and primitive objects

        term_list = list(pset.terminals.values())[0]
        self.terminals = {}
        for terminal in term_list:
            if isinstance(terminal, deap.gp.Terminal):
                self.terminals[terminal.name] = terminal

        primitives = list(pset.primitives.values())[0]
        self.pset = pset
        # ---------------------------------------------------------------------------------------------------------------------
        # STOP GP INTEGRATION CODE*************************************************************************************************************************************
        self.condition = None

        self.errorSum = 0
        self.errorCount = 0
        self.phenCount = 0  # number of times phenProb added to count for testing reasons
        self.phenSum = 0  # sum of phenotype probability calculation values for continuous variables
        self.totalFreq = 1
        self.id = ''.join(random.choice(string.ascii_lowercase) for i in range(7))
        self.one_count = 0
        self.zero_count = 0
        self.isTree = True
        # Major Parameters --------------------------------------------------
        self.phenotype = None  # Class if the endpoint is discrete, and a continuous phenotype if the endpoint is continuous
        self.phenotype_RP = None  # NEW - probability of this phenotype occurring by chance.

        # Fitness Metrics ---------------------------------------------------
        self.accuracy = 0.0  # Classifier accuracy - Accuracy calculated using only instances in the dataset which this rule matched.
        self.accuracyComponent = None  # Accuracy adjusted based on accuracy by random chance, and transformed to give more weight to early changes in accuracy.
        self.coverDiff = 1  # Number of instance correctly covered by rule beyond what would be expected by chance.
        self.indFitness = 0.0  # Fitness from the perspective of an individual rule (strength/accuracy based)
        self.relativeIndFitness = None
        self.fitness = 1  # CHANGED: Original self.fitness = cons.init_fit
        # Classifier fitness - initialized to a constant initial fitness value

        self.numerosity = 1  # The number of rule copies stored in the population.  (Indirectly stored as incremented numerosity)
        self.aveMatchSetSize = None  # A parameter used in deletion which reflects the size of match sets within this rule has been included.
        self.deletionVote = None  # The current deletion weight for this classifier.

        # Experience Management ---------------------------------------------
        self.epochComplete = False  # Has this rule existed for a complete epoch (i.e. a cycle through training set).

        # Fitness Metrics ---------------------------------------------------
        # self.freqComponent = None

        # Fitness Sharing
        # self.adjustedAccuracy = 0.0
        # self.relativeAccuracy = 0.0
        self.lastMatch = 0

        self.lastFitness = 0.0
        self.sumIndFitness = 1.0  # experimental
        self.partOfCorrect = True
        self.lastMatchFitness = 1.0
        # self.isMatchSetFitness = False #For Debugging
        self.aveRelativeIndFitness = None
        self.matchedAndFrontEstablished = False
        self.totalFreq = 1

        # START GP INTEGRATION CODE*************************************************************************************************************************************
        if tree == None:
            self.timeStampGA = 0  # Time since rule last in a correct set.
            self.initTimeStamp = 0  # Iteration in which the rule first appeared.
            self.aveMatchSetSize = 1

            self.lastMatch = 0  # Experimental - for brief fitness update
            # Classifier Accuracy Tracking --------------------------------------
            self.matchCount = 0  # Known in many LCS implementations as experience i.e. the total number of times this classifier was in a match set
            self.correctCount = 0  # The total number of times this classifier was in a correct set
            self.matchCover = 0  # The total number of times this classifier was in a match set within a single epoch. (value fixed after epochComplete)
            self.correctCover = 0  # The total number of times this classifier was in a correct set within a single epoch. (value fixed after epochComplete)

            # Covering sets initially overly optimistic prediction values - this takes care of error with prediction which previously had only zero value fitness and indFitness scores for covered rules.
            self.indFitness = 1.0
            self.fitness = 1.0

            tree = creator.Individual(self.toolbox.population(n=1)[0])  # tree actually created
            self.form = tree



        elif exploreIter != None:
            # print "New Tree Created"
            self.treeClone(tree, exploreIter)
            # print self.initTimeStamp

        else:
            print("Error with initializing")

        # get specified tree args
        tree_args = set()
        for idx, node in enumerate(self.form):
            if isinstance(node, deap.gp.Terminal):
                if node.name.split('G')[0] == 'AR':
                    nodeAtt = int(float(node.name.split('G')[1]))
                    tree_args.add(nodeAtt)

        self.specifiedAttList = list(tree_args)
        # STOP GP INTEGRATION CODE*************************************************************************************************************************************

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER CONSTRUCTION METHODS
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------`
    # START GP INTEGRATION CODE*************************************************************************************************************************************
    # clone tree
    def treeClone(self, clOld, exploreIter):
        """  Constructs an identical Classifier.  However, the experience of the copy is set to 0 and the numerosity
        is set to 1 since this is indeed a new individual in a population. Used by the genetic algorithm to generate
        offspring based on parent classifiers."""
        self.form = copy.deepcopy(clOld.form)
        self.phenotype = None
        self.phenotype_RP = None

        self.timeStampGA = exploreIter  # consider starting at 0 instead???
        self.initTimeStamp = exploreIter
        self.lastMatch = exploreIter
        self.aveMatchSetSize = copy.deepcopy(clOld.aveMatchSetSize)
        self.fitness = clOld.fitness  # Test removal
        self.accuracy = clOld.accuracy
        self.relativeIndFitness = clOld.relativeIndFitness
        self.indFitness = clOld.indFitness
        self.sumIndFitness = clOld.sumIndFitness
        # self.sumPreFitnessNEC = 1.0
        # self.sumPreFitnessEC = 1.0
        self.accuracyComponent = clOld.accuracyComponent
        self.matchCount = 1  # Known in many LCS implementations as experience i.e. the total number of times this classifier was in a match set
        self.correctCount = None  # The total number of times this classifier was in a correct set
        self.matchCover = 1  # The total number of times this classifier was in a match set within a single epoch. (value fixed after epochComplete)
        self.correctCover = None  # The total number of times this classifier was in a correct set within a single epoch. (value fixed after epochComplete)

    def updateClonePhenotype(self, phenotype):
        if phenotype == self.phenotype:
            self.correctCount = 1
            self.correctCover = 1
        else:
            self.correctCount = 0
            self.correctCover = 0

            # STOP GP INTEGRATION CODE*************************************************************************************************************************************

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def match(self, state):
        return True

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Set Phenotype - GP TREE!!
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def setPhenotype(self,
                     args):  # Ensure that for every instance we recalculate the tree phenotype prediction.  It never stays fixed.
        args = [int(i) for i in args]
        try:
            func = self.toolbox.compile(expr=self.form)
        except:
            print("Self form: " + str(self.form))
            print("String: " + str(self.tree_string()))
            print("Len String: " + str(len(self.form)))
            print("Args: " + str(args))
            raise NameError("Problem with args")
        dataInfo = cons.env.formatData

        # -------------------------------------------------------
        # BINARY PHENOTYPE - ONLY
        # -------------------------------------------------------

        if dataInfo.discretePhenotype and len(dataInfo.phenotypeList) == 2:
            if func(*args) > 0:
                self.phenotype = '1'
                if not self.epochComplete:
                    self.one_count += 1
            else:
                self.phenotype = '0'
                if not self.epochComplete:
                    self.zero_count += 1

            if not self.epochComplete:  # Not sure where Ben came up with this but it seems to makes reasonable sense.  May want to examie more carefully.
                self.phenotype_RP = ((cons.env.formatData.classProportions['1'] * self.one_count) + (
                            cons.env.formatData.classProportions['0'] * self.zero_count)) / (
                                                self.one_count + self.zero_count)
                # For trees there could be one uniform random chance. Ultimately we want to use balanced accuracy for trees (do we do better than by chance) but for now we will just use 0.5 for simplicity.

        # -------------------------------------------------------
        # MULTICLASS PHENOTYPE
        # -------------------------------------------------------
        elif dataInfo.discretePhenotype and not len(dataInfo.phenotypeList) == 2:

            if func(*args) < 0:  # lowest class
                self.phenotype = dataInfo.phenotypeList[0]
            elif func(*args) >= len(dataInfo.phenotypeList) - 1:  # lowest class
                self.phenotype = dataInfo.phenotypeList[len(dataInfo.phenotypeList) - 1]
            else:  # one of the middle classes
                count = 1
                notfoundClass = True
                while notfoundClass:
                    if func(*args) < count and func(*args) >= count - 1:
                        self.phenotype = dataInfo.phenotypeList[count]
                        notfoundClass = False
                    if count > len(dataInfo.phenotypeList):
                        notfoundClass = False
                        print("ERROR:setPhenotype in tree: Failed to find class")
                    count += 1

            if not self.epochComplete:  # Not sure where Ben came up with this but it seems to makes reasonable sense.  May want to examie more carefully.
                self.phenotype_RP = 0.5
        # -------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        # -------------------------------------------------------
        else:  # ContinuousCode #########################
            self.phenotype = func(*args)
            if not self.epochComplete:  # Not sure where Ben came up with this but it seems to makes reasonable sense.  May want to examie more carefully.
                self.phenotype_RP = 0.5

    def setPhenProb(self):
        pass

    def calcPhenProb(self, error):
        if self.epochComplete:
            return
        phenRange = [self.phenotype - error, self.phenotype + error]
        count = 0
        ref = 0
        #         print cons.env.formatData.phenotypeRanked
        #         print self.phenotype
        while ref < len(cons.env.formatData.phenotypeRanked) and cons.env.formatData.phenotypeRanked[ref] <= phenRange[
            1]:
            if cons.env.formatData.phenotypeRanked[ref] >= phenRange[0]:
                count += 1
            ref += 1
        self.phenSum += count / float(cons.env.formatData.numTrainInstances)
        self.phenCount += 1

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GENETIC ALGORITHM MECHANISMS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # START GP INTEGRATION CODE*************************************************************************************************************************************
    def cxOnePoint(ind1, ind2):
        """Randomly select in each individual and exchange each subtree with the
        point as root between each individual.
        :param ind1: First tree participating in the crossover.
        :param ind2: Second tree participating in the crossover.
        :returns: A tuple of two trees.
        """
        if len(ind1) < 2 or len(ind2) < 2:
            # No crossover on single node tree
            return ind1, ind2

        # List all available primitive types in each individual
        types1 = defaultdict(list)
        types2 = defaultdict(list)
        if ind1.root.ret == __type__:
            # Not STGP optimization
            types1[__type__] = range(1, len(ind1))
            types2[__type__] = range(1, len(ind2))
            common_types = [__type__]
        else:
            for idx, node in enumerate(ind1[1:], 1):
                types1[node.ret].append(idx)
            for idx, node in enumerate(ind2[1:], 1):
                types2[node.ret].append(idx)
            common_types = set(types1.keys()).intersection(set(types2.keys()))

        if len(common_types) > 0:
            type_ = random.choice(list(common_types))

            index1 = random.choice(types1[type_])
            index2 = random.choice(types2[type_])

            slice1 = ind1.searchSubtree(index1)
            slice2 = ind2.searchSubtree(index2)
            ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

        return ind1, ind2

    # does not return clone
    # get this to return boolean
    def uniformCrossover(self, cl, state, phenotype):  # If called, assumes self is a tree!!
        if cl.isTree:  # BOTH TREES - use build in DEAP crossover mechanism
            # print self.form
            # print "------------------------------------------------------"
            # print cl.form
            # print "------------------------------------------------------"
            orig_form = copy.deepcopy(self.form)
            other = copy.deepcopy(cl.form)
            try:
                gp.cxOnePoint(self.form, cl.form)  # method defined just above!
            except:
                print('Single point crossover failure')
                print("Self form: " + str(self.form))
                print("Self String: " + str(self.tree_string()))
                print("Other form: " + str(cl.form))
                print("Other string: " + str(cl.tree_string()))
                print(5 / 0)

            try:
                self.test_arity()
            except:
                print("Original Form: " + str(orig_form))
                print("Original String: " + str(self.tree_string_form(orig_form)))
                print("Original Other: " + str(other))
                print("Original Other String: " + str(self.tree_string_form(other)))
                print("New Form: " + str(self.form))
                print("New String: " + self.tree_string())
                print("New Other: " + str(cl.form))
                print("New Other String: " + str(cl.tree_string()))
                raise NameError("Crossover Returning invalid")

            tree_args = set()
            for idx, node in enumerate(self.form):
                if isinstance(node, deap.gp.Terminal):
                    if node.name.split('G')[0] == 'AR':
                        nodeAtt = int(float(node.name.split('G')[1]))
                        tree_args.add(nodeAtt)

            self.specifiedAttList = list(tree_args)
            tree_args = set()
            for idx, node in enumerate(cl.form):
                if isinstance(node, deap.gp.Terminal):
                    if node.name.split('G')[0] == 'AR':
                        nodeAtt = int(float(node.name.split('G')[1]))
                        tree_args.add(nodeAtt)

            cl.specifiedAttList = list(tree_args)
        # print "*********************************************************************"
        # print self.form
        # print "------------------------------------------------------"
        # print cl.form
        # print "------------------------------------------------------"
        # raise NameError("crossover with two trees")
        # self.form = new_self
        # cl.form = new_t

        else:  # SELF is a tree by the other entity is a Rule - Utilize special custom crossover proceedure.
            orig_form = copy.deepcopy(self.form)
            orig_rule = copy.deepcopy(cl)
            self.rule_crossover(cl, state)  # PERFORM THE RULE/TREE Crossover!!!!!!

            # Checking that a rule does not go beyond specLimit
            if len(cl.specifiedAttList) > cons.env.formatData.specLimit:
                cl.specLimitFix(cl)

            # fix other crossover mistakes in rules (like interval range swaps)
            v = 0
            while v < len(cl.specifiedAttList) - 1:
                attributeInfo = cons.env.formatData.attributeInfo[cl.specifiedAttList[v]]
                if attributeInfo[0]:  # If continuous feature
                    if cl.condition[v][0] > cl.condition[v][1]:
                        print('crossover error: cl')
                        print(cl.condition)
                        temp = cl.condition[v][0]
                        cl.condition[v][0] = cl.condition[v][1]
                        cl.condition[v][1] = temp
                        # cl.origin += '_fix' #temporary code

                    if state[cl.specifiedAttList[
                        v]] == cons.labelMissingData:  # If example is missing, don't attempt range, instead generalize attribute in rule.
                        # print 'removed '+str(cl.specifiedAttList[v])
                        cl.specifiedAttList.pop(v)
                        cl.condition.pop(v)  # buildMatch handles both discrete and continuous attributes
                        v -= 1
                    else:
                        if not cl.condition[v][0] < state[cl.specifiedAttList[v]] or not cl.condition[v][1] > state[
                            cl.specifiedAttList[v]]:
                            # print 'crossover range error'
                            attRange = attributeInfo[1][1] - attributeInfo[1][0]
                            rangeRadius = random.randint(25,
                                                         75) * 0.01 * attRange / 2.0  # Continuous initialization domain radius.
                            Low = state[cl.specifiedAttList[v]] - rangeRadius
                            High = state[cl.specifiedAttList[v]] + rangeRadius
                            condList = [Low,
                                        High]  # ALKR Representation, Initialization centered around training instance  with a range between 25 and 75% of the domain size.
                            cl.condition[v] = condList
                            # self.origin += '_RangeError' #temporary code
                v += 1

            if len(self.form) == 0:
                print("Original Form: " + str(orig_form))
                print("Original Rule Spec: " + str(orig_rule.specifiedAttList))
                print("Original Rule Cond: " + str(orig_rule.condition))
                raise NameError("Crossover Returning empty tree")

            try:
                self.test_arity()
            except:
                print("Original Form: " + str(orig_form))
                print("Original String: " + str(self.tree_string_form(orig_form)))
                print("Original Rule Spec: " + str(orig_rule.specifiedAttList))
                print("New Form: " + str(self.form))
                print("New String: " + self.tree_string())
                print("New Rule: " + str(cl.specifiedAttList))
                raise NameError("Crossover Returning invalid")

        if orig_form == self.form and (cl.isTree and other == cl.form):
            print('No change after crossover')
            return False
        else:
            return True

    def return_constants(self):
        terminal_idx = []
        index = 0
        for node in self.form:

            if isinstance(node, deap.gp.Terminal):
                if not node.name.split('G')[0] == 'AR':
                    terminal_idx.append(index)
            # if isinstance(node, deap.gp.rand101):
            #    terminal_idx.append(index)
            index += 1
        for idx in terminal_idx:
            print(self.form[idx].value)
            pass

    def replace(self, idx, argument):
        # hack that works for now. Ask randy about this
        terminal = self.terminals["ARG" + str(argument)]
        self.form[idx] = terminal

    # does not return clone
    def rule_crossover(self, rule, state):  # 'Mates' a rule with a tree.  #self is the tree, and rule is the rule.
        # (1)remove unique features from rule (2)

        # Handles the different scenarios of rule/tree crossover.  (1) Equal swap, more tree than rule
        tree_cross = []  # tree unique features information - index and attNumber
        rule_cross = []
        tree_args = set()  # unique set of attributes found in tree but not rule
        rule_orig = copy.deepcopy(rule.specifiedAttList)
        tree_orig = copy.deepcopy(self.form)
        # Identify all attributes specified in tree that are not in the rule, we do this instead of grab from self.specifiedAttList because we also store idx for each.
        for idx, node in enumerate(self.form):
            if isinstance(node, deap.gp.Terminal):
                if node.name.split('G')[0] == 'AR':
                    nodeAtt = int(float(node.name.split('G')[1]))

                    if nodeAtt not in rule.specifiedAttList:
                        tree_cross.append((idx, nodeAtt))
                    tree_args.add(nodeAtt)
        # print('tree/cross')
        # print(tree_cross)
        # Pick a random subset of unique tree attributes to exchange
        tree_cross[:] = [elt for elt in tree_cross if random.choice([0, 1]) == 0]

        # Select attributes to remove from rule and put in tree. They are removed from rule.
        for idx, val in enumerate(rule.specifiedAttList):
            if val not in tree_args:
                if random.choice([0, 1]) == 0:
                    rule_cross.append(rule.specifiedAttList.pop(
                        idx))  # what is idx here?  should be the position in the list (may be an error here)
                    rule.condition.pop(idx)  # also remove

        tree_cross_orig = copy.deepcopy(tree_cross)  # for debugging
        rule_cross_orig = copy.deepcopy(rule_cross)  # for debugging

        while len(tree_cross) != 0 or len(rule_cross) != 0:
            if len(tree_cross) == 0:  # Add an extra feature to tree
                try:
                    rule_arg = rule_cross.pop(0)
                    if random.choice([0, 1]) == 0:  # pick random subset of unique rule attributes to exchange
                        if not self.replace_constant(rule_arg):
                            self.add_terminal(rule_arg)
                    else:
                        if not self.add_terminal(rule_arg):
                            self.replace_constant(rule_arg)

                except Exception as exc:
                    print("Orig: " + str(tree_orig))
                    print("Form: " + str(self.form))
                    print("String: " + str(self.tree_string()))
                    print("Rule Cross: " + str(rule_cross_orig))
                    print(exc)
                    raise NameError("Problem with crossing")

            elif len(rule_cross) == 0:  # Add an extra feature to rule
                tree_arg = tree_cross.pop(0)
                # print "Removing: " + str(tree_arg[0]) + " Arg: " + str(tree_arg[1])
                # print "Tree cross" + str(tree_cross)
                # print self.form
                # print self.tree_string()
                removed = self.remove_from_tree(tree_arg[0], [])
                # print self.form
                # print self.tree_string()
                if tree_arg[1] not in rule.specifiedAttList:
                    rule.specifiedAttList.append(tree_arg[1])
                    rule.condition.append(self.buildMatch(tree_arg[1], state))

                # update indices of removed
                for index, arg in enumerate(tree_cross):
                    number = arg[0]
                    for idx in removed:
                        if idx < number:
                            number -= 1
                    new_arg = (number, arg[1])
                    tree_cross[index] = new_arg

            else:  # EVEN exhange between tree and rule!!!
                tree_arg = tree_cross.pop(0)
                rule_arg = rule_cross.pop(0)
                self.replace(tree_arg[0], rule_arg)
                if tree_arg[1] not in rule.specifiedAttList:
                    rule.specifiedAttList.append(tree_arg[1])
                    # have this append 1 for now, figure out how to do later
                    rule.condition.append(self.buildMatch(tree_arg[1], state))

        # temporary fix to get tree specAttlist correct
        tree_args = set()
        for idx, node in enumerate(self.form):
            if isinstance(node, deap.gp.Terminal):
                if node.name.split('G')[0] == 'AR':
                    nodeAtt = int(float(node.name.split('G')[1]))
                    tree_args.add(nodeAtt)

        self.specifiedAttList = list(tree_args)

        if len(rule.specifiedAttList) != len(set(rule.specifiedAttList)):
            print("Orig Tree: " + str(tree_orig))
            print("Orig Rule: " + str(rule_orig))
            print("Tree Args: " + str(tree_args))
            print("Tree Cross: " + str(tree_cross_orig))
            print("Rule Cross: " + str(rule_cross_orig))
            print("New Rule: " + str(rule.specifiedAttList))
            raise NameError("Duplicated in specAtt")

    """

    def remove_from_tree(self, idx, removed):

        if type(self.form[idx - 1]) == deap.gp.Primitive and self.form[idx-1].arity == 1:
            #print "recurse"
            del self.form[idx]
            removed.append(idx)
            return self.remove_from_tree(idx - 1, removed)

        else:
            count = 0
            curr = idx - 1
            while (curr + 1) != 0:
                #print "Curr: " + str(curr)
                count += 1
                #print "Count: " + str(count)
                if type(self.form[curr]) == deap.gp.Primitive:
                    count -= self.form[curr].arity
                    if count < 1:
                        del self.form[idx]
                        removed.append(idx)
                        del self.form[curr]
                        removed.append(curr)
                        break
                curr -= 1
            return removed

    """

    def remove_from_tree(self, idx, removed):
        removed.append(idx)
        count = 0
        curr = idx - 1
        while (curr + 1) != 0:
            # print "Curr: " + str(curr)
            count += 1
            # print "Count: " + str(count)
            if isinstance(self.form[curr], deap.gp.Primitive):
                count -= self.form[curr].arity
                if count < 1:
                    if self.form[curr].arity == 1:
                        removed.append(curr)
                    else:
                        removed.append(curr)
                        break
            curr -= 1
        if len(removed) < len(self.form):
            for item in removed:
                try:
                    del self.form[item]
                except:
                    print(item)
                    raise NameError("Problem with deletion")
        return removed

    def replace_constant(self, ruleAtt):

        terminal_idx = []
        for idx, node in enumerate(self.form):

            if isinstance(node, deap.gp.Terminal):
                if not node.name.split('G')[0] == 'AR':
                    terminal_idx.append(idx)

            # if isinstance(node, deap.gp.rand101):
            # terminal_idx.append(idx)
            if isinstance(node, deap.gp.Terminal):
                # print node.value
                # print node.name
                # rint getattr(node.ret)
                pass
        if len(terminal_idx) != 0:
            # select a rand101 terminal index at random
            idx = random.choice(terminal_idx)
            # make the value of the terminal at switch index the given rule
            self.replace(idx, ruleAtt)
            return True
        print("Problem in replace constant")  # is there a problem here?
        return False

    """

    def add_terminal(self, attr, max_depth):
        #get all terminals not at max depth
        #print "Problem in add terminal"

        #get terminal object of attr
        terminal = terminals["ARG" + str(attr)]
        #get arity 2 primitives
        prim_list = []
        for prim in primitives:
            if prim.arity == 2:
                prim_list.append(prim)
        #choose random 2-arity primitive
        ran = random.choice(range(len(prim_list)))
        primitive = prim_list[ran]

        branches = self.search_branch(max_depth)
        if len(branches) == 0:
            return False
        idx = random.choice(range(len(branches)))
        branch = branches[idx]

        #insert attributes to right place
        try:
            self.form.insert(branch, terminal)
            self.form.insert(branch, primitive)
        except:
            raise NameError("Problem in insertion")

        return True
    """

    def searchSubtree(form, begin):
        """Return a slice object that corresponds to the
        range of values that defines the subtree which has the
        element with index *begin* as its root.
        """
        end = begin + 1
        total = form[begin].arity
        while total > 0:
            total += form[end].arity - 1
            end += 1
        return slice(begin, end)

    def add_terminal(self, attr):
        """Inserts a new branch at a random position in *individual*. The subtree
        at the chosen position is used as child node of the created subtree, in
        that way, it is really an insertion rather than a replacement. Note that
        the original subtree will become one of the children of the new primitive
        inserted, but not perforce the first (its position is randomly selected if
        the new primitive has more than one child).
        :param individual: The normal or typed tree to be mutated.
        :returns: A tuple of one tree.
        """

        # get terminal object of attr
        terminal = self.terminals["ARG" + str(attr)]
        # get arity 2 primitives
        individual = self.form

        index = random.randrange(len(individual))
        node = individual[index]
        slice_ = individual.searchSubtree(index)
        choice = random.choice

        # As we want to keep the current node as children of the new one,
        # it must accept the return value of the current node
        primitives = [p for p in self.pset.primitives[node.ret] if node.ret in p.args and p.arity == 2]

        if len(primitives) == 0:
            return individual,

        new_node = choice(primitives)
        new_subtree = [None] * len(new_node.args)
        position = choice([i for i, a in enumerate(new_node.args) if a == node.ret])

        for i, arg_type in enumerate(new_node.args):
            if i != position:
                new_subtree[i] = terminal

        new_subtree[position:position + 1] = individual[slice_]
        new_subtree.insert(0, new_node)
        individual[slice_] = new_subtree

        self.form = individual

        return True

    # find all branches at less than max depth
    def search_branch(self, max_depth):
        tree = self.form
        stack = []
        idx = 0
        begin = tree[idx]
        ter_list = []
        depth = 1
        stack.insert(0, begin.arity)
        while len(stack) != 0:
            stack[0] -= 1
            idx += 1
            node = tree[idx]
            if isinstance(node, deap.gp.Primitive):
                # print "adding primitive"
                stack.insert(0, node.arity)
                depth += 1
            else:
                # print "adding terminal"
                if depth < max_depth:
                    ter_list.append(idx)
            # print stack
            while (len(stack) != 0 and stack[0] == 0):
                stack.pop(0)
                depth -= 1
            # print stack

        return ter_list
        # STOP GP INTEGRATION CODE*************************************************************************************************************************************

    def buildMatch(self, attRef, state):
        """ Builds a matching condition element given an attribute to be specified for the classifierCovering method. """
        attributeInfo = cons.env.formatData.attributeInfo[attRef]
        # -------------------------------------------------------
        # CONTINUOUS ATTRIBUTE
        # -------------------------------------------------------
        if attributeInfo[0]:
            attRange = attributeInfo[1][1] - attributeInfo[1][0]
            rangeRadius = random.randint(25, 75) * 0.01 * attRange / 2.0  # Continuous initialization domain radius.
            Low = state[attRef] - rangeRadius
            High = state[attRef] + rangeRadius
            condList = [Low,
                        High]  # ALKR Representation, Initialization centered around training instance  with a range between 25 and 75% of the domain size.
        # -------------------------------------------------------
        # DISCRETE ATTRIBUTE
        # -------------------------------------------------------
        else:
            condList = state[attRef]  # State already formatted like GABIL in DataManagement

        return condList

    # -------------------------------------------------
    # Mutation
    # -------------------------------------------------

    # START GP INTEGRATION CODE*************************************************************************************************************************************
    # get this to return boolean
    def Mutation(self, state, phenotype):
        origform = copy.deepcopy(self.form)
        # print "Original: " + str(self.form)
        self.form = self.toolbox.mutate(self.form)[0]
        self.form = self.toolbox.mutate(self.form)[0]
        self.form = self.toolbox.mutate(self.form)[0]
        # print "New: " + str(self.form)
        # print "Same: " + str(self.form == origform)
        # print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
        return self.form != origform

        # I guess for now just consider mutation failed?
        # STOP GP INTEGRATION CODE*************************************************************************************************************************************

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SUBSUMPTION METHODS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # figure out which booleans these should return

    def subsumes(self, cl):
        """ Returns if the classifier (self) subsumes cl """
        return False

    def isSubsumer(self):
        """ Returns if the classifier (self) is a possible subsumer. A classifier must have sufficient experience (one epoch) and it must also be as or more accurate than the classifier it is trying to subsume.  """
        return False

    def isMoreGeneral(self, cl):
        """ Returns if the classifier (self) is more general than cl. Check that all attributes specified in self are also specified in cl. """
        return False

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # DELETION METHOD
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def getDelProp(self, meanFitness):
        """  Returns the vote for deletion of the classifier. """
        if self.fitness / self.numerosity >= cons.delta * meanFitness or self.matchCount < cons.theta_del:
            self.deletionVote = self.aveMatchSetSize * self.numerosity

        elif self.fitness == 0.0 or self.fitness / self.numerosity == 0.0:
            self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (cons.init_fit / self.numerosity)
        else:
            # print self.fitness
            self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (
                        self.fitness / self.numerosity)  # note, numerosity seems redundant (look into theory of deletion in LCS.
        return self.deletionVote

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER METHODS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def equals(self, cl):
        if not cl.isTree:
            return False
        else:
            return self.form == cl.form

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PARAMETER UPDATES
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def updateEpochStatus(self, exploreIter):
        """ Determines when a learning epoch has completed (one cycle through training data). """
        # if not self.epochComplete and (exploreIter - self.initTimeStamp-1) >= cons.env.formatData.numTrainInstances and cons.offlineData:
        if not self.epochComplete and (
                exploreIter - self.initTimeStamp) >= cons.env.formatData.numTrainInstances and cons.offlineData:
            self.epochComplete = True
            cons.firstEpochComplete = True
            # randomProbClass = cons.env.formatData.classProportions[self.phenotype] #Put this in as a fix - for rules that become epoch complete after having been extrapolated on a previous run.
            # self.usefulDiff = self.correctCover - randomProbClass*self.matchCover
            self.usefulDiff = (self.correctCover - self.phenotype_RP * self.matchCover)  # *len(self.specifiedAttList)
            # Pareto Fitness - Epoch Complete Front Construction
            if self.accuracyComponent > 0 and self.usefulDiff > 0:
                objectivePair = [self.accuracyComponent, self.usefulDiff]
                changedme = cons.env.formatData.ecFront.updateFront(objectivePair)

            #                 if changedme:
            #                     print 'new'
            #                     print self.accuracyComponent
            #                     print self.usefulDiff
            #                     print self.initTimeStamp
            #             if self.accuracyComponent == 0.692582281546 and self.usefulDiff == 204.723:
            #                 print'I was born and now am complete'
            #                 print self.initTimeStamp

            return True
        return False

    def updateExperience(self):
        """ Increases the experience of the classifier by one. Once an epoch has completed, rule accuracy can't change."""
        self.matchCount += 1

        if self.epochComplete:  # Once epoch Completed, number of matches for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.matchCover += 1

        ######MOVE TO NEW PLACE EVENTUALLY
        if not cons.env.formatData.discretePhenotype:
            self.phenotype_RP = 0.5
            # print "gets here"
            """
            self.phenotype_RP = float(self.phenSum) / float(self.phenCount)
            if self.phenotype_RP > 1:
                print("phenSum: " + str(self.phenSum))
                print("matchCover: " + str(self.matchCover))
                print("RP: " + str(self.phenotype_RP))
                print("Count: " + str(self.phenCount))
                raise NameError("Problem with phenotype_RP")
            """

    def updateCorrect(self):
        """ Increases the correct phenotype tracking by one. Once an epoch has completed, rule accuracy can't change."""
        self.correctCount += 1
        if self.epochComplete:  # Once epoch Completed, number of correct for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.correctCover += 1

    def updateError(self, trueEndpoint):
        if not self.epochComplete:
            # Error caclulations are limited to extremes of observed training data phenotypes in calculating the range centroid.
            if self.phenotype > cons.env.formatData.phenotypeList[1]:
                adjustedError = 1
            elif self.phenotype < cons.env.formatData.phenotypeList[0]:
                adjustedError = 1
            else:
                error = abs(self.phenotype - trueEndpoint)
                adjustedError = error / (cons.env.formatData.phenotypeList[1] - cons.env.formatData.phenotypeList[0])

            self.errorSum += adjustedError  # Error is fraction of total phenotype range (i.e. maximum possible error)
            # if adjustedError == 0:
            #    print("#########################################")

            self.errorCount += 1

    def updateIncorrectError(self):
        if not self.epochComplete:
            self.errorSum += 1.0
            self.errorCount += 1

    # NEW
    def updateAccuracy(self, exploreIter):
        """ Update the accuracy tracker """
        nonUsefulDiscount = 0.001
        coverOpportunity = 1000
        adjAccuracy = 0
        # -----------------------------------------------------------------------------------
        # CALCULATE ACCURACY
        # -----------------------------------------------------------------------------------
        if cons.env.formatData.discretePhenotype:
            self.accuracy = self.correctCover / float(self.matchCover)
        else:  # ContinuousCode #########################
            self.accuracy = 1 - (
                        self.errorSum / self.matchCover)  # 1- average error based on range centroid.  Should be natural pressure to achieve narrow endpoint range.

        # -----------------------------------------------------------------------------------
        # CALCULATE ADJUSTED ACCURACY
        # -----------------------------------------------------------------------------------
        if self.accuracy > self.phenotype_RP:
            adjAccuracy = self.accuracy - self.phenotype_RP
        elif self.matchCover == 2 and self.correctCover == 1 and not self.epochComplete and (
                exploreIter - self.timeStampGA) < coverOpportunity:
            adjAccuracy = self.phenotype_RP / 2.0
        else:
            adjAccuracy = self.accuracy * nonUsefulDiscount
        # -----------------------------------------------------------------------------------
        # CALCULATE ACCURACY COMPONENT
        # -----------------------------------------------------------------------------------
        maxAccuracy = 1 - self.phenotype_RP
        if maxAccuracy == 0:
            self.accuracyComponent = 0
        else:
            self.accuracyComponent = adjAccuracy / float(
                maxAccuracy)  # Accuracy contribution scaled between 0 and 1 allowing for different maximum accuracies
        self.accuracyComponent = 2 * ((1 / float(1 + math.exp(-5 * self.accuracyComponent))) - 0.5) / float(
            0.98661429815)
        self.accuracyComponent = math.pow(self.accuracyComponent, 1)

    def updateCorrectCoverage(self):
        """ """

        self.coverDiff = (self.correctCover - self.phenotype_RP * self.matchCover)

    #         print self.coverDiff
    #         expectedCoverage = cons.env.formatData.numTrainInstances*self.totalFreq
    #         self.coverDiff = self.correctCover - expectedCoverage

    # NEW
    def updateIndFitness(self, exploreIter):
        """ Calculates the fitness of an individual rule based on it's accuracy and correct coverage relative to the 'Pareto' front """
        coverOpportunity = 1000
        if self.coverDiff > 0:
            #             print 'quality'
            # -----------------------------------------------------------------------------------
            # CALCULATE CORRECT COVER DIFFERENCE COMPONENT
            # -----------------------------------------------------------------------------------
            # NOTES: Coverage is directly comparable when epoch complete, otherwise we want to estimate what coverage might be farther out.
            if self.epochComplete:
                # Get Pareto Metric
                self.indFitness = cons.env.formatData.ecFront.getParetoFitness([self.accuracyComponent, self.coverDiff])

            else:  # Rule Not epoch complete
                # EXTRAPOLATE coverDiff up to number of trainin instances (i.e. age at epoch complete)
                ruleAge = exploreIter - self.initTimeStamp + 1  # Correct, because we include the current instance we are on.
                self.coverDiff = self.coverDiff * cons.env.formatData.numTrainInstances / float(ruleAge)
                objectivePair = [self.accuracyComponent, self.coverDiff]
                # BEFORE PARETO FRONTS BEGIN TO BE UPDATED
                if len(
                        cons.env.formatData.necFront.paretoFrontAcc) == 0:  # Nothing stored yet on incomplete epoch front
                    # Temporarily use accuracy as fitness in this very early stage.
                    # print 'fit path 1'
                    self.indFitness = self.accuracyComponent
                    if ruleAge >= coverOpportunity:  # attempt to update front
                        cons.env.formatData.necFront.updateFront(objectivePair)

                # PARETO FRONTS ONLINE
                else:  # Some pareto front established.
                    if len(cons.env.formatData.ecFront.paretoFrontAcc) > 0:  # Leave epoch incomplete front behind.
                        self.indFitness = cons.env.formatData.ecFront.getParetoFitness(objectivePair)
                        # print 'fit path 2'
                    else:  # Keep updating and evaluating with epoch incomplete front.
                        if ruleAge < coverOpportunity:  # Very young rules can not affect bestCoverDiff
                            self.preFitness = cons.env.formatData.necFront.getParetoFitness(objectivePair)
                            # print 'fit path 3'
                        else:
                            cons.env.formatData.necFront.updateFront(objectivePair)
                            self.indFitness = cons.env.formatData.necFront.getParetoFitness(objectivePair)
                            self.matchedAndFrontEstablished = True
                            # print 'fit path 4'
        else:
            #             print 'poor'
            #             print self.accuracyComponent
            self.indFitness = self.accuracyComponent / float(1000)

        if self.indFitness < 0:
            print("negative fitness error")
        if round(self.indFitness,
                 5) > 1:  # rounding added to handle odd division error, where 1.0 was being treated as a very small decimal just above 1.0
            print("big fitness error")

        # self.indFitness = math.pow(self.indFitness,cons.nu)  #Removed 11/25/15 - seems redundant with accuracy version (use one or the other)
        if self.indFitness < 0:
            print("CoverDiff: " + str(self.coverDiff))
            print("Accuracy: " + str(self.accuracyComponent))
            print("Fitness: " + str(self.indFitness))
            raise NameError("Problem with fitness")

        self.lastIndFitness = copy.deepcopy(self.indFitness)

    # NEW
    def updateRelativeIndFitness(self, indFitSum, partOfCorrect, exploreIter):
        """  Updates the relative individual fitness calculation """
        self.sumIndFitness = indFitSum
        self.partOfCorrect = partOfCorrect

        # self.lastRelativeIndFitness = copy.deepcopy(self.relativeIndFitness)   #Is this needed????

        if partOfCorrect:
            self.relativeIndFitness = self.indFitness * self.numerosity / float(self.sumIndFitness)
            # self.relativeIndFitness = self.indFitness/float(self.sumIndFitness) #Treat epoch complete or incomplete equally here.  This will give young rules a boost (in this method, relative fitness can be larger than 1 for NEC rules.
            if self.relativeIndFitness > 1.0:
                self.relativeIndFitness = 1.0
        #             if self.epochComplete: #Fitness shared only with other EC rules.
        #                 self.relativeIndFitness = self.indFitness*self.numerosity / self.sumIndFitness
        #             else:
        #                 if indFitSum == 0:
        #                     self.relativeIndFitness = 0
        #                 else:
        #                     self.relativeIndFitness = self.indFitness*self.numerosity*(exploreIter-self.initTimeStamp+1) / self.sumIndFitness
        else:
            self.relativeIndFitness = 0

    # NEW
    def updateFitness(self, exploreIter):
        """ Update the fitness parameter. """
        if self.epochComplete:
            percRuleExp = 1.0
        else:
            percRuleExp = (exploreIter - self.initTimeStamp + 1) / float(cons.env.formatData.numTrainInstances)
        # Consider the useful accuracy cutoff -  -maybe use a less dramatic change or ...na...
        beta = 0.2
        if self.matchCount >= 1.0 / beta:
            # print 'fit A'
            #                 print self.fitness
            #                 print self.relativePreFitness
            self.fitness = self.fitness + beta * percRuleExp * (self.relativeIndFitness - self.fitness)
        elif self.matchCount == 1 or self.aveRelativeIndFitness == None:  # second condition handles special case after GA rule generated, but not has not gone through full matching yet
            # print 'fit B'
            self.fitness = self.relativeIndFitness
            self.aveRelativeIndFitness = self.relativeIndFitness
        #                 if self.initTimeStamp == 2:
        #                     print self.aveRelativePreFitness
        #                     print 5/0
        else:
            # print 'fit C'
            self.fitness = (self.aveRelativeIndFitness * (
                        self.matchCount - 1) + self.relativeIndFitness) / self.matchCount  # often, last releative prefitness is 0!!!!!!!!!!!!!!!!!!!!!!!
            self.aveRelativeIndFitness = (self.aveRelativeIndFitness * (
                        self.matchCount - 1) + self.relativeIndFitness) / self.matchCount

        self.lastMatchFitness = copy.deepcopy(self.fitness)

        self.fitness = self.indFitness

        if self.fitness < 0 or round(self.fitness, 4) > 1:
            print('Negative Fitness')
            print(self.fitness)
            raise NameError("problem with fitness")

        # RULE FITNESS TESTING CODE------------------------------------------------------------------------------------------------------
        #         ruleTimeID = 3279
        #         if self.initTimeStamp == ruleTimeID:
        #             print '-----------------'
        #             print str(self.condition) + str(self.specifiedAttList)
        #             print 'sumfitness '+str(self.sumIndFitness)
        #             print 'correct? '+str(self.partOfCorrect)
        #             print 'relIndFit ' +str(self.relativeIndFitness)
        #             print 'fitness ' +str(self.fitness)

        # self.isMatchSetFitness = True

        #         if self.initTimeStamp == 2:
        #             self.reportClassifier('matchupdate')
        #         if self.fitness < 0.0001:
        #             print 'fit= '+str(self.fitness)
        #             print 'uDiff= '+str(self.usefulDiff)
        """
        if self.initTimeStamp == 1171:
            print('Iteration: '+str(exploreIter))
            print(self.numerosity)
            print(self.aveMatchSetSize)
            print(self.indFitness)
            print(self.fitness)
            print(self.deletionVote)
        """

    # NEW
    def briefUpdateFitness(self, exploreIter):
        # print 'briefupdateFit'
        # Activated durring matchign for all epoch complete rules -
        # Recalculate ENC rule fitness based on progression of algorithm (i.e. recalculate Coverage extrapolation and recheck pareto front - don't need to check for nondominated points
        # this is because points can only have lower extrapolation inbetween being in a match set.  This will effect rules that deal with rare states the most.

        # ALSO we need to adapt exstracs to really large datasets, where epoch complete is rare. ( consider prediction (weight votes by experience),
        # also consider that INcomplete pareto front might be most important (maybe not shift entirely over to EC front

        # Also revisit compaction - for big data - don't want to automatically remove rules that are not EC - want to include a percent data cutoff - could be same as for iterations passing, so that for small datasets, we alway use epoch complete
        # but for big ones we use experienced ENC rules.
        # Also revisit adjustment of all fitness to be no higher than best area ratio when over the front in extrapolation.
        # self.reportClassifier('lastupdate')

        # Recalculate coverDiff (this should not have changed)  This could be stored, so as not to have to recalculate
        # coverDiff = self.correctCover - self.phenotype_RP*self.matchCover
        indFitness = None
        # Criteria for a fitness update:
        # if self.partOfCorrect and coverDiff > 0 and self.matchedAndFrontEstablished == True and (len(cons.env.formatData.necFront.paretoFrontAcc) > 0 or len(cons.env.formatData.ecFront.paretoFrontAcc) > 0): #fitness only changes if was last part of a correct set - cause otherwise releativePreFitness stays at 0
        # print(self.coverDiff)
        if self.coverDiff > 0 and self.matchedAndFrontEstablished == True and (
                len(cons.env.formatData.necFront.paretoFrontAcc) > 0 or len(
                cons.env.formatData.ecFront.paretoFrontAcc) > 0):  # fitness only changes if was last part of a correct set - cause otherwise releativePreFitness stays at 0
            #             print 'NEW NEW NEW NEW NEW NEW NEW NEW'
            #             print 'Before correction-----------------------'
            #             print 'fitTYPE= ' +str(self.isMatchSetFitness)
            #             print 'fit= '+str(self.fitness)
            #             print 'pFit= '+str(self.preFitness)
            #             print 'uDiff= '+str(self.coverDiff)
            # lastPreFitness = copy.deepcopy(self.indFitness)

            # EXTRAPOLATE coverDiff up to number of training instances (i.e. age at epoch complete) This changes because more iterations have gone by.*******
            ruleAge = exploreIter - self.initTimeStamp + 1  # Correct, because we include the current instance we are on.
            coverDiff = self.coverDiff * cons.env.formatData.numTrainInstances / float(ruleAge)
            #             if coverDiff > self.coverDiff and self.coverDiff != None:
            #                 print 'exploreIter= '+str(exploreIter)
            #                 print 'InitTimestamp+1= ' + str(self.initTimeStamp+1)
            #                 print 'ruleAge= '+ str(ruleAge)
            #                 x = 5/0

            # Get new pareto fitness
            objectivePair = [self.accuracyComponent, coverDiff]
            # BEFORE PARETO FRONTS BEGIN TO BE UPDATED
            if len(cons.env.formatData.ecFront.paretoFrontAcc) > 0:  # Leave epoch incomplete front behind.
                indFitness = cons.env.formatData.ecFront.getParetoFitness(objectivePair)
                # print 'EC'
            else:  # Keep updating and evaluating with epoch incomplete front.
                indFitness = cons.env.formatData.necFront.getParetoFitness(objectivePair)
                # print 'ENC'
            #             print 'pFit= '+str(self.indFitness
            indFitness = math.pow(indFitness, cons.nu)

            #             if self.lastPreFitness < self.indFitness or self.lastPreFitness < indFitness:
            #                 if self.initTimeStamp == 2:
            #                     print "SWITCHING OVER FROM ACCURCY TO PARETO FRONT"
            #                     self.reportClassifier('update')
            # x = 5/0

            # Calculate new adjusted fitness by using last matching update score and recalculating.  (preserve originals so that updates are always based on originals, not on updated estimate)
            tempSumIndFitness = copy.deepcopy(self.sumIndFitness)
            # tempSumIndFitness = tempSumIndFitness - self.indFitness*self.numerosity   #this is not the true sum, but the sum with out the current rule's fitness.
            tempSumIndFitness = tempSumIndFitness - self.indFitness
            if self.lastIndFitness != indFitness:  # ok because with many new rules they may still be maxed out at highest fitness possible.
                #             if self.epochComplete: #Fitness shared only with other EC rules.
                #                 self.relativeIndFitness = self.indFitness*self.numerosity / self.sumPreFitness
                #             else:
                #                 self.relativeIndFitness = self.indFitness*self.numerosity*(exploreIter-self.initTimeStamp+1) / self.sumPreFitness
                # NOTE - have to re-adjust sumprefitnessnec to account for change in indFitness

                # Readjust sumIndFitness with new indFitness information. (this is an estimate, because the other rules may have changed.
                # tempSumIndFitness = tempSumIndFitness + indFitness*self.numerosity
                tempSumIndFitness = tempSumIndFitness + indFitness
                # self.relativeIndFitness = indFitness*self.numerosity / tempSumIndFitness
                self.relativeIndFitness = indFitness / float(tempSumIndFitness)

                percRuleExp = (exploreIter - self.initTimeStamp + 1) / float(cons.env.formatData.numTrainInstances)
                # Consider the useful accuracy cutoff -  -maybe use a less dramatic change or ...na...
                beta = 0.2
                if self.matchCount >= 1.0 / beta:

                    self.fitness = self.lastMatchFitness + beta * percRuleExp * (
                                self.relativeIndFitness - self.lastMatchFitness)
                elif self.matchCount == 1 or self.aveRelativeIndFitness == None:  # second condition handles special case after GA rule generated, but not has not gone through full matching yet
                    # print 'fit B'
                    self.fitness = self.relativeIndFitness
                    # self.aveRelativeIndFitness = self.relativeIndFitness
                #                 if self.initTimeStamp == 2:
                #                     print self.aveRelativePreFitness
                #                     print 5/0
                else:
                    # print 'fit C'
                    self.fitness = (self.aveRelativeIndFitness * (
                                self.matchCount - 1) + self.relativeIndFitness) / self.matchCount  # often, last releative prefitness is 0!!!!!!!!!!!!!!!!!!!!!!!
                    # self.aveRelativeIndFitness = (self.aveRelativeIndFitness*(self.matchCount-1)+self.relativeIndFitness)/self.matchCount

            #                 if cons.env.formatData.discretePhenotype:
            #                     #Consider the useful accuracy cutoff -  -maybe use a less dramatic change or ...na...
            #                     beta = 0.2
            #                     if self.matchCount >= 1.0/beta:
            #                         self.fitness = self.lastMatchFitness + beta*(self.relativeIndFitness-self.lastMatchFitness)
            #                     elif self.matchCount == 1:
            #                         self.fitness = self.relativeIndFitness
            #                     else:
            #                         self.fitness = (self.aveRelativeIndFitness*(self.matchCount-1)+self.relativeIndFitness)/self.matchCount
            #
            #
            #                     #self.fitness = self.indFitness #TEMPORARY
            #                 else: #ContinuousCode #########################
            #                     #UPDATE NEEDED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #                     if (self.phenotype[1]-self.phenotype[0]) >= cons.env.formatData.phenotypeRange:
            #                         self.fitness = pow(0.0001, 5)
            #                     else:
            #                         if self.matchCover < 2 and self.epochComplete:
            #                             self.fitness = pow(0.0001, 5)
            #                         else:
            #                             self.fitness = pow(self.accuracy, cons.nu) #- (self.phenotype[1]-self.phenotype[0])/cons.env.formatData.phenotypeRange)

            # self.isMatchSetFitness = False
            #                 x= 5/0
            #                 if self.initTimeStamp == 2:
            #                     self.reportClassifier('update')
            else:  # No fitness extrapolation update required
                #                 if self.initTimeStamp == 2:
                #                     print 'no update required B'
                pass
        else:  # No fitness extrapolation update required
            #             if self.initTimeStamp == 2:
            #                 print 'no update required A'
            pass

        # self.reportClassifier('update')
        if round(self.fitness, 5) > 1:
            self.fitness = 1.0
            print('FITNESS ERROR - adjust - too high')
        if self.fitness < 0:
            self.fitness = 0.0
            print('FITNESS ERROR - adjust - too low')
            # print self.fitness

            # x = 5/0

        self.lastIndFitness = copy.deepcopy(
            indFitness)  # TARGET - won't this cause below to never run?  - also where is lastIndFitness first stored??don't see it above.
        self.fitness = self.indFitness

    def updateNumerosity(self, num):
        """ Alters the numberosity of the classifier.  Notice that num can be negative! """
        self.numerosity += num

    def updateMatchSetSize(self, matchSetSize):
        """  Updates the average match set size. """
        if self.matchCount == 1:
            self.aveMatchSetSize = matchSetSize
        elif self.matchCount < 1.0 / cons.beta:  # < 5
            self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount - 1) + matchSetSize) / float(
                self.matchCount)
            # If matchCount = 2 -- 1 *
        else:
            self.aveMatchSetSize = self.aveMatchSetSize + cons.beta * (matchSetSize - self.aveMatchSetSize)

    def updateTimeStamp(self, ts):
        """ Sets the time stamp of the classifier. """
        self.timeStampGA = ts

    def setAccuracy(self, acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc

    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit

    def rangeCheck(self):
        pass

    # no idea what this is but it seems like I don't need this
    def calcClassifierStateFreq(self):
        # Fitness Frequency Calculation ------------------------------------------------------------------
        # -----------------------------------------------------------------------------------
        # CALCULATE STATE FREQUENCY COMPONENT
        # -----------------------------------------------------------------------------------
        self.totalFreq = 1

        """
        for i in range(0, len(self.specifiedAttList)):
            #-------------------------------------------------------
            # DISCRETE ATTRIBUTE
            #-------------------------------------------------------
            if not cons.env.formatData.attributeInfo[self.specifiedAttList[i]][0]:
                self.totalFreq = self.totalFreq * cons.env.formatData.attributeInfo[self.specifiedAttList[i]][1][self.condition[i]] #get the frequency for this attributes's specified state

            #-------------------------------------------------------
            # CONTINUOUS ATTRIBUTE
            #-------------------------------------------------------
            #Consider changing this.
            else:
                #Prepare to calculate estimated frequency
                attMax = float(cons.env.formatData.attributeInfo[self.specifiedAttList[i]][1][1])
                attMin = float(cons.env.formatData.attributeInfo[self.specifiedAttList[i]][1][0])

                binInterval = (attMax - attMin) / float(cons.numBins)
                ruleAttMax = self.condition[i][1]
                ruleAttMin = self.condition[i][0]

                correction = 0
                if attMin < 0:
                    correction = math.fabs(attMin)
                if attMin > 0:
                    correction = -1*attMin

                lowBin = (float(ruleAttMin)+correction)/ float(binInterval)
                highBin = (float(ruleAttMax)+correction)/ float(binInterval)

                if lowBin < 0: #May occur if rule min is lower than min observed in training data.
                    lowBin = 0
                    lowBinPercent = 1
                else:
                    lowBinPercent = (binInterval*(int(lowBin)+1)-(float(ruleAttMin)+correction))/float(binInterval)

                if int(highBin) >= cons.numBins:
                    highBin = cons.numBins - 1
                    highBinPercent = 1
                else:
                    highBinPercent = ((float(ruleAttMax)+correction)-(binInterval*(int(highBin))))/float(binInterval)

                if int(lowBin) == int(highBin):
                    if int(lowBin) == 0 and float(ruleAttMin)+correction < 0: #special case (LOW BIN under low boundary)
                        binPercent = (float(ruleAttMax)+correction)/float(binInterval)
                        self.totalFreq = self.totalFreq * binPercent * cons.env.formatData.attributeInfo[self.specifiedAttList[i]][2][int(lowBin)]

                    elif int(lowBin) == (cons.numBins - 1) and float(ruleAttMax)+correction > attMax+correction: #special case (HIGH BIN over upper boundary)
                        binPercent = (binInterval*(int(highBin))-(float(ruleAttMin)+correction))/float(binInterval)
                        self.totalFreq = self.totalFreq * binPercent * cons.env.formatData.attributeInfo[self.specifiedAttList[i]][2][int(lowBin)]
                    else: #middle bin
                        binPercent = ((float(ruleAttMax)+correction)-(float(ruleAttMin)+correction))/float(binInterval)
                        self.totalFreq = self.totalFreq * binPercent * cons.env.formatData.attributeInfo[self.specifiedAttList[i]][2][int(lowBin)]
                else:
                    frequencySum = lowBinPercent*cons.env.formatData.attributeInfo[self.specifiedAttList[i]][2][int(lowBin)]+highBinPercent*cons.env.formatData.attributeInfo[self.specifiedAttList[i]][2][int(highBin)]
                    binCount = lowBinPercent+highBinPercent
                    if int(highBin) - int(lowBin) > 1: #include full bin frequencies in count.
                        for x in range(int(lowBin)+1,int(highBin)):
                            frequencySum = frequencySum + cons.env.formatData.attributeInfo[self.specifiedAttList[i]][2][x]
                            binCount += 1

                    self.totalFreq = self.totalFreq * (frequencySum)


                if self.totalFreq > cons.env.formatData.maxFreq[len(self.specifiedAttList)]:
                    cons.env.formatData.maxFreq[len(self.specifiedAttList)] = self.totalFreq
                if self.totalFreq < cons.env.formatData.minFreq[len(self.specifiedAttList)]:
                    cons.env.formatData.minFreq[len(self.specifiedAttList)] = self.totalFreq

        #Include class frequency
        self.totalFreq = self.totalFreq * cons.env.formatData.classProportions[self.phenotype]

        """
        # print self.totalFreq

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PRINT CLASSIFIER FOR POPULATION OUTPUT FILE
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def printClassifier(self):

        classifierString = ""
        classifierString += str(self.specifiedAttList) + "\t"
        classifierString += str(self.form) + "\t"
        # -------------------------------------------------------------------------------
        specificity = float(cons.env.formatData.numAttributes)
        epoch = 0
        if self.epochComplete:
            epoch = 1
        classifierString += "None" + "\t"
        #         if cons.env.formatData.discretePhenotype:
        #             classifierString += str(self.phenotype)+"\t"
        #         else: #ContinuousCode #########################
        #             classifierString += str(self.phenotype[0])+';'+str(self.phenotype[1])+"\t"
        self.globalFitness = self.fitness * self.indFitness
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        classifierString += str(self.fitness) + "\t" + str(self.accuracy) + "\t" + str(self.numerosity) + "\t" + str(
            self.aveMatchSetSize) + "\t" + str(self.timeStampGA) + "\t" + str(self.initTimeStamp) + "\t" + str(
            specificity) + "\t"
        classifierString += str(self.deletionVote) + "\t" + str(self.correctCount) + "\t" + str(
            self.matchCount) + "\t" + str(self.correctCover) + "\t" + str(self.matchCover) + "\t" + str(
            epoch) + "\t" + str(self.accuracyComponent) + "\t" + str(self.coverDiff) + "\t" + str(
            self.indFitness) + "\t" + str(self.fitness) + "\t" + str(self.globalFitness) + "\t" + str(
            self.totalFreq) + "\n"

        return classifierString

    def reportClassifier(self, task):
        print(task)
        # print 'exploreIter= '+str(exploreIter)
        # print 'fitTYPE= ' +str(self.isMatchSetFitness)
        print('fit= ' + str(self.fitness))
        print('relativeIndFit= ' + str(self.relativeIndFitness))
        print('indFitness= ' + str(self.indFitness))
        print('cDiff= ' + str(self.coverDiff))
        print('self.correctCover=' + str(self.correctCover))
        print('self.matchCover= ' + str(self.matchCover))
        # print 'specAttList=' +str(self.specifiedAttList)
        print('self.accuracyComponent= ' + str(self.accuracyComponent))
        # ruleAge = exploreIter - self.initTimeStamp+1
        print('InitTimestamp+1= ' + str(self.initTimeStamp + 1))
        # print 'ruleAge= '+ str(ruleAge)
        print('self.self.aveMatchSetSize= ' + str(self.aveMatchSetSize))
        print('epochComplete= ' + str(self.epochComplete))
        print('numerosity= ' + str(self.numerosity))

        print('-----------------------------------------------')

    # START GP INTEGRATION CODE*************************************************************************************************************************************
    # -------------------------------------------------------------------
    # Tree Test Methods
    # -------------------------------------------------------------------

    def print_tree(self):
        print((self.form))

    def set_tree(self, string):
        self.form = deap.gp.PrimitiveTree.from_string(string, pset)

    def tree_string_form(self, form):
        string_list = [node.name for node in form]
        return "-".join(string_list)

    def tree_string(self):
        string_list = [node.name for node in self.form]
        return "-".join(string_list)

    def test_arity(self):
        arity = 0
        for node in self.form:
            if isinstance(node, deap.gp.Primitive):
                arity += node.arity

        if arity + 1 != len(self.form):
            # print "Tree String: " + self.tree_string()
            # print "Form: " + str(self.form)
            raise NameError("Invalid Tree")
    # STOP GP INTEGRATION CODE*************************************************************************************************************************************

