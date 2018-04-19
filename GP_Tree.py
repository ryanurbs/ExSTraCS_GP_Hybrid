"""
Name:        GP_Tree.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     February 20, 2018 (Siddharth Verma - Netaji Subhas Institute of Technology, Delhi, India.)
Description: This module defines an individual GP Tree classifier within the rule population, along with all respective parameters.
             Also included are tree-level methods, including matching, subsumption, crossover, and mutation.
             Parameter update methods are also included.

---------------------------------------------------------------------------------------------------------------------------------------------------------
ExSTraCS V2.0: Extended Supervised Tracking and Classifying System - An advanced LCS designed specifically for complex, noisy classification/data mining tasks,
such as biomedical/bioinformatics/epidemiological problem domains.  This algorithm should be well suited to any supervised learning problem involving
classification, prediction, data mining, and knowledge discovery.  This algorithm would NOT be suited to function approximation, behavioral modeling,
or other multi-step problems.  This LCS algorithm is most closely based on the "UCS" algorithm, an LCS introduced by Ester Bernado-Mansilla and
Josep Garrell-Guiu (2003) which in turn is based heavily on "XCS", an LCS introduced by Stewart Wilson (1995).

Copyright (C) 2014 Ryan Urbanowicz
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABLILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation,
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""


import math
import numpy as np
import copy
import random
import string
import Functions
from collections import deque
from operator import attrgetter
from exstracs_constants import *
from exstracs_pareto import *


class GP_Tree:
    """Genetic-Programming Syntax Tree created for performing the common GP operations on trees. This tree is traversed
and represented according to Depth-First-search (DFS) traversal.

Parameters and functions used to describe the tree are described as follows:"""

    # Node class of the tree which contains the terminal or the function with its children.
    class _Node:

        def __init__(self, data):
            """
            :rtype: _Node object
            """
            # Data is the function or the terminal constant
            self.data = data

            # Parent of every node will also be provided (Not implemented yet).
            self.parent = None

            # Its size would be equal to the arity of the function. For the terminals, it would be none
            self.children = None

        # This is overriden to define the representation of every node. The function is called recursively
        # to build the whole representation of the tree.
        def __str__(self):

            # "retval" is the final string that would be returned. Here the content of the node is added.
            retval = str(self.data) + "=>"

            # Content of the children of the node is added here in retval.
            if self.children is not None:
                for child in self.children:
                    retval += str(child.data) + ", "

            retval += "END\n"  # After every node and its children, string is concatenated with "END"

            # Recursive calls to all the nodes of the tree.
            if self.children is not None:
                for child in self.children:
                    retval += child.__str__()

            return retval

    def __init__(self, function_set=("add", "mul", "sub", "div", "cos", "max", "sin", "neg", "lt", "gt"), \
                 num_features=None, min_depth=2, max_depth=3):

        """The constructor of GP_Tree accepts the function set, terminal set, number of features to keep in the tree
        (eg: if the value of num_features =1, features in tree would be X0, if num_features=3, features in tree would be
        X0, X1, X2), max and min depth values and the fitness metric to be used."""

        self.function_set = function_set
        # A list of functions to be used. Custom functions can be created.'

        self.terminal_set = [random.randint(-5, 5), random.randint(-5, 5)]
        # List of floating point or zero arity functions acting as the terminals
        # of the tree

        if num_features is not None:
            self.num_features = num_features
        else:
            self.num_features = int(cons.env.formatData.numAttributes)
        # Specifies the num of features in the input file

        self.min_depth = min_depth
        # Specifies the minimum depth of the tree.

        self.max_depth = max_depth
        # Specifies the maximum depth of the tree.

        ###################################################################
        #                       ExSTraCS parameters                       #
        ###################################################################
        self.timeStampGA = 0  # Time since rule last in a correct set.
        self.initTimeStamp = 0  # Iteration in which the rule first appeared.
        self.aveMatchSetSize = 1

        self.lastMatch = 0  # Experimental - for brief fitness update
        # Classifier Accuracy Tracking --------------------------------------
        self.matchCount = 0  # Known in many LCS implementations as experience i.e. the total number of times this
        # classifier was in a match set
        self.correctCount = 0  # The total number of times this classifier was in a correct set
        self.matchCover = 0  # The total number of times this classifier was in a match set within a single epoch.
        # (value fixed after epochComplete)
        self.correctCover = 0  # The total number of times this classifier was in a correct set within a single epoch.
        # (value fixed after epochComplete)

        # Covering sets initially overly optimistic prediction values - this takes care of error with prediction which
        # previously had only zero value fitness and indFitness scores for covered rules.
        self.indFitness = 1.0
        self.fitness = 1.0

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
        self.phenotype = None  # Class if the endpoint is discrete, and a continuous phenotype if the endpoint is
        # continuous
        self.phenotype_RP = None  # NEW - probability of this phenotype occurring by chance.

        # Fitness Metrics ---------------------------------------------------
        self.accuracy = 0.0  # Classifier accuracy - Accuracy calculated using only instances in the data set which this
        # rule matched.
        self.accuracyComponent = None  # Accuracy adjusted based on accuracy by random chance, and transformed to give
        # more weight to early changes in accuracy.
        self.coverDiff = 1  # Number of instance correctly covered by rule beyond what would be expected by chance.
        self.indFitness = 0.0  # Fitness from the perspective of an individual rule (strength/accuracy based)
        self.relativeIndFitness = None
        self.fitness = 1  # CHANGED: Original self.fitness = cons.init_fit
        # Classifier fitness - initialized to a constant initial fitness value

        self.numerosity = 1  # The number of rule copies stored in the population.
        # (Indirectly stored as incremented numerosity)
        self.aveMatchSetSize = None  # A parameter used in deletion which reflects the size of match sets within this
        # rule has been included.
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

        ###################################################################
        # Some required parameters for the tree:
        ###################################################################
        self.root = None
        # This is the root of the tree. It is from here, that the tree is traversed by every member function.

        self.number_of_terminals = 0
        self.number_of_functions = 0
        # These parameters are required for calculating the "terminal_ratio" in generation methods.

        self._add_features_in_terminal_set(prefix="X")
        # Features are added in the final terminal_set

        self.generate_half_and_half()
        # Generate the Ramped Half and Half structure of the tree.

        self.specifiedAttList = self.getSpecifiedAttList()
        # Get the Specified Attribute List of the tree containing all the attributes in the tree.

        ####################################################################

    # this returns the string representation of the root which builds the representation of the whole tree recursively.
    def __str__(self):
        return self.root.__str__()

    @property
    def terminal_ratio(self):
        # Returns the ratio of the number of terminals to the number of all the functions in the tree.
        return self.number_of_terminals / float(self.number_of_terminals + self.number_of_functions)

    # Adds the number of arguments as specified in num_features in the syntax tree. The arguments is prefixed
    # with "X" followed by the index number. Eg: X0, X1, X2 ....
    def _add_features_in_terminal_set(self, prefix):

        temp_list = []
        for i in range(self.num_features):
            feature_str = "{prefix}{index}".format(prefix=prefix, index=i)
            temp_list.append(feature_str)

        temp_list.extend(self.terminal_set)
        self.terminal_set = temp_list

    @staticmethod
    def create_node(data):
        return GP_Tree._Node(data)

    #####################################################################################
    #                            Tree Generation Methods                                #
    #####################################################################################

    # The main tree generation function. Recursive function that starts building the tree from the root and returns the
    # root of the constructed tree.
    def _generate(self, condition, depth, height):

        node = None  # The node that would be returned and get assigned to the root of the tree.
        # See functions: 'generate_full' and 'generate_grow' for assignment to the root of the tree.

        # Condition to check if currently function is to be added. If the condition is false, then the terminal
        # is not yet reached and a function should be inserted.
        if condition(depth, height) is False:
            node_data = random.choice(self.function_set)  # Randomly choosing a function from the function set
            node_arity = Functions.get_arity(
                node_data)  # Getting the arity of the function to determine the node's children

            node = GP_Tree._Node(node_data)  # Creating the node.
            self.number_of_functions += 1

            node.children = []  # Creating the empty children list
            for _ in range(node_arity):
                child = self._generate(condition, depth + 1, height)
                child.parent = node
                node.children.append(child)  # Children are added recursively.

        else:  # Now the terminal should be inserted
            node_data = random.choice(self.terminal_set)  # Choosing the terminal randomly.
            node = GP_Tree._Node(node_data)  # Creating the terminal node
            self.number_of_terminals += 1
            node.children = None  # Children is none as the arity of the terminals is 0.

        return node  # Return the node created.

    def generate_full(self):
        # The method constructs the full tree. Note that only the function 'condition' is different in the
        # 'generate_grow()' and 'generate_full()' methods.

        def condition(depth, height):
            return depth == height

        height = random.randint(self.min_depth, self.max_depth)
        self.root = self._generate(condition, 0, height)

    def generate_grow(self):
        # The method constructs a grown tree.

        def condition(depth, height):
            return depth == height or (depth >= self.min_depth and random.random() < self.terminal_ratio)

        height = random.randint(self.min_depth, self.max_depth)
        self.root = self._generate(condition, 0, height)

    def generate_half_and_half(self):
        # Half the time, the expression is generated with 'generate_full()', the other half,
        # the expression is generated with 'generate_grow()'.

        # Selecting grow or full method randomly.
        method = random.choice((self.generate_grow, self.generate_full))
        # Returns either a full or a grown tree
        method()

    #####################################################################################
    #    Tree Traversal Methods: Different ways of representing the tree expression     #
    #####################################################################################

    # Depth-First-Traversal of the tree. It first reads the node, then the left child and then the right child.
    def tree_expression_DFS(self):

        expression = []  # expression to be built and returned

        # Recursive function as an helper to this function.
        self._tree_expression_DFS_helper(self.root, expression)

        return expression

    # Helper recursive function needed by the function "tree_expression_DFS()".
    def _tree_expression_DFS_helper(self, node, expression):

        expression.append(node.data)  # Expression to be built.

        if node.children is not None:
            for child in node.children:  # Adding children to the expression recursively.
                self._tree_expression_DFS_helper(child, expression)

        return

        # Breadth-First-Traversal of the tree. It first reads the left child, then the node itself and then the
        # right child.

    def tree_expression_BFS(self):
        q = deque()  # BFS is implemented using a queue (FIFO)
        expression = []  # Expression to be built and returned.

        # Adding root to the queue
        node = self.root
        q.append(node)

        while q:
            popped_node = q.popleft()
            if popped_node.children is not None:
                for child in popped_node.children:
                    q.append(child)

            expression.append(popped_node.data)

        return expression

    #####################################################################################
    #                                  Matching                                         #
    #####################################################################################
    def match(self, state):
        return True

    #####################################################################################
    #                            Tree Evaluation Methods                                #
    #####################################################################################

    # Function that sets the phenotype of the tree classifier. As it is taken from the exstracs code.
    def setPhenotype(self,
                     args):  # Ensure that for every instance we recalculate the tree phenotype prediction.  It never stays fixed.
        args = [int(i) for i in args]

        # Tuple to numpy array conversion. This is done because the evaluate function accepts only numpy arrays.
        # Reshaping denotes that this is a single sample with number of features = the length of the input args.
        args = np.asarray(args).reshape(1, len(args))
        dataInfo = cons.env.formatData

        # -------------------------------------------------------
        # BINARY PHENOTYPE - ONLY
        # -------------------------------------------------------

        if dataInfo.discretePhenotype and len(dataInfo.phenotypeList) == 2:
            if self.evaluate(args) > 0:
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

            if self.evaluate(args) < 0:  # lowest class
                self.phenotype = dataInfo.phenotypeList[0]
            elif self.evaluate(args) >= len(dataInfo.phenotypeList) - 1:  # lowest class
                self.phenotype = dataInfo.phenotypeList[len(dataInfo.phenotypeList) - 1]
            else:  # one of the middle classes
                count = 1
                notfoundClass = True
                while notfoundClass:
                    if self.evaluate(args) < count and self.evaluate(args) >= count - 1:
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
            self.phenotype = self.evaluate(args)
            if not self.epochComplete:  # Not sure where Ben came up with this but it seems to makes reasonable sense.  May want to examie more carefully.
                self.phenotype_RP = 0.5


    # This function evaluates the syntax tree and returns the value evaluated by the tree.
    def evaluate(self, X_Data):

        # X_Data : shape = [n_samples, num_features]
        # Training vectors, where n_samples is the number of samples and num_features is the number of features.

        # Return: Y_Pred: shape = [n_samples]
        # Evaluated value of the n_samples.

        Y_Pred = []
        # print (X_Data)
        for features in X_Data:
            if features.size != self.num_features:
                print(self.num_features, features.size)
                raise ValueError("Number of input features in X_Data is not equal to the parameter: 'num_features'.")

            Y_Pred.append(self._evaluate_helper(self.root, features))
        return float(np.asarray(Y_Pred))

    # Helper function for the func: "evaluate()". This makes recursive calls for the evaluation.
    def _evaluate_helper(self, node, X_Data):

        # Terminal nodes
        if node.children is None:

            if isinstance(node.data, str):
                feature_name = node.data
                index = int(feature_name[1:])
                return X_Data[index]

            else:
                return node.data

        args = []  # Will contain the input arguments i.e the children of the function in the tree.

        for child in node.children:
            args.append(self._evaluate_helper(child, X_Data))  # Evaluation by the recursive calls.

        func = Functions.get_function(node.data)  # Get the function from the alias name
        return float(func(*args))  # Return the computed value

    #####################################################################################
    # Genetic operations: These functions are made just for the integration with exstracs#
    #####################################################################################
    def uniformCrossover(self, classifier, state, phenotype):
        """This function is made only for the sake of integration with exstracs. Real
        crossover takes place takes place in the func: "uniformCrossover" which is
        declared outside this class."""

        return uniformCrossover(self, classifier, state)

    def Mutation(self, state, phenotype):
        origform = copy.deepcopy(self)

        '''Need to confirm from Ryan, which method to call?
        # Option 1: Call this method
        mutation_NodeReplacement(self)

        # Option 2: Call this method
        randomTree=GP_Tree(min_depth=3, max_depth=4)
        randomTree.generate_half_and_half()
        mutation_Uniform(self, randomTree)
        '''

        return str(self) == str(origform)

    #####################################################################################
    #                               Deletion Method                                     #
    #####################################################################################

    # Function to calculate the deletion vote of the tree.
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

    #####################################################################################
    #                                 Subsumption Methods                               #
    #####################################################################################
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

    #####################################################################################
    #            Some Miscellaneous methods needed in exstracs framework               #
    #####################################################################################

    def setPhenProb(self):
        pass

    # Taken from the exstracs code
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

    def equals(self, cl):
        if not cl.isTree:
            return False
        else:
            return str(cl) == str(self)

    def getSpecifiedAttList(self):
        tree_args = set()
        self._getSpecifiedAttList_Helper(self.root, tree_args)
        return list(tree_args)

    def _getSpecifiedAttList_Helper(self, node, tree_args):

        if (node.children is None):
            if isinstance(node.data, str):
                nodeAtt = int(float(node.data.split('X')[1]))
                tree_args.add(nodeAtt)
            return

        for child in node.children:
            self._getSpecifiedAttList_Helper(child, tree_args)

    #####################################################################################
    #                                Parameters Update                                  #
    #####################################################################################

    def updateClonePhenotype(self, phenotype):
        if phenotype == self.phenotype:
            self.correctCount = 1
            self.correctCover = 1
        else:
            self.correctCount = 0
            self.correctCover = 0

    def updateEpochStatus(self, exploreIter):
        """ Determines when a learning epoch has completed (one cycle through training data). """
        # if not self.epochComplete and (exploreIter - self.initTimeStamp-1) >= cons.env.formatData.numTrainInstances and cons.offlineData:
        if not self.epochComplete and (
                exploreIter - self.initTimeStamp) >= cons.env.formatData.numTrainInstances and cons.offlineData:
            self.epochComplete = True
            cons.firstEpochComplete = True
            # randomProbClass = cons.env.formatData.classProportions[self.phenotype] #Put this in as a fix - for rules that become epoch complete after having been extrapolated on a previous run.
            # self.usefulDiff = self.correctCover - randomProbClass*self.matchCover
            self.usefulDiff = (
                    self.correctCover - self.phenotype_RP * self.matchCover)  # *len(self.specifiedAttList)
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
        self.coverDiff = (self.correctCover - self.phenotype_RP * self.matchCover)

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

    #####################################################################################
    #                    Print classifier for population output file                    #
    #####################################################################################

    def printClassifier(self):

        classifierString = ""
        classifierString += str(self.specifiedAttList) + "\t"
        classifierString += str(self) + "\t"
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


#####################################################################################################
''' The Definition of the functions that are related to the population of the syntax trees.'''


#####################################################################################################

#####################################################################################
#           Tree Cloning: Used for creating copies of the parents by GA         #
#####################################################################################

# This will be called in runGA to make a copy of the classifier.
def tree_Clone(clOld, exploreIter):
    """  Constructs an identical Classifier.  However, the experience of the copy is set to 0 and the numerosity
    is set to 1 since this is indeed a new individual in a population. Used by the genetic algorithm to generate
    offspring based on parent classifiers."""

    offspring = copy.deepcopy(clOld)
    offspring.phenotype = None
    offspring.phenotype_RP = None

    offspring.timeStampGA = exploreIter  # consider starting at 0 instead???
    offspring.initTimeStamp = exploreIter
    offspring.lastMatch = exploreIter
    offspring.aveMatchSetSize = copy.deepcopy(clOld.aveMatchSetSize)
    offspring.fitness = clOld.fitness  # Test removal
    offspring.accuracy = clOld.accuracy
    offspring.relativeIndFitness = clOld.relativeIndFitness
    offspring.indFitness = clOld.indFitness
    offspring.sumIndFitness = clOld.sumIndFitness
    offspring.accuracyComponent = clOld.accuracyComponent
    offspring.matchCount = 1  # Known in many LCS implementations as experience i.e. the total number of times this
    # classifier was in a match set
    offspring.correctCount = None  # The total number of times this classifier was in a correct set
    offspring.matchCover = 1  # The total number of times this classifier was in a match set within a single epoch.
    # (value fixed after epochComplete)
    offspring.correctCover = None  # The total number of times this classifier was in a correct set within a single
    # epoch. (value fixed after epochComplete)

    offspring.condition = None
    offspring.errorSum = 0
    offspring.errorCount = 0
    offspring.phenCount = 0  # number of times phenProb added to count for testing reasons
    offspring.phenSum = 0  # sum of phenotype probability calculation values for continuous variables
    offspring.totalFreq = 1
    offspring.id = ''.join(random.choice(string.ascii_lowercase) for i in range(7))
    offspring.one_count = 0
    offspring.zero_count = 0

    offspring.coverDiff = 1  # Number of instance correctly covered by rule beyond what would be expected by chance.
    offspring.numerosity = 1  # The number of rule copies stored in the population.  (Indirectly stored as incremented numerosity)
    offspring.deletionVote = None  # The current deletion weight for this classifier.

    # Experience Management ---------------------------------------------
    offspring.epochComplete = False  # Has this rule existed for a complete epoch (i.e. a cycle through training set).

    offspring.lastFitness = 0.0
    offspring.partOfCorrect = True
    offspring.lastMatchFitness = 1.0
    offspring.aveRelativeIndFitness = None
    offspring.matchedAndFrontEstablished = False
    offspring.totalFreq = 1

    return offspring


#####################################################################################
#                            Population Initialization                              #
#####################################################################################

# Not for exstracs. This was made for other uses of GP Trees.
def initialize_population(function_set, terminal_set, num_features, min_depth, max_depth, fitness_metric,
                          population_size):
    """This function initializes the population of the trees. This needs to be changed according to the covering mechanism.
    Currently, it takes a parameter 'population_size' and returns the population of randomly created trees."""

    population = []
    for _ in range(population_size):
        tree = GP_Tree(function_set, terminal_set, num_features, min_depth, max_depth, fitness_metric)
        tree.generate_half_and_half()
        population.append(tree)

    return population


#####################################################################################
#               Parent selection method for applying genetic operators              #
#####################################################################################

def tournament_selection(population, k, tournament_size):
    """This function selects the best individual (based on fitness values) among 'tournament_size' randomly chosen
    individual trees, 'k' times. The list returned contains references to the syntax tree objects.

    population: A list of syntax trees to select from.
    k: The number of individuals to select.
    tournament_size: The number of individual trees participating in each tournament.
    returns: A list of selected individual trees.
    """
    selected = []
    for _ in range(k):
        selected_aspirants = [random.choice(population) for i in range(tournament_size)]
        selected.append(max(selected_aspirants, key=attrgetter("fitness")))
    return selected


#####################################################################################
#                            GP Cross-over: One-point crossover                     #
#####################################################################################

def crossover_onepoint(first_parent, second_parent):
    """ This method performs the one point crossover in the tree. Randomly select in each individual tree and exchange
    each subtree with the point as root between each individual."""

    # These lists will contain all the nodes that are present in the trees.
    subtree1_nodes = all_nodes(first_parent)
    subtree2_nodes = all_nodes(second_parent)

    if len(subtree1_nodes) > 1 and len(subtree2_nodes) > 1:
        # Select the random node index to make a slice.
        slice_index1 = random.randint(1, len(subtree1_nodes) - 1)
        slice_index2 = random.randint(1, len(subtree2_nodes) - 1)

        # Select nodes using the slice_index.
        slice_node1 = subtree1_nodes[slice_index1]
        slice_node2 = subtree2_nodes[slice_index2]

        # "ancestor_node1" and "ancestor_node2" are parent nodes of the sliced node. These will be needed while
        # applying actual crossover.
        ancestor_node1 = slice_node1.parent
        ancestor_node2 = slice_node2.parent

        # Finding the selected node to slice in its parent node
        for i in range(len(ancestor_node1.children)):
            if ancestor_node1.children[i] == slice_node1:
                ancestor_node1.children[i] = slice_node2  # Putting the subtree selected from the other tree.
                slice_node2.parent = ancestor_node1  # Making the parent reference
                break  # Can come out of the loop from here.

        # Same thing happening for the other tree.
        for i in range(len(ancestor_node2.children)):
            if ancestor_node2.children[i] == slice_node2:
                ancestor_node2.children[i] = slice_node1
                slice_node1.parent = ancestor_node2
                break

    return first_parent, second_parent


def uniformCrossover(first_parent, second_parent, state):
    """Performs the crossover in LCS for both rule and tree based population. If both the parents passed are trees,
    then the standard one point crossover algorithm is used to cross the parents. If one parent is the tree and second
    parent is the rule, then the novel approach as explained in the paper: "Problem driven machine learning by
    co-evolving genetic programming trees and rules in a learning classifier system" is used for the crossover between
    the rule and the tree."""

    if first_parent.isTree and second_parent.isTree:  # BOTH TREES-use the above implemented one-point crossover method.

        origForm1 = copy.deepcopy(first_parent)
        origForm2 = copy.deepcopy(second_parent)

        try:
            crossover_onepoint(first_parent, second_parent)
        except:
            print('One-point crossover failure')
            print("First Parent: " + str(origForm1))
            print("Second Parent " + str(origForm2))
            print(5 / 0)

        # Setting up specifiedAttList for both the parents.
        first_parent.specifiedAttList = first_parent.getSpecifiedAttList()
        second_parent.specifiedAttList = second_parent.getSpecifiedAttList()

    else:  # One parent is the Tree and the other entity is a Rule - Utilize special custom crossover procedure.
        origForm1 = copy.deepcopy(first_parent)
        origForm2 = copy.deepcopy(second_parent)

        # Making the first_parent as the GP_Tree instance.
        if isinstance(first_parent, GP_Tree):
            pass  # Do Nothing
        elif isinstance(second_parent, GP_Tree):
            # Swap and make the first_parent as the GP_Tree instance.
            first_parent, second_parent = second_parent, first_parent
        else:
            raise NameError("At least one parent should be a GP_Tree instance.")

        # PERFORM THE RULE/TREE Crossover!!!!!!
        rule_crossover(first_parent, second_parent, state)

        # Checking that a rule does not go beyond specLimit
        if len(second_parent.specifiedAttList) > cons.env.formatData.specLimit:
            second_parent.specLimitFix(second_parent)

        # Fix other crossover mistakes in rules (like interval range swaps)
        v = 0
        while v < len(second_parent.specifiedAttList) - 1:
            attributeInfo = cons.env.formatData.attributeInfo[second_parent.specifiedAttList[v]]
            if attributeInfo[0]:  # If continuous feature
                if second_parent.condition[v][0] > second_parent.condition[v][1]:
                    print('crossover error: cl')
                    print(second_parent.condition)
                    temp = second_parent.condition[v][0]
                    second_parent.condition[v][0] = second_parent.condition[v][1]
                    second_parent.condition[v][1] = temp

                if state[second_parent.specifiedAttList[v]] == cons.labelMissingData:
                    # If example is missing, don't attempt range, instead generalize attribute in rule.
                    second_parent.specifiedAttList.pop(v)
                    second_parent.condition.pop(v)  # buildMatch handles both discrete and continuous attributes
                    v -= 1
                else:
                    if not second_parent.condition[v][0] < state[second_parent.specifiedAttList[v]] or not \
                            second_parent.condition[v][1] > state[
                                second_parent.specifiedAttList[v]]:
                        # print 'crossover range error'
                        attRange = attributeInfo[1][1] - attributeInfo[1][0]
                        # Continuous initialization domain radius.
                        rangeRadius = random.randint(25, 75) * 0.01 * attRange / 2.0
                        Low = state[second_parent.specifiedAttList[v]] - rangeRadius
                        High = state[second_parent.specifiedAttList[v]] + rangeRadius
                        # ALKR Representation, Initialization centered around training instance  with a range between
                        # 25 and 75% of the domain size.
                        condList = [Low, High]
                        second_parent.condition[v] = condList
            v += 1

        if first_parent.root is None:
            print("Original Form: " + str(origForm1))
            print("Original Rule Spec: " + str(origForm2.specifiedAttList))
            print("Original Rule Cond: " + str(origForm2.condition))
            raise NameError("Crossover Returning empty tree.")

    if str(origForm1) == str(first_parent) and (
            isinstance(second_parent, GP_Tree) and str(origForm2) == str(second_parent)):
        print('No change after crossover')
        return False
    else:
        return True


# Performs the novel method of rule-tree crossover
def rule_crossover(tree, rule, currentState):
    """Mates a rule with a tree. "tree" and "rule" attributes are the parent classifiers and currentState is the
    current input state we have."""

    # Handles the different scenarios of rule/tree crossover.

    tree_cross = []  # tree unique features information - index and attNumber
    rule_cross = []
    tree_args = set()  # Unique set of attributes found in tree. Used for making the list "tree_cross".

    rule_orig = copy.deepcopy(rule.specifiedAttList)
    tree_orig = copy.deepcopy(tree)

    # Identify all attributes specified in tree that are not in the rule, we do this instead of grab from
    # self.specifiedAttList because we also store idx for each.

    terminals = terminal_nodes(tree)  # Contains all the terminal nodes of the tree.

    for idx, node in enumerate(terminals):
        if isinstance(node.data, str):  # Of the form "X0", "X1".... We want to obtain the features from terminal list
            # which also contains constants.
            nodeAtt = int(node.data.split('X')[1])

            if nodeAtt not in rule.specifiedAttList:
                tree_cross.append((idx, nodeAtt))
            tree_args.add(nodeAtt)

    # Pick a random subset of unique tree attributes to exchange
    tree_cross[:] = [elt for elt in tree_cross if random.choice([0, 1]) == 0]

    # Select attributes to remove from rule and put in tree. They are removed from rule.
    for idx, val in enumerate(rule.specifiedAttList):
        if val not in tree_args:
            if random.choice([0, 1]) == 0:
                rule_cross.append(rule.specifiedAttList.pop(idx))  # What is idx here? Should be the position in the
                # list (may be an error here).
                rule.condition.pop(idx)  # also remove

    tree_cross_orig = copy.deepcopy(tree_cross)  # for debugging
    rule_cross_orig = copy.deepcopy(rule_cross)  # for debugging

    # Handling all the three scenarios.
    while len(tree_cross) != 0 or len(rule_cross) != 0:

        # Scenario 3: Extra unique feature in Rule.
        if len(tree_cross) == 0:  # Add an extra feature to tree
            try:
                rule_arg = rule_cross.pop(0)
                # print ("Adding: " + str(rule_arg))
                # print ("Rule Cross" + str(rule_cross))
                # print("Tree cross" + str(tree_cross))
                # print (str(tree))

                if random.choice([0, 1]) == 0:
                    if not replace_constant(terminals, rule_arg):
                        add_terminal(tree, rule_arg)  # Case 1: Add the terminals in the tree.
                else:
                    if not add_terminal(tree, rule_arg):
                        replace_constant(terminals,
                                         rule_arg)  # Case 2: Replace the constants in the tree with the feature.
                # print(str(tree))

            except Exception:
                print("Original tree: " + str(tree_orig))
                print("Crossed tree: " + str(tree))
                print("Rule Cross: " + str(rule_cross_orig))
                raise NameError("Problem with crossing")

        # Scenario 2: Extra unique feature in Tree.
        elif len(rule_cross) == 0:  # Add an extra feature to rule

            tree_arg = tree_cross.pop(0)
            # print ("Removing: " + str(tree_arg[0]) + " Arg: " + str(tree_arg[1]))
            # print ("Tree cross" + str(tree_cross))
            # print ("Tree1: " + str(tree))
            # print("Terminals1: ", terminals)
            remove_from_tree(tree, terminals, tree_arg[0])  # Removing from tree
            # print("Tree2: " + str(tree))
            # print("Terminals2: ", terminals)

            if tree_arg[1] not in rule.specifiedAttList:
                rule.specifiedAttList.append(tree_arg[1])
                rule.condition.append(buildMatch(tree_arg[1], currentState))

        # Scenario 1: Equal feature swap :-
        else:  # EVEN exchange between tree and rule!!!
            tree_arg = tree_cross.pop(0)
            rule_arg = rule_cross.pop(0)
            # print ("Replacing: " + str(tree_arg[0]) + " Arg: " + str(tree_arg[1]))
            # print ("With: " + str(rule_arg))
            # print ("Tree cross" + str(tree_cross))
            # print("Rule cross" + str(rule_cross))
            # print (str(tree))
            replace(terminals, tree_arg[0], rule_arg)

            # print(str(tree))

            if tree_arg[1] not in rule.specifiedAttList:
                rule.specifiedAttList.append(tree_arg[1])
                # have this append 1 for now, figure out how to do later
                rule.condition.append(buildMatch(tree_arg[1], currentState))

    if tree.root is not None:
        tree.specifiedAttList = tree.getSpecifiedAttList()
    else:
        tree.SpecifiedAttList = []

    if len(rule.specifiedAttList) != len(set(rule.specifiedAttList)):
        print("Orig Tree: " + str(tree_orig))
        print("Orig Rule: " + str(rule_orig))
        print("Tree Args: " + str(tree_args))
        print("Tree Cross: " + str(tree_cross_orig))
        print("Rule Cross: " + str(rule_cross_orig))
        print("New Rule: " + str(rule.specifiedAttList))
        raise NameError("Duplicated in specAtt")


# ---------------------------------------------------
#  Miscellaneous methods for Rule-Tree Cross-over   #
# ---------------------------------------------------
def replace(terminals, index, argument):
    node = terminals[index]
    node.data = "X" + str(argument)


def buildMatch(attRef, state):
    """ Builds a matching condition element given an attribute to be specified for the classifierCovering method. """
    attributeInfo = cons.env.formatData.attributeInfo[attRef]

    # -----------------------
    # CONTINUOUS ATTRIBUTE
    # -----------------------
    if attributeInfo[0]:
        attRange = attributeInfo[1][1] - attributeInfo[1][0]
        rangeRadius = random.randint(25, 75) * 0.01 * attRange / 2.0  # Continuous initialization domain radius.
        Low = state[attRef] - rangeRadius
        High = state[attRef] + rangeRadius
        condList = [Low, High]  # ALKR Representation, Initialization centered around training instance  with a range
        # between 25 and 75% of the domain size.
        return condList

    # -------------------
    # DISCRETE ATTRIBUTE
    # -------------------
    else:
        condList = state[attRef]  # State already formatted like GABIL in DataManagement
        return condList


def remove_from_tree(tree, terminals, index):
    node = terminals[index]

    # Check whether or not sibling is there with the node to be deleted.
    isSiblingPresent = False

    if node == tree.root:  # Of the form "X0 => END"
        isSiblingPresent = False

    elif Functions.get_arity(node.parent.data) > 1:
        isSiblingPresent = True

    if isSiblingPresent:
        parentNode = node.parent
        grandParentNode = parentNode.parent

        # Finding the sibling of the node to be deleted. This sibling would become the child of the grand-parent.
        sibling = None
        for child in parentNode.children:
            if child != node:
                sibling = child
                break

        # Parent node is the root node. In this case, make the sibling node as the root of the tree.
        if grandParentNode is None:
            tree.root = sibling
            sibling.parent = None
            del parentNode

        # Make the sibling as the child of the grand parent node and delete the parent node.
        else:
            for i in range(len(grandParentNode.children)):
                if grandParentNode.children[i] == parentNode:
                    grandParentNode.children[i] = sibling
                    sibling.parent = grandParentNode
                    del parentNode
                    break

    else:
        while node.parent is not None:
            if (Functions.get_arity(node.parent.data) > 1):
                isSiblingPresent = True
                break
            node = node.parent

        if isSiblingPresent:
            parentNode = node.parent
            grandParentNode = parentNode.parent

            # Finding the sibling of the node to be deleted. This sibling would become the child of the grand-parent.
            sibling = None
            for child in parentNode.children:
                if child != node:
                    sibling = child
                    break

            # Parent node is the root node. In this case, make the sibling node as the root of the tree.
            if grandParentNode is None:
                tree.root = sibling
                sibling.parent = None
                del parentNode

            # Make the sibling as the child of the grand parent node and delete the parent node.
            else:
                for i in range(len(grandParentNode.children)):
                    if grandParentNode.children[i] == parentNode:
                        grandParentNode.children[i] = sibling
                        sibling.parent = grandParentNode
                        del parentNode
                        break


def replace_constant(terminals, ruleAtt):
    """This function replaces the constants in the tree with the specifier rule attributes. This is done according
    to the Scenario-3 (1st case) from the proposed strategy of rule-tree cross-over."""

    terminal_idx = []
    for idx, node in enumerate(terminals):

        if not isinstance(node.data, str):  # Of the form of constants
            terminal_idx.append(idx)

    if len(terminal_idx) != 0:
        # Select a random index from terminals containing all the constants in the tree.
        idx = random.choice(terminal_idx)

        # Make the value of the terminal at switch index the given rule
        replace(terminals, idx, ruleAtt)
        return True

    # print("No constant present to be replaced in Scenario 3 of rule-tree crossover.")
    print("Problem in replace constant")
    return False


def add_terminal(tree, ruleAtt):
    """Inserts a new branch at a random position in *individual*. The subtree
    at the chosen position is used as child node of the created subtree, in
    that way, it is really an insertion rather than a replacement. Note that
    the original subtree will become one of the children of the new primitive
    inserted, but not perforce the first (its position is randomly selected if
    the new primitive has more than one child).
    """

    terminal_data = "X" + str(ruleAtt)
    terminal_node = GP_Tree.create_node(terminal_data)

    functions = [func for func in tree.function_set if Functions.get_arity(func) == 2]
    function_data = random.choice(functions)
    function_node = GP_Tree.create_node(function_data)

    nodes = all_nodes(tree)
    slice_node = random.choice(nodes)

    slice_node_parent = slice_node.parent

    if slice_node_parent is None:  # Case of the root node.
        if random.choice([0, 1]) == 0:
            function_node.children = [terminal_node, slice_node]
        else:
            function_node.children = [slice_node, terminal_node]

        slice_node.parent = function_node
        terminal_node.parent = function_node

        tree.root = function_node

    else:
        for i in range(len(slice_node_parent.children)):
            if slice_node_parent.children[i] == slice_node:

                # Break the parent and slice node relation. Make the randomly selected function node as the child of the
                # parent.
                slice_node_parent.children[i] = function_node
                function_node.parent = slice_node_parent  # function_nodes parent is now the slice_node_parent

                # Assigning the children of the newly inserted function_node. One child would be the newly created terminal
                # node made from the rule attribute and the second child would be the sliced subtree.
                if random.choice([0, 1]) == 0:
                    function_node.children = [terminal_node, slice_node]
                else:
                    function_node.children = [slice_node, terminal_node]

                # Assigning the parents to the newly created children.
                slice_node.parent = function_node
                terminal_node.parent = function_node
                break
    return True


#####################################################################################
#                                 GP Mutation                                       #
#####################################################################################

def mutation_NodeReplacement(parent):
    """Replaces a randomly chosen node from the individual tree by a randomly chosen node with the same number
    of arguments from the attribute: "arity" of the individual node. It takes the input "parent" and performs the
    mutation operation on it."""

    par = parent

    # Root of the parent.
    root_parent = par.root

    # List to store all the nodes in the parent chosen. This list is populated using the function "mutation_helper".
    all_nodes = []

    # Populating the list "all_nodes".
    mutation_helper(root_parent, all_nodes)

    mutation_point = random.choice(all_nodes)  # Choosing the mutation point

    if mutation_point.children is None:  # Case 1: Mutation point is a terminal.
        new_terminal_data = random.choice(par.terminal_set)
        mutation_point.data = new_terminal_data

    else:  # Case 2: Mutation point is a function
        mutation_point_arity = Functions.get_arity(mutation_point.data)

        while True:  # Finding the same arity function.
            new_function_data = random.choice(par.function_set)
            if Functions.get_arity(new_function_data) == mutation_point_arity:
                mutation_point.data = new_function_data
                break

    offspring = par
    offspring.specifiedAttList = offspring.getSpecifiedAttList()
    return offspring


# This function is the wrapper function that will be used by the user. Actually, the function, "mutation_Uniform_Helper"
# performs the mutation.
def mutation_Uniform(parent, random_subtree):
    return mutation_Uniform_Helper(parent, random_subtree.root)


# Actually performing Uniform mutation.
def mutation_Uniform_Helper(parent_to_mutate, random_subtree_root):
    """Randomly select a mutation point in the individual tree, then replace the subtree at that point
    as a root by the "random_subtree_root" that was generated using one of the initialization methods."""

    parent = parent_to_mutate

    # Root of the parent.
    root_parent = parent.root

    # List to store all the nodes in the parent chosen. This list is populated using the function "mutation_helper".
    all_nodes = []

    # Populating the list "all_nodes".
    mutation_helper(root_parent, all_nodes)

    mutation_point = random.choice(all_nodes)  # Choosing the mutation point randomly.
    ancestor_mutation_point = mutation_point.parent  # Saving the parent node of the mutation point

    if ancestor_mutation_point is None:  # If root is chosen as the Mutation-point, perform mutation through
        # Node-Replacement method. This has been suggested by Ryan.

        print("Root was selected as the mutation point. Therefore, Uniform Mutation can't occur. Performing Mutation \
        through Node-Replacement Method")
        offspring = mutation_NodeReplacement(parent_to_mutate)
        return offspring

    # Performing the mutation.
    for i in range(len(ancestor_mutation_point.children)):
        if ancestor_mutation_point.children[i] == mutation_point:
            ancestor_mutation_point.children[i] = random_subtree_root
            del mutation_point
            random_subtree_root.parent = ancestor_mutation_point
            break

    offspring = parent
    offspring.specifiedAttList = offspring.getSpecifiedAttList()

    return offspring


def mutation_helper(node, all_nodes):
    """ Helper function to store all the nodes in a tree and return the list of the stored nodes. Used by mutation
    methods."""

    if node.children is None:
        all_nodes.append(node)
        return

    all_nodes.append(node)

    for child in node.children:
        mutation_helper(child, all_nodes)


#####################################################################################
# Methods to return the list containing the terminal nodes and function nodes in    #
#                                   the tree.                                       #
#####################################################################################

def terminal_nodes(tree):
    """This function returns a list containing all the terminal nodes present in the tree. This uses the func:
    "terminal_nodes_helper" for making the recursive calls."""
    terminals = []
    terminal_nodes_helper(tree.root, terminals)
    return terminals


def terminal_nodes_helper(node, terminals):
    """Helper function for creating the terminal list. Used by the func: "terminal_nodes" for making the recursive
    calls"""

    if node.children is None:
        terminals.append(node)  # Only append the node if it is terminal, i.e. it has no children.
        return

    for child in node.children:
        terminal_nodes_helper(child, terminals)  # Making the recursive calls for the other nodes in the tree.


def function_nodes(tree):
    """This function returns a list containing all the function nodes present in the tree. This uses the func:
    "function_nodes_helper" for making the recursive calls."""
    functions = []
    function_nodes_helper(tree.root, functions)
    return functions


def function_nodes_helper(node, functions):
    """Helper function for creating the function list. Used by the func: "function_nodes" for making the recursive
    calls"""
    if node.children is None:
        return  # Base-case.

    functions.append(node)  # Appending the node in the list.

    for child in node.children:
        function_nodes_helper(child, functions)  # Recursive calls for the other nodes.


def all_nodes(tree):
    """This function returns a list containing all the nodes present in the tree. This uses the func:
    "all_nodes_helper" for making the recursive calls."""
    all_nodes = []
    all_nodes_helper(tree.root, all_nodes)
    return all_nodes


def all_nodes_helper(node, all_nodes):
    """Helper function for creating the list containing all the nodes. Used by the func: "all_nodes" for making the recursive
    calls"""

    all_nodes.append(node)  # Appending the node in the list.

    if node.children is None:
        return  # Base-case.

    for child in node.children:
        all_nodes_helper(child, all_nodes)  # Recursive calls for the other nodes.


#####################################################################################################################
#####################################################################################################################
#                                                     TESTING                                                       #
#####################################################################################################################
#####################################################################################################################


if __name__ == "__main__":
    # Example to show the formation and representation of a single gp-tree.

    function_set = ("add", "mul", "sub", "div", "cos", "sqrt", "abs", "sin", "tan")
    tree = GP_Tree(function_set, num_features=4, min_depth=2, max_depth=4)

    print("Tree representation:\n" + str(
        tree))  # Printing the tree. The tree is printed in the form of the actual tree. It starts with the root,
    # followed by an arrow with its children and an "END". Then the subsequent lines show the children nodes traversed down
    # along the depth. Terminal nodes are just ended by an "END".

    # Evaluating the tree on the sample input (1,2,3,4).
    print("Evaluating the tree: %f" % tree.evaluate(np.asarray([[1, 2, 3, 4]])))

    print("Depth First Traversal: ", tree.tree_expression_DFS())  # Printing the tree in a Depth first order
    print("Breadth First Traversal: ", tree.tree_expression_BFS())  # Printing the tree in Breadth first order

    tree1 = GP_Tree(function_set, num_features=5, min_depth=2, max_depth=4)
    # Generating a random subtree for mutation.
    mut_subtree = GP_Tree(function_set, num_features=5, min_depth=2, max_depth=3)

    print("Tree1: " + str(tree1))
    print("Mutation Subtree: " + str(mut_subtree))
    # Passing the population and the random subtree to the mutation function
    offspring = mutation_Uniform(tree1, mut_subtree)
    print("Tree1: " + str(tree1))

    # Performing the mutation operation on the GP tree through Node Replacement method.
    offspring = mutation_NodeReplacement(tree1)
    print("Tree1: " + str(tree1))

    # Performing the crossover operation on the GP tree through one point crossover method.

    tree2 = GP_Tree(function_set, num_features=5, min_depth=2, max_depth=4)
    tree3 = GP_Tree(function_set, num_features=3, min_depth=2, max_depth=4)

    print("Tree2: " + str(tree2))
    print("Tree3: " + str(tree3))
    crossover_onepoint(tree2,
                       tree3)  # There can be a case of NO change in the tree occured, due to asymmetrical parents.
    print("Tree2: " + str(tree2))
    print("Tree3: " + str(tree3))