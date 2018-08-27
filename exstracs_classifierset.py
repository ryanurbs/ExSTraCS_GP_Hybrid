"""
Name:        ExSTraCS_ClassifierSet.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Description: This module handles all classifier sets (population, match set, correct set) along with mechanisms and heuristics that act on these sets.

---------------------------------------------------------------------------------------------------------------------------------------------------------
ExSTraCS V1.0: Extended Supervised Tracking and Classifying System - An advanced LCS designed specifically for complex, noisy classification/data mining tasks,
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
#Import Required Modules-------------------------------
from exstracs_constants import *
from exstracs_classifier import Classifier
# from exstracs_tree import *
from GP_Tree import *
print()
import random
import copy
import sys
#------------------------------------------------------

class ClassifierSet:
    def __init__(self, a=None):
        """ Overloaded initialization: Handles creation of a new population or a rebooted population (i.e. a previously saved population). """
        # Major Parameters-----------------------------------
        self.popSet = []        # List of classifiers/rules
        self.matchSet = []      # List of references to rules in population that match
        self.correctSet = []    # List of references to rules in population that both match and specify correct phenotype
        self.microPopSize = int(cons.popInitGP * cons.N)   # Tracks the current micro population size, i.e. the population size which takes rule numerosity into account.

#         #Epoch Pool Deletion---------------------------------
#         self.ECPopSize = 0      #Epoch Complete - Micro Pop Size
#         self.ENCPopSize = 0     #Epoch Not Complete - Micro Pop Size

        #Evaluation Parameters-------------------------------
        self.aveGenerality = 0.0
        self.expRules = 0.0
        self.attributeSpecList = []
        self.attributeAccList = []
        self.avePhenotypeRange = 0.0

        #Test parameters ------------------------------------
        self.tree_cross_count = 0
        self.rule_cross_count = 0
        self.both_cross_count = 0

        #Parameter for continuous trees
        self.tree_error = None #changing error threshold for trees to be considered in the correct set



        #Set Constructors-------------------------------------
        if a==None:
            self.makePop()  #Initialize a new population
        elif isinstance(a,str):
            self.rebootPop(a) #Initialize a population based on an existing saved rule population
        else:
            print("ClassifierSet: Error building population.")


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # POPULATION CONSTRUCTOR METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def makePop(self):
        """ Initializes the rule population """
        self.popSet = []
        #Initialize Population of Trees----------------------
        #hardcode num Trees for now

        if cons.useGP:
            #Initialize Population of Trees----------------------
            gpInit = cons.popInitGP * cons.N
            print("Initializing Tree population with "+str(int(gpInit))+" GP trees.")
            #initialize marked tree for testing
            for x in range(0, int(cons.popInitGP * cons.N)-1):
                # newTree = Tree() ## For older DEAP code.
                newTree = GP_Tree()
                self.popSet.append(newTree)
            print("Tree Initialization Complete")


    def rebootPop(self, remakeFile):
        """ Remakes a previously evolved population from a saved text file. """
        print("Rebooting the following population: " + str(remakeFile)+"_RulePop.txt")
        #*******************Initial file handling**********************************************************
        try:
            datasetList = []
            f = open(remakeFile+"_RulePop.txt", 'rU')
            self.headerList = f.readline().rstrip('\n').split('\t')   #strip off first row
            for line in f:
                lineList = line.strip('\n').split('\t')
                datasetList.append(lineList)
            f.close()

        except IOError as xxx_todo_changeme:
            (errno, strerror) = xxx_todo_changeme.args
            print ("Could not Read Remake File!")
            print(("I/O error(%s): %s" % (errno, strerror)))
            raise
        except ValueError:
            print ("Could not convert data to an integer.")
            raise
        except:
            print(("Unexpected error:", sys.exc_info()[0]))
            raise
        #**************************************************************************************************
        for each in datasetList:
            cl = Classifier(each)
            self.popSet.append(cl) #Add classifier to the population
            numerosityRef = 5  #location of numerosity variable in population file.
            self.microPopSize += int(each[numerosityRef])
#             if cl.epochComplete:
#                 self.ECPopSize += int(each[numerosityRef])
#             else:
#                 self.ENCPopSize += int(each[numerosityRef])


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER SET CONSTRUCTOR METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def makeMatchSet(self, state_phenotype, exploreIter):
        """ Constructs a match set from the population. Covering is initiated if the match set is empty or a rule with the current correct phenotype is absent. """
        #Initial values----------------------------------
        state = state_phenotype[0]
        phenotype = state_phenotype[1]

        #--------------------------------------------------------
        # Define phenotypes for trees
        #--------------------------------------------------------

        for i in range(len(self.popSet)):
            cl = self.popSet[i]
            if cl.isTree:
                cl.setPhenotype(state)

        #ContinuousCode #########################
        #Calculate error threshold

        if not cons.env.formatData.discretePhenotype:
            totalError = 0
            tree_count = 0
            for cl in self.popSet:
                if cl.isTree:
                    error = abs(float(phenotype) - float(cl.phenotype))
                    dataInfo = cons.env.formatData
                    if error > (dataInfo.phenotypeList[1] - dataInfo.phenotypeList[0]):
                        error = (dataInfo.phenotypeList[1] - dataInfo.phenotypeList[0])
                    totalError += error
                    #print abs(float(phenotype) - float(cl.phenotype))
                    tree_count += 1
            newError = totalError / tree_count
            if self.tree_error != None:
                #change learning rate to be in constants
                self.tree_error = self.tree_error + (0.2 * (newError - self.tree_error))
            else:
                self.tree_error = newError

            #print("New Error: " + str(newError) + " Tree Error: " + str(self.tree_error) + "Total Error: " + str(totalError))

            for cl in self.popSet:
                if cl.isTree:
                    cl.calcPhenProb(self.tree_error)


        doCovering = True # Covering check: Twofold (1)checks that a match is present, and (2) that at least one match dictates the correct phenotype.
        setNumerositySum = 0
        #-------------------------------------------------------
        # MATCHING
        #-------------------------------------------------------
        cons.timer.startTimeMatching()
        for i in range(len(self.popSet)):                       # Go through the population
            cl = self.popSet[i]                                 # One classifier at a time
            epochCompleted = False
            epochCompleted = cl.updateEpochStatus(exploreIter)                   # Note whether this classifier has seen all training data at this point.
#             if epochCompleted:
#                 self.ECPopSize += cl.numerosity       #Epoch Complete - Micro Pop Size
#                 self.ENCPopSize -= cl.numerosity      #Epoch Not Complete - Micro Pop Size
            #Fitness Update------------------------------
            if not cl.epochComplete and (exploreIter - cl.lastMatch) >= cons.noMatchUpdate:
                cl.briefUpdateFitness(exploreIter)

            if cl.match(state):                                 # Check for match
                cl.lastMatch = exploreIter                     # Experimental::::: for brief fitness update.
                self.matchSet.append(i)                         # If match - add classifier to match set
                setNumerositySum += cl.numerosity               # Increment the set numerosity sum
                #Covering Check--------------------------------------------------------
                if cons.env.formatData.discretePhenotype:       # Discrete phenotype
                    if cl.phenotype == phenotype and not cl.isTree:               # Check for phenotype coverage
                        doCovering = False
                else: #ContinuousCode #########################
                    if not cl.isTree and float(cl.phenotype[0]) <= float(phenotype) <= float(cl.phenotype[1]):        # Check for phenotype coverage
                        doCovering = False

        cons.timer.stopTimeMatching()
        #-------------------------------------------------------
        # COVERING
        #-------------------------------------------------------
        if doCovering:
            #print('Covered new rule')
            pass
        while doCovering:
            cons.timer.startTimeCovering()
            newCl = Classifier(setNumerositySum+1,exploreIter, state, phenotype)
            self.addCoveredClassifierToPopulation(newCl)
            self.matchSet.append(len(self.popSet)-1)  # Add covered classifier to matchset
            doCovering = False
            cons.timer.stopTimeCovering()
        """

        if exploreIter % 100 == 0 and exploreIter > 0:
            numRules = 0
            numTrees = 0

            for i in range(len(self.popSet)):
                cl = self.popSet[i]
                if cl.isTree:
                    numTrees += 1
                else:
                    numRules += 1

            #print params of iteration
            print "Iter: " + str(exploreIter) + " PopSize: " + str(len(self.popSet)) + " MatchSize: " + str(len(self.matchSet))
            print "MicroPopSize: " + str(self.microPopSize) + " NumTrees: " + str(numTrees) + " NumRules: " + str(numRules)
            print "Tree: " + str(self.tree_cross_count) + " Rule: " + str(self.rule_cross_count) + " Both: " + str(self.both_cross_count) + " Total: " + str(self.tree_cross_count + self.rule_cross_count + self.both_cross_count)


            best = 0
            best_tree = None
            for cl in self.popSet:
                if cl.isTree:
                    if cl.accuracy > best:
                        best = cl.accuracy
                        best_tree = cl
            print "Best tree accuracy: " + str(best)
            if best_tree:
                print "Best Tree: " + str(best_tree.form)
                #print "Fitness: " + str(best_tree.fitness)
                print "ID: " + str(best_tree.id)

        """
        """

        found = False
        for cl in self.popSet:
            if cl.marked:
                print "Accuracy: " + str(cl.accuracy) + " Fitness: " + str(cl.fitness) + " MatchCount: " + str(cl.matchCount) + " CorrectCount: " + str(cl.correctCount)

                #print cl.form
                found = True
                break
        if not found:
            print "Deleted"

        """

        #track young tree

        """

        for cl in self.popSet:
            if cl.isTree:
                if cl.initTimeStamp > 0:
                    print "Young Tree: " + str(cl.form)
                    print "Fitness: " + str(cl.fitness)
                    print "ID: " + str(cl.id)
        """

        #Last used reporting!!!!!!!!!!!!!!!!!!!!
        """
        numRules = 0
        numTrees = 0

        for i in range(len(self.popSet)):
            cl = self.popSet[i]
            if cl.isTree:
                numTrees += 1
            else:
                numRules += 1

        #print params of iteration
        print("Iter: " + str(exploreIter) + " PopSize: " + str(len(self.popSet)) + " MatchSize: " + str(len(self.matchSet)))
        print("MicroPopSize: " + str(self.microPopSize) + " NumTrees: " + str(numTrees) + " NumRules: " + str(numRules))
        print("Tree: " + str(self.tree_cross_count) + " Rule: " + str(self.rule_cross_count) + " Both: " + str(self.both_cross_count) + " Total: " + str(self.tree_cross_count + self.rule_cross_count + self.both_cross_count))


        best = 0
        best_tree = None
        for cl in self.popSet:
            if cl.isTree:
                if cl.fitness > best:
                    best = cl.fitness
                    best_tree = cl
        print("Best tree fitness: " + str(best))
        if best_tree:
            print("Best Tree: " + str(best_tree.form))
            print("AccuracyComp: " + str(best_tree.accuracyComponent))
            print("CoverDiff: " + str(best_tree.coverDiff))
            print("Ind Fitness: " + str(best_tree.indFitness))
            print("Epoch Complete: " + str(best_tree.epochComplete) + " MatchCount: " + str(best_tree.matchCount))
            #print "ID: " + str(best_tree.id)
        """






    def makeCorrectSet(self, phenotype):
        """ Constructs a correct set out of the given match set. """
        for i in range(len(self.matchSet)):
            ref = self.matchSet[i]
            #-------------------------------------------------------
            # DISCRETE PHENOTYPE
            #-------------------------------------------------------
            if cons.env.formatData.discretePhenotype:
                if self.popSet[ref].phenotype == phenotype:
                    self.correctSet.append(ref)
            #-------------------------------------------------------
            # CONTINUOUS PHENOTYPE
            #-------------------------------------------------------
            else: #ContinuousCode #########################
                if not self.popSet[ref].isTree: #RULES
                    if float(phenotype) <= float(self.popSet[ref].phenotype[1]) and float(phenotype) >= float(self.popSet[ref].phenotype[0]):
                        self.correctSet.append(ref)
                else: #TREES
                    #We won't use any notion of a correct set for GP trees.  USe error to determine correct set for att tracking downstream, but not used for accuracy update.
                    if abs(float(phenotype) - float(self.popSet[ref].phenotype)) <= self.tree_error:
                        self.correctSet.append(ref)


        """
        if exploreIter % 100 == 0 and exploreIter > 0:
            numRules = 0
            numTrees = 0

            for i in self.correctSet:
                cl = self.popSet[i]
                if cl.isTree:
                    numTrees += 1
                else:
                    numRules += 1

            print "CorrectSet - NumTrees: " + str(numTrees) + " NumRules: " + str(numRules)
        """


    def makeEvalMatchSet(self, state):
        """ Constructs a match set for evaluation purposes which does not activate either covering or deletion. """
        for i in range(len(self.popSet)):       # Go through the population
            cl = self.popSet[i]                 # A single classifier
            
            if cl.isTree: #In evaluation we still need to update the phenotype for each instance for trees
                cl.setPhenotype(state)
                
            if cl.match(state):                 # Check for match
                self.matchSet.append(i)         # Add classifier to match set


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER DELETION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def deletion(self, exploreIter):
        """ Returns the population size back to the maximum set by the user by deleting rules. """
        cons.timer.startTimeDeletion()
        while self.microPopSize > cons.N:
            self.deleteFromPopulation()
        cons.timer.stopTimeDeletion()

    def deleteFromPopulation(self):
        """ Deletes one classifier in the population.  The classifier that will be deleted is chosen by roulette wheel selection
        considering the deletion vote. Returns the macro-classifier which got decreased by one micro-classifier. """
        meanFitness = self.getPopFitnessSum()/float(self.microPopSize)

        #Calculate total wheel size------------------------------
        sumCl = 0.0
        voteList = []
        for cl in self.popSet:
            vote = cl.getDelProp(meanFitness)
            sumCl += vote
            voteList.append(vote)
        #--------------------------------------------------------
        choicePoint = sumCl * random.random() #Determine the choice point

        newSum=0.0
        for i in range(len(voteList)):
            cl = self.popSet[i]
            newSum = newSum + voteList[i]
            if newSum > choicePoint or newSum == float('Inf'): #Select classifier for deletion
                #Delete classifier----------------------------------
                cl.updateNumerosity(-1)
                self.microPopSize -= 1
                if cl.numerosity < 1: # When all micro-classifiers for a given classifier have been depleted.
                    self.removeMacroClassifier(i)
                    self.deleteFromMatchSet(i)
                    self.deleteFromCorrectSet(i)
                return
        print(choicePoint)
        print(sumCl)
        print(newSum)
        print("ClassifierSet: No eligible rules found for deletion in deleteFrom population.")
        return

#
#     def deleteFromPopulation(self, exploreIter):
#         """ Deletes one classifier in the population.  The classifier that will be deleted is chosen by roulette wheel selection
#         considering the deletion vote. Returns the macro-classifier which got decreased by one micro-classifier. """
#         meanFitness = self.getPopFitnessSum()/float(self.microPopSize)
#
#         Epoch Pool Deletion:------------------------------------
#         Three situations to deal with: too many EC - delete from EC: EC full - delete from ENC :  EC not full - treat all equal
#
#         maxEpochCompletePool = int(cons.N *0.5) #Half the rule pop is reserved for complete rules
#         deleteFromEC = False
#         deleteFromENC = False
#         quickDelete = False
#          print 'new'
#          print self.ECPopSize
#          print self.ENCPopSize
#         if self.ECPopSize > maxEpochCompletePool:
#             cons.epochPoolFull = True  #One time switch - once full it should stay full.
#             deleteFromEC = True
#             print 'deleteFromEC'
#         else:
#             if self.ECPopSize == maxEpochCompletePool:
#                 deleteFromENC = True
#                 print 'deleteFromENC'
#         Calculate total wheel size------------------------------
#         sumCl = 0.0
#         voteList = []
#         x = 0
#
#         for cl in self.popSet:
#             vote = cl.getDelProp(meanFitness)
#             vote = cl.getDeletionVote()
#             if vote[1]:
#                 quickDelete = True
#                 break
#             else:
#                 sumCl += vote[0]
#                 voteList.append(vote[0])
#             x += 1
#
#
#         choicePoint = sumCl * random.random() #Determine the choice point
#         newSum=0.0
#         for i in range(len(voteList)):
#             cl = self.popSet[i]
#             newSum = newSum + voteList[i]
#             if newSum > choicePoint: #Select classifier for deletion
#                 Delete classifier----------------------------------
#                 cl.updateNumerosity(-1)
#                 self.microPopSize -= 1
#                 if cl.epochComplete:
#                     self.ECPopSize -= 1
#                 else:
#                     self.ENCPopSize -= 1
#
#
#                 if cl.numerosity < 1: # When all micro-classifiers for a given classifier have been depleted.
#                     self.removeMacroClassifier(i)
#                     self.deleteFromMatchSet(i)
#                     self.deleteFromCorrectSet(i)
#                 return
#
#          if quickDelete:
#              cl = self.popSet[x]
#              self.microPopSize -= cl.numerosity
#              if cl.epochComplete:
#                  self.ECPopSize -= cl.numerosity
#              else:
#                  self.ENCPopSize -= cl.numerosity
#              self.removeMacroClassifier(x)
#              self.deleteFromMatchSet(x)
#              self.deleteFromCorrectSet(x)
#          else:
#              choicePoint = sumCl * random.random() #Determine the choice point
#              newSum=0.0
#              for i in range(len(voteList)):
#                  cl = self.popSet[i]
#                  newSum = newSum + voteList[i]
#                  if newSum > choicePoint: #Select classifier for deletion
#                      #Delete classifier----------------------------------
#                      cl.updateNumerosity(-1)
#                      self.microPopSize -= 1
#                      if cl.epochComplete:
#                          self.ECPopSize -= 1
#                      else:
#                          self.ENCPopSize -= 1
#
#
#                      if cl.numerosity < 1: # When all micro-classifiers for a given classifier have been depleted.
#                          self.removeMacroClassifier(i)
#                          self.deleteFromMatchSet(i)
#                          self.deleteFromCorrectSet(i)
#                      return
#
#             print "ClassifierSet: No eligible rules found for deletion in deleteFrom population."


#    def deleteFromPopulation(self, exploreIter):
#        """ Deletes one classifier in the population.  The classifier that will be deleted is chosen by roulette wheel selection
#        considering the deletion vote. Returns the macro-classifier which got decreased by one micro-classifier. """
#        meanFitness = self.getPopFitnessSum()/float(self.microPopSize)
#
#        #Epoch Pool Deletion:------------------------------------
#        #Three situations to deal with: too many EC - delete from EC: EC full - delete from ENC :  EC not full - treat all equal
#
#        maxEpochCompletePool = int(cons.N *0.5) #Half the rule pop is reserved for complete rules
#        deleteFromEC = False
#        deleteFromENC = False
#        quickDelete = False
##         print 'new'
##         print self.ECPopSize
##         print self.ENCPopSize
#        if self.ECPopSize > maxEpochCompletePool:
#            cons.epochPoolFull = True  #One time switch - once full it should stay full.
#            deleteFromEC = True
#            #print 'deleteFromEC'
#        else:
#            if self.ECPopSize == maxEpochCompletePool:
#                deleteFromENC = True
#                #print 'deleteFromENC'
#
#        #Calculate total wheel size------------------------------

#        sumCl = 0.0
#        voteList = []
#        if deleteFromEC:
#            for cl in self.popSet:
#                if cl.epochComplete:
#                    vote = cl.getDelProp(meanFitness)
#                    #vote = cl.getDeletionVote()
#                    sumCl += vote
#                    voteList.append(vote)
#                else:
#                    voteList.append(0)
#        elif deleteFromENC:
#            for cl in self.popSet:
#                if cl.epochComplete:
#                    voteList.append(0)
#                else:
#                    vote = cl.getDelProp(meanFitness)
#                    #vote = cl.getDeletionVote()
#                    sumCl += vote
#                    voteList.append(vote)
#        else: #All rules treated equally
#            for cl in self.popSet:
#                vote = cl.getDelProp(meanFitness)
#                #vote = cl.getDeletionVote()
#                sumCl += vote
#                voteList.append(vote)
#
#        #--------------------------------------------------------
#        choicePoint = sumCl * random.random() #Determine the choice point
#
#        newSum=0.0
#        for i in range(len(voteList)):
#            cl = self.popSet[i]
#            newSum = newSum + voteList[i]
#            if newSum > choicePoint: #Select classifier for deletion
#                #Delete classifier----------------------------------
#                cl.updateNumerosity(-1)
#                self.microPopSize -= 1
#                if cl.epochComplete:
#                    self.ECPopSize -= 1
#                else:
#                    self.ENCPopSize -= 1
#
#
#                if cl.numerosity < 1: # When all micro-classifiers for a given classifier have been depleted.
#                    self.removeMacroClassifier(i)
#                    self.deleteFromMatchSet(i)
#                    self.deleteFromCorrectSet(i)
#                return
#
#        print "ClassifierSet: No eligible rules found for deletion in deleteFrom population."


    def removeMacroClassifier(self, ref):
        """ Removes the specified (macro-) classifier from the population. """
        self.popSet.pop(ref)


    def deleteFromMatchSet(self, deleteRef):
        """ Delete reference to classifier in population, contained in self.matchSet."""
        if deleteRef in self.matchSet:
            self.matchSet.remove(deleteRef)
        #Update match set reference list--------
        for j in range(len(self.matchSet)):
            ref = self.matchSet[j]
            if ref > deleteRef:
                self.matchSet[j] -= 1


    def deleteFromCorrectSet(self, deleteRef):
        """ Delete reference to classifier in population, contained in self.matchSet."""
        if deleteRef in self.correctSet:
            self.correctSet.remove(deleteRef)
        #Update match set reference list--------
        for j in range(len(self.correctSet)):
            ref = self.correctSet[j]
            if ref > deleteRef:
                self.correctSet[j] -= 1

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GENETIC ALGORITHM
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def runGA(self, exploreIter, state, phenotype):
        """ The genetic discovery mechanism in ExSTraCS is controlled here. """
        #-------------------------------------------------------
        # GA RUN REQUIREMENT
        #-------------------------------------------------------
        if (exploreIter - self.getIterStampAverage()) < cons.theta_GA:  #Does the correct set meet the requirements for activating the GA?
            return
        #print "GA RUNNING ###############################################################"
        self.setIterStamps(exploreIter) #Updates the iteration time stamp for all rules in the correct set (which the GA operates on).
        changed = False
        #-------------------------------------------------------
        # SELECT PARENTS - Niche GA - selects parents from the correct class
        #-------------------------------------------------------
        cons.timer.startTimeSelection()
        selectList = self.selectClassifierRW()
        clP1 = selectList[0]
        clP2 = selectList[1]
        #test selection for tree and rule crossover

        """
        if cons.selectionMethod == "roulette":
            selectList = self.selectClassifierRW()
            clP1 = selectList[0]
            clP2 = selectList[1]
        elif cons.selectionMethod == "tournament":
            #selectList = self.selectClassifierT()
            selectList = self.selectClassifierT(exploreIter)
            clP1 = selectList[0]
            clP2 = selectList[1]
        else:
            print "ClassifierSet: Error - requested GA selection method not available."
        """
        cons.timer.stopTimeSelection()
        
        #START GP INTEGRATION CODE*************************************************************************************************************************************
        #-------------------------------------------------------
        # INITIALIZE OFFSPRING
        #-------------------------------------------------------
        if clP1.isTree:
            # cl1 = Tree(clP1, exploreIter) ## For older Deap code
            cl1 = tree_Clone(clP1, exploreIter)
        else:
            cl1 = Classifier(clP1, exploreIter)
        if clP2 == None:  #If there was only one parent - then both 'parents' will be from the same source.  No reason to do crossover if this is the case, only mutation.
            #print("Only one parent available")
            if clP1.isTree:
                # cl2 = Tree(clP1, exploreIter)   ## For older Deap code
                cl2 = tree_Clone(clP1, exploreIter)
            else:
                cl2 = Classifier(clP1, exploreIter)
        else:
            if clP2.isTree:
                # cl2 = Tree(clP2, exploreIter)   ## For older Deap code
                cl2 = tree_Clone(clP2, exploreIter);
            else:
                cl2 = Classifier(clP2, exploreIter)
        
        #COUNTERS------------------ TEMPORARY
        if cl1.isTree and cl2.isTree: #both entities are trees
            self.tree_cross_count += 1
        elif not cl1.isTree and not cl2.isTree:
            self.rule_cross_count += 1
        else:
            self.both_cross_count += 1
            
        #-------------------------------------------------------
        # CROSSOVER OPERATOR - Uniform Crossover Implemented (i.e. all attributes have equal probability of crossing over between two parents)
        #-------------------------------------------------------
        if cl1.equals(cl2):
            #print("it happened")
            pass
        if not cl1.equals(cl2) and random.random() < cons.chi:  #If parents are the same don't do crossover.
            cons.timer.startTimeCrossover()
            #print('------------------------------Performing Crossover')
            
            #REPORTING CROSSOVER EVENTS!------------TEMP
            if cl1.isTree and cl2.isTree:
                #print('Crossing 2 Trees')
                #print(str(cl1.form)+str(cl1.specifiedAttList))
                #print(str(cl2.form)+str(cl2.specifiedAttList))
                pass
            elif not cl1.isTree and not cl2.isTree:
                #print('Crossing 2 Rules')
                pass
            else: #one of each
                #print('Crossing a Tree with a Rule')
                if cl1.isTree:
                    #print(str(cl1.form)+str(cl1.specifiedAttList))
                    #print(str(cl2.condition)+str(cl2.specifiedAttList))
                    pass
                else:
                    #print(str(cl2.form)+str(cl2.specifiedAttList))
                    #print(str(cl1.condition)+str(cl1.specifiedAttList))
                    pass
            #--------------------------------------------------
            
            changed = cl1.uniformCrossover(cl2,state, phenotype) #PERFORM CROSSOVER!  #calls either the rule or tree crossover method depending on whether cl1 is tree or rule.
            cons.timer.stopTimeCrossover()
        #-------------------------------------------------------
        # INITIALIZE KEY OFFSPRING PARAMETERS
        #-------------------------------------------------------
        if changed:
            cl1.setFitness(cons.fitnessReduction * (cl1.fitness + cl2.fitness)/2.0)
            cl2.setFitness(cl1.fitness)
        else:
            cl1.setFitness(cons.fitnessReduction * cl1.fitness)
            cl2.setFitness(cons.fitnessReduction * cl2.fitness)
        #-------------------------------------------------------
        # MUTATION OPERATOR
        #-------------------------------------------------------
        #cons.timer.startTimeMutation()
        #nowchanged = cl1.Mutation(state, phenotype)
        #howaboutnow = cl2.Mutation(state, phenotype)
        nowchanged = True
        howaboutnow = True
        #cons.timer.stopTimeMutation()
        
        #Get current data point evaluation on new tree - determine phenotype and detemine if correct. 
        if cl1.isTree:
            #Get phenotype for current instance
            cl1.setPhenotype(state)
            #Update correct count accordingly.
            cl1.updateClonePhenotype(phenotype)
            
        if cl2.isTree:
            #Get phenotype for current instance
            cl2.setPhenotype(state)
            #Update correct count accordingly.
            cl2.updateClonePhenotype(phenotype)
        #STOP GP INTEGRATION CODE*************************************************************************************************************************************
        """
        for cl in self.popSet:
            if cl.isTree:
                cl.Mutation(state, phenotype)
        """
        #print('Crossover output')
        if cl1.isTree:
            #print(str(cl1.form)+str(cl1.specifiedAttList))
            pass
        if cl2.isTree:
            #print(str(cl2.form)+str(cl2.specifiedAttList))
            pass
        
        #Generalize any continuous attributes that span then entire range observed in the dataset.
        if cons.env.formatData.continuousCount > 0:
            cl1.rangeCheck()
            cl2.rangeCheck()
        #-------------------------------------------------------
        # CONTINUOUS ENDPOINT - phenotype range probability correction
        #-------------------------------------------------------
        if not cons.env.formatData.discretePhenotype: #ContinuousCode #########################
            cl1.setPhenProb()
            cl2.setPhenProb()

        #-------------------------------------------------------
        # ADD OFFSPRING TO POPULATION
        #-------------------------------------------------------
        #print changed
        #print nowchanged
        #print howaboutnow
        #print "##############################################"

        if changed or nowchanged or howaboutnow:
            self.insertDiscoveredClassifiers(cl1, cl2, clP1, clP2, exploreIter) #Includes subsumption if activated.


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SELECTION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #test method to force tree and rule crossover for testing purposes
    def selectTreeRule(self):
        setList = copy.deepcopy(self.correctSet) #correct set is a list of reference IDs

        if len(setList) > 2:
            selectList = [None, None]
            tree_list = []
            rule_list = []
            for index in setList:
                cl = self.popSet[index]
                if cl.isTree:
                    tree_list.append(cl)
                else:
                    rule_list.append(cl)
            if len(tree_list) != 0:
                random_index = random.randrange(0,len(tree_list))
                selectList[0] = tree_list[random_index]
            else:
                raise NameError("Empty tree list")
            if len(rule_list) != 0:
                random_index = random.randrange(0,len(rule_list))
                selectList[1] = rule_list[random_index]
            else:
                raise NameError("Empty rule list")

        elif len(setList) == 2:
            selectList = [self.popSet[setList[0]],self.popSet[setList[1]]]
        elif len(setList) == 1:
            selectList = [self.popSet[setList[0]],self.popSet[setList[0]]]
        else:
            print("ClassifierSet: Error in parent selection.")

        return selectList



    def selectClassifierRW(self):
        """ Selects parents using roulette wheel selection according to the fitness of the classifiers. """
        setList = copy.deepcopy(self.correctSet) #correct set is a list of reference IDs
        if len(setList) > 2:
            selectList = [None, None]
            currentCount = 0
            while currentCount < 2:
                fitSum = self.getFitnessSum(setList)

                choiceP = random.random() * fitSum
                i=0
                sumCl = self.popSet[setList[i]].fitness
                while choiceP > sumCl:
                    i=i+1
                    sumCl += self.popSet[setList[i]].fitness

                selectList[currentCount] = self.popSet[setList[i]] #store reference to the classifier
                setList.remove(setList[i])
                currentCount += 1

        elif len(setList) == 2:
            selectList = [self.popSet[setList[0]],self.popSet[setList[1]]]
        elif len(setList) == 1:
            selectList = [self.popSet[setList[0]],self.popSet[setList[0]]]
        else:
            print("ClassifierSet: Error in parent selection.")

        return selectList


    def selectClassifierT(self,exploreIter):
        """  Selects parents using tournament selection according to the fitness of the classifiers. """
        setList = copy.deepcopy(self.correctSet) #correct set is a list of reference IDs
        if len(setList) > 2:
            selectList = [None, None]
            currentCount = 0
            while currentCount < 2:
                tSize = int(len(setList)*cons.theta_sel)
                posList = random.sample(setList,tSize)

                bestF = 0
                bestC = setList[0]
                percentExperience = None
                for j in posList:
                    if self.popSet[j].epochComplete:
                        percentExperience = 1.0
                    else:
                        #print exploreIter - self.popSet[j].initTimeStamp
                        percentExperience = (exploreIter - self.popSet[j].initTimeStamp) / float(cons.env.formatData.numTrainInstances)
                        if percentExperience <= 0 or percentExperience > 1:
                            print('tournament selection error')
                            print(percentExperience)
                    #if self.popSet[j].fitness > bestF:
                    if self.popSet[j].fitness*percentExperience > bestF:
                        bestF = self.popSet[j].fitness*percentExperience
                        bestC = j
                setList.remove(j) #select without re-sampling
                selectList[currentCount] = self.popSet[bestC]
                currentCount += 1
        elif len(setList) == 2:
            selectList = [self.popSet[setList[0]],self.popSet[setList[1]]]
        elif len(setList) == 1:
            selectList = [self.popSet[setList[0]],self.popSet[setList[0]]]
        else:
            print("ClassifierSet: Error in parent selection.")

        return selectList

    #Original Tournament Selection
#     def selectClassifierT(self):
#         """  Selects parents using tournament selection according to the fitness of the classifiers. """
#         setList = copy.deepcopy(self.correctSet) #correct set is a list of reference IDs
#         if len(setList) > 2:
#             selectList = [None, None]
#             currentCount = 0
#             while currentCount < 2:
#                 tSize = int(len(setList)*cons.theta_sel)
#                 posList = random.sample(setList,tSize)
#
#                 bestF = 0
#                 bestC = setList[0]
#                 for j in posList:
#                     if self.popSet[j].fitness > bestF:
#                         bestF = self.popSet[j].fitness
#                         bestC = j
#                 setList.remove(j) #select without re-sampling
#                 selectList[currentCount] = self.popSet[bestC]
#                 currentCount += 1
#         elif len(setList) == 2:
#             selectList = [self.popSet[setList[0]],self.popSet[setList[1]]]
#         elif len(setList) == 1:
#             selectList = [self.popSet[setList[0]],self.popSet[setList[0]]]
#         else:
#             print "ClassifierSet: Error in parent selection."
#
#         return selectList

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SUBSUMPTION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def subsumeClassifier(self, exploreIter, cl=None, cl1P=None, cl2P=None):
        """ Tries to subsume a classifier in the parents. If no subsumption is possible it tries to subsume it in the current set. """
        if cl.isTree:
            self.addGAClassifierToPopulation(cl, exploreIter)

        if cl1P!=None and cl1P.subsumes(cl):
            self.microPopSize += 1
#             if cl1P.epochComplete:
#                 self.ECPopSize += 1
#             else:
#                 self.ENCPopSize += 1
            cl1P.updateNumerosity(1)
        elif cl2P!=None and cl2P.subsumes(cl):
            self.microPopSize += 1
#             if cl2P.epochComplete:
#                 self.ECPopSize += 1
#             else:
#                 self.ENCPopSize += 1
            cl2P.updateNumerosity(1)
        else:
            self.subsumeClassifier2(cl, exploreIter); #Try to subsume in the correct set.


    def subsumeClassifier2(self, cl, exploreIter):
        """ Tries to subsume a classifier in the correct set. If no subsumption is possible the classifier is simply added to the population considering
        the possibility that there exists an identical classifier. """
        choices = []
        for ref in self.correctSet:
            if self.popSet[ref].subsumes(cl):
                choices.append(ref)

        if len(choices) > 0: #Randomly pick one classifier to be subsumer
            choice = int(random.random()*len(choices))
            self.popSet[choices[choice]].updateNumerosity(1)
            self.microPopSize += 1
#             if self.popSet[choices[choice]].epochComplete:
#                 self.ECPopSize += 1
#             else:
#                 self.ENCPopSize += 1
            cons.timer.stopTimeSubsumption()
            return

        cons.timer.stopTimeSubsumption()
        self.addGAClassifierToPopulation(cl, exploreIter) #If no subsumer was found, check for identical classifier, if not then add the classifier to the population


    def doCorrectSetSubsumption(self):
        """ Executes correct set subsumption.  The correct set subsumption looks for the most general subsumer classifier in the correct set
        and subsumes all classifiers that are more specific than the selected one. """
        subsumer = None
        for ref in self.correctSet:
            cl = self.popSet[ref]
            if cl.isSubsumer():
                if subsumer == None or cl.isMoreGeneral(subsumer):
                    subsumer = cl

        if subsumer != None: #If a subsumer was found, subsume all more specific classifiers in the correct set
            i=0
            while i < len(self.correctSet):
                ref = self.correctSet[i]
                if subsumer.isMoreGeneral(self.popSet[ref]):
                    subsumer.updateNumerosity(self.popSet[ref].numerosity)
#                     if subsumer.epochComplete:
#                         if not self.popSet[ref].epochComplete:
#                             self.ECPopSize += 1
#                             self.ENCPopSize -= 1
#                     else:
#                         if self.popSet[ref].epochComplete:
#                             self.ECPopSize -= 1
#                             self.ENCPopSize += 1

                    self.removeMacroClassifier(ref)
                    self.deleteFromMatchSet(ref)
                    self.deleteFromCorrectSet(ref)
                    i = i - 1
                i = i + 1

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER KEY METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#    def addClassifierToPopulation(self, cl, covering, exploreIter=None):
#        """ Adds a classifier to the set and increases the numerositySum value accordingly."""
#        cons.timer.startTimeAdd()
#        oldCl = None
#        if not covering:
#            oldCl = self.getIdenticalClassifier(cl)
#        if oldCl != None: #found identical classifier
#            oldCl.updateNumerosity(1)
#            self.microPopSize += 1
#        else:
#            #NEW Fitness-----------------------------
#            cl.calcClassifierStateFreq()  #Calculates classifier state frequency once when rule is added to the population from covering or GA
#            if not covering:
#                #GA rules will have an initial fitness calculated based on this first exposure
#                cl.updateFitness(exploreIter)
#            #-----------------------------------------
#            self.popSet.append(cl)
#            self.microPopSize += 1
#        cons.timer.stopTimeAdd()


    def addGAClassifierToPopulation(self, cl, exploreIter):
        """ Adds a classifier to the set and increases the numerositySum value accordingly."""

        #print "Classifier Added"
        cons.timer.startTimeAdd()
        oldCl = self.getIdenticalClassifier(cl)

        if oldCl != None: #found identical classifier
            oldCl.updateNumerosity(1)
            self.microPopSize += 1
#             if oldCl.epochComplete:
#                 self.ECPopSize += 1
#             else:
#                 self.ENCPopSize += 1
        else:
            #NEW Fitness-----------------------------
            #cl.calcClassifierStateFreq()  #Calculates classifier state frequency once when rule is added to the population from covering or GA
            #GA rules will have an initial fitness calculated based on this first exposure
            #cl.updateFitness(exploreIter)
            #-----------------------------------------

            self.popSet.append(cl)
            self.microPopSize += 1
#             if cl.epochComplete:
#                 self.ECPopSize += 1
#             else:
#                 self.ENCPopSize += 1
        cons.timer.stopTimeAdd()


    def addCoveredClassifierToPopulation(self, cl):
        """ Adds a classifier to the set and increases the numerositySum value accordingly."""
        cons.timer.startTimeAdd()
        #NEW Fitness-----------------------------
        #cl.calcClassifierStateFreq()  #Calculates classifier state frequency once when rule is added to the population from covering or GA
        #-----------------------------------------

        self.popSet.append(cl)
        self.microPopSize += 1
#         if cl.epochComplete:
#             self.ECPopSize += 1
#         else:
#             self.ENCPopSize += 1
        cons.timer.stopTimeAdd()

    #START GP INTEGRATION CODE*************************************************************************************************************************************
    def addClassifierForInit(self, state, phenotype):
        #temporarily turn off expert knowledge

        cl = Classifier(1, 0, state, phenotype)
        oldCl = self.getIdenticalClassifier(cl)
        if oldCl != None: #Copy found
            oldCl.numerosity += 1
        else: #Brand new rule
            #cl.updateExperience()
            self.popSet.append(cl)
        
        self.microPopSize += 1 #global (numerosity inclusive) popsize
            
        """
        count = 0
        while oldCl != None:
            cl = Classifier(1, 0, state, phenotype)
            oldCl = self.getIdenticalClassifier(cl)
            count+=1
            if count > 50:
                print("New Classifier: " + str(cl.specifiedAttList) + " || " + str(cl.condition))
                print("Old Classifier: " + str(oldCl.specifiedAttList) + " || " + str(oldCl.condition))

                cl = Classifier(1, 0, state, phenotype)
                print("Other Classifier: " + str(cl.specifiedAttList) + " || " + str(cl.condition))
                raise NameError("Classifier Covering")

        #cl.calcClassifierStateFreq()
        """


    #STOP GP INTEGRATION CODE*************************************************************************************************************************************

    def insertDiscoveredClassifiers(self, cl1, cl2, clP1, clP2, exploreIter):
        """ Inserts both discovered classifiers keeping the maximal size of the population and possibly doing GA subsumption.
        Checks for default rule (i.e. rule with completely general condition) prevents such rules from being added to the population. """
        #-------------------------------------------------------
        # SUBSUMPTION
        #-------------------------------------------------------
        if cons.doSubsumption:
            cons.timer.startTimeSubsumption()

            if cl1.isTree or len(cl1.specifiedAttList) > 0:
                self.subsumeClassifier(exploreIter, cl1, clP1, clP2)
            if cl2.isTree or len(cl2.specifiedAttList) > 0:
                self.subsumeClassifier(exploreIter, cl2, clP1, clP2)
        #-------------------------------------------------------
        # ADD OFFSPRING TO POPULATION
        #-------------------------------------------------------
        else:
            if cl1.isTree or len(cl1.specifiedAttList) > 0:
                self.addGAClassifierToPopulation(cl1, exploreIter)
            if cl2.isTree or len(cl2.specifiedAttList) > 0:
                self.addGAClassifierToPopulation(cl2, exploreIter)


#    def updateSets(self, exploreIter):
#        """ Updates all relevant parameters in the current match and correct sets. """
#        matchSetNumerosity = 0
#        for ref in self.matchSet:
#            matchSetNumerosity += self.popSet[ref].numerosity
#
#        for ref in self.matchSet:
#            self.popSet[ref].updateExperience()
#            self.popSet[ref].updateMatchSetSize(matchSetNumerosity) #Moved to match set to be like GHCS
#            if ref in self.correctSet:
#                self.popSet[ref].updateCorrect()
#
#            self.popSet[ref].updateAccuracy()
#            self.popSet[ref].updateFitness(exploreIter) #NEW FITNESS
#            #self.popSet[ref].updateFitness()

    #NEW
    def updateSets(self, exploreIter,trueEndpoint):
        """ Updates all relevant parameters in the current match and correct sets. """
        matchSetNumerosity = 0
#         preFitSumNEC = 0.0
#         preFitSumEC = 0.0
        #preFitSumList = [0.0]*cons.env.formatData.specLimit
        indFitSum = 0.0
        #weightSum = 0.0
        #print 'lenMatchSet= '+str(len(self.matchSet))
#        correctSetNumerosity = 0
        for ref in self.matchSet:
            matchSetNumerosity += self.popSet[ref].numerosity
        #Experimental!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#        for ref in self.correctSet:
#            correctSetNumerosity += self.popSet[ref].numerosity

        for ref in self.matchSet:
            self.popSet[ref].updateExperience()
            self.popSet[ref].updateMatchSetSize(matchSetNumerosity) #Moved to match set to be like GHCS
            if ref in self.correctSet:
                self.popSet[ref].updateCorrect()
                if not cons.env.formatData.discretePhenotype: #Continuous endpoint
                    self.popSet[ref].updateError(trueEndpoint)
            else: #Continuous endpoint gets Error added for not being in the correct set.
                if not cons.env.formatData.discretePhenotype: #Continuous endpoint
                    self.popSet[ref].updateIncorrectError()



            self.popSet[ref].updateAccuracy(exploreIter)
            self.popSet[ref].updateCorrectCoverage()
            self.popSet[ref].updateIndFitness(exploreIter)
            #preFitSumAll += self.popSet[ref].numerosity*self.popSet[ref].indFitness

            if ref in self.correctSet:
                if self.popSet[ref].epochComplete:
                    indFitSum += self.popSet[ref].numerosity*self.popSet[ref].indFitness
                    #indFitSum += self.popSet[ref].indFitness
                    #weightSum += 1.0
                else: #epoch not complete (rules contribution to indFitness is proportional to experience)
                    percRuleExp = (exploreIter-self.popSet[ref].initTimeStamp+1)/float(cons.env.formatData.numTrainInstances) #Weight for weighted average
                    indFitSum += self.popSet[ref].numerosity*self.popSet[ref].indFitness*percRuleExp
                    #indFitSum += self.popSet[ref].indFitness*percRuleExp
                    #weightSum += percRuleExp


#             if self.popSet[ref].epochComplete:
#                 if ref in self.correctSet:
#                     preFitSumEC += self.popSet[ref].numerosity*self.popSet[ref].indFitness
#             else:
#                 if ref in self.correctSet:
#                     preFitSumNEC += self.popSet[ref].numerosity*self.popSet[ref].indFitness
                    #print self.popSet[ref].numerosity*self.popSet[ref].indFitness



        #Fitness Sharing
        #indWeightedFitSum = indFitSum/float(weightSum)
        #print 'new'
        #print indFitSum
        #print weightSum
        #print indWeightedFitSum
        for ref in self.matchSet:
            if ref in self.correctSet:
                partOfCorrect = True
            else:
                partOfCorrect = False
            self.popSet[ref].updateRelativeIndFitness(indFitSum, partOfCorrect, exploreIter)
            #self.popSet[ref].updateRelativePreFitness(preFitSumNEC,preFitSumEC, partOfCorrect)
            self.popSet[ref].updateFitness(exploreIter)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def getIterStampAverage(self):
        """ Returns the average of the time stamps in the correct set. """
        sumCl=0.0
        numSum=0.0
        for i in range(len(self.correctSet)):
            ref = self.correctSet[i]
            sumCl += self.popSet[ref].timeStampGA * self.popSet[ref].numerosity
            numSum += self.popSet[ref].numerosity #numerosity sum of correct set
        return sumCl/float(numSum)


    def setIterStamps(self, exploreIter):
        """ Sets the time stamp of all classifiers in the set to the current time. The current time
        is the number of exploration steps executed so far.  """
        for i in range(len(self.correctSet)):
            ref = self.correctSet[i]
            self.popSet[ref].updateTimeStamp(exploreIter)


    def getFitnessSum(self, setList):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl=0.0
        for i in range(len(setList)):
            ref = setList[i]
            sumCl += self.popSet[ref].fitness
        return sumCl


    def getPopFitnessSum(self):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl=0.0
        for cl in self.popSet:
            sumCl += cl.fitness *cl.numerosity
        return sumCl


    def getIdenticalClassifier(self, newCl):
        """ Looks for an identical classifier in the population. """
        for cl in self.popSet:
            if newCl.equals(cl):
                return cl
        return None


    def clearSets(self):
        """ Clears out references in the match and correct sets for the next learning iteration. """
        self.matchSet = []
        self.correctSet = []


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # EVALUTATION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def runPopAveEval(self, exploreIter):
        """ Determines current generality of population """
        genSum = 0
        agedCount = 0

        #CHANGED_EVALUATION
        for cl in self.popSet:
            genSum += ((cons.env.formatData.numAttributes - len(cl.specifiedAttList)) / float(cons.env.formatData.numAttributes)) * cl.numerosity
            if (exploreIter - cl.initTimeStamp) > cons.env.formatData.numTrainInstances:
                agedCount += 1

        if self.microPopSize == 0:
            self.aveGenerality = 'NA'
            self.expRules = 'NA'
        else:
            self.aveGenerality = genSum / float(self.microPopSize)
            if cons.offlineData:
                self.expRules = agedCount / float(len(self.popSet))
            else:
                self.expRules = 'NA'
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        #-------------------------------------------------------
        ###CHANGED FOR TREE
        if not cons.env.formatData.discretePhenotype: #ContinuousCode #########################
            sumRuleRange = 0
            for cl in self.popSet:
                if not cl.isTree:
                    high = cl.phenotype[1]
                    low = cl.phenotype[0]
                    if high > cons.env.formatData.phenotypeList[1]:
                        high = cons.env.formatData.phenotypeList[1]
                    if low < cons.env.formatData.phenotypeList[0]:
                        low = cons.env.formatData.phenotypeList[0]
                    sumRuleRange += (cl.phenotype[1] - cl.phenotype[0])*cl.numerosity
            phenotypeRange = cons.env.formatData.phenotypeList[1] - cons.env.formatData.phenotypeList[0]
            self.avePhenotypeRange = (sumRuleRange / float(self.microPopSize)) / float(phenotypeRange)


    def runAttGeneralitySum(self):
        """ Determine the population-wide frequency of attribute specification, and accuracy weighted specification. """
        self.attributeSpecList = []
        self.attributeAccList = []
        for i in range(cons.env.formatData.numAttributes):
            self.attributeSpecList.append(0)
            self.attributeAccList.append(0.0)
        for cl in self.popSet:
            for ref in cl.specifiedAttList:
                self.attributeSpecList[ref] += cl.numerosity
                self.attributeAccList[ref] += cl.numerosity * cl.accuracy


    def recalculateNumerositySum(self):
        """ Recalculate the NumerositySum after rule compaction. """
        self.microPopSize = 0
        for cl in self.popSet:
            self.microPopSize += cl.numerosity


    def getPopTrack(self, accuracy, exploreIter, trackingFrequency):
        """ Returns a formated output string to be printed to the Learn Track output file. """
        
        #GP integration code!!!
        numTrees = 0
        numRules = 0
        for i in range(len(self.popSet)):
            cl = self.popSet[i]
            if cl.isTree:
                numTrees += 1
            else:
                numRules += 1
        
        
        trackString = str(int(exploreIter/trackingFrequency)) + "\t" + str(exploreIter)+ "\t" + str(len(self.popSet)) + "\t" + str(self.microPopSize) + "\t" + str(accuracy) + "\t" + str(numRules)  + "\t" + str(numTrees) + "\t" + str(self.aveGenerality) + "\t" + str(self.expRules)  + "\t" + str(cons.timer.returnGlobalTimer())+ "\n"
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        if cons.env.formatData.discretePhenotype:
            print(("Epoch: "+str(int(exploreIter/trackingFrequency))+"\t Iteration: " + str(exploreIter) + "\t MacroPop: " + str(len(self.popSet))+ "\t MicroPop: " + str(self.microPopSize) + "\t AccEstimate: " + str(accuracy)+ "\t RuleCount: " + str(numRules)+ "\t TreeCount: " + str(numTrees) + "\t AveGen: " + str(self.aveGenerality) + "\t ExpRules: " + str(self.expRules)  + "\t Time: " + str(cons.timer.returnGlobalTimer())))
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        #-------------------------------------------------------
        else:
            print(("Epoch: "+str(int(exploreIter/trackingFrequency))+"\t Iteration: " + str(exploreIter) + "\t MacroPop: " + str(len(self.popSet))+ "\t MicroPop: " + str(self.microPopSize) + "\t AccEstimate: " + str(accuracy)+ "\t RuleCount: " + str(numRules)+ "\t TreeCount: " + str(numTrees) + "\t AveGen: " + str(self.aveGenerality) + "\t ExpRules: " + str(self.expRules) + "\t PhenRange: " +str(self.avePhenotypeRange) + "\t Time: " + str(cons.timer.returnGlobalTimer())))


        return trackString