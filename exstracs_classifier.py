"""
Name:        ExSTraCS_Classifier.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: This module defines an individual classifier within the rule population, along with all respective parameters.
             Also included are classifier-level methods, including constructors(covering, copy, reboot) matching, subsumption,
             crossover, and mutation.  Parameter update methods are also included.

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

from exstracs_constants import *
from exstracs_pareto import *
import random
import copy
import math
import ast

class Classifier:
    def __init__(self,a=None,b=None,c=None,d=None):


        self.errorSum = 0
        self.errorCount = 0
        self.totalFreq = 1
        self.isTree = False
        #Major Parameters --------------------------------------------------
        self.specifiedAttList = []      # Attribute Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
        self.condition = []             # States of Attributes Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
        self.phenotype = None           # Class if the endpoint is discrete, and a continuous phenotype if the endpoint is continuous
        self.phenotype_RP = None        #NEW - probability of this   occurring by chance.

        #Fitness Metrics ---------------------------------------------------
        self.accuracy = 0.0             # Classifier accuracy - Accuracy calculated using only instances in the dataset which this rule matched.
        self.accuracyComponent = 0.0   # Accuracy adjusted based on accuracy by random chance, and transformed to give more weight to early changes in accuracy.
        self.coverDiff = 1          #Number of instance correctly covered by rule beyond what would be expected by chance.
        self.indFitness = 0.0           # Fitness from the perspective of an individual rule (strength/accuracy based)
        self.relativeIndFitness = None
        self.fitness = 0.01    #CHANGED: Original self.fitness = cons.init_fit
                                        # Classifier fitness - initialized to a constant initial fitness value

        self.numerosity = 1             # The number of rule copies stored in the population.  (Indirectly stored as incremented numerosity)
        self.aveMatchSetSize = None     # A parameter used in deletion which reflects the size of match sets within this rule has been included.
        self.deletionVote = None        # The current deletion weight for this classifier.

        #Experience Management ---------------------------------------------
        self.epochComplete = False      # Has this rule existed for a complete epoch (i.e. a cycle through training set).

        #Fitness Metrics ---------------------------------------------------
        #self.freqComponent = None

        #Fitness Sharing
        #self.adjustedAccuracy = 0.0
        #self.relativeAccuracy = 0.0
        self.lastMatch = 0

        self.lastFitness = 0.0
        self.sumIndFitness = 1.0  #experimental
        self.partOfCorrect = True
        self.lastMatchFitness = 1.0
        #self.isMatchSetFitness = False #For Debugging
        self.aveRelativeIndFitness = None
        self.matchedAndFrontEstablished = False

        if isinstance(a, tuple):
            self.dummyCreate(a)
        elif isinstance(c,list):
            self.classifierCovering(a,b,c,d)
        elif isinstance(a,Classifier):
            self.classifierCopy(a, b)
        elif isinstance(a,list) and b == None:
            self.rebootClassifier(a)
        else:
            print("Classifier: Error building classifier.")


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER CONSTRUCTION METHODS
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #dummy constructor used
    def dummyCreate(self, attributes):
        self.specifiedAttList = attributes[0]
        self.condition = attributes[1]

    def classifierCovering(self, setSize, exploreIter, state, phenotype):
        """ Makes a new classifier when the covering mechanism is triggered.  The new classifier will match the current training instance.
        Covering will NOT produce a default rule (i.e. a rule with a completely general condition). """
        #print 'covering made me'
        #print phenotype
        #Initialize new classifier parameters----------
        self.timeStampGA = exploreIter      # Time since rule last in a correct set.
        self.initTimeStamp = exploreIter    # Iteration in which the rule first appeared.
        self.aveMatchSetSize = setSize

        self.lastMatch = exploreIter        #Experimental - for brief fitness update
        #Classifier Accuracy Tracking --------------------------------------
        self.matchCount = 1             # Known in many LCS implementations as experience i.e. the total number of times this classifier was in a match set
        self.correctCount = 1           # The total number of times this classifier was in a correct set
        self.matchCover = 1             # The total number of times this classifier was in a match set within a single epoch. (value fixed after epochComplete)
        self.correctCover = 1           # The total number of times this classifier was in a correct set within a single epoch. (value fixed after epochComplete)

        #Covering sets initially overly optimistic prediction values - this takes care of error with prediction which previously had only zero value fitness and indFitness scores for covered rules.
        self.indFitness = 1.0
        self.fitness = 1.0

        dataInfo = cons.env.formatData
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        if dataInfo.discretePhenotype:
            self.phenotype = phenotype
            self.phenotype_RP = cons.env.formatData.classProportions[self.phenotype]
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        #-------------------------------------------------------
        else: #ContinuousCode #########################
            phenotypeRange = dataInfo.phenotypeList[1] - dataInfo.phenotypeList[0]
            rangeRadius = random.randint(25,75)*0.01*phenotypeRange / 2.0 #Continuous initialization domain radius.
            Low = float(phenotype) - rangeRadius
            High = float(phenotype) + rangeRadius
            self.phenotype = [Low,High] #ALKR Representation, Initialization centered around training instance  with a range between 25 and 75% of the domain size.
            self.setPhenProb()
        #-------------------------------------------------------
        # GENERATE MATCHING CONDITION - With Expert Knowledge Weights
        #-------------------------------------------------------
        #DETERMINISTIC STRATEGY
        if cons.useExpertKnowledge:
            toSpecify = random.randint(1,dataInfo.specLimit) # Pick number of attributes to specify
            i = 0

            while len(self.specifiedAttList) < toSpecify:
                target = cons.EK.EKRank[i]
                if state[target] != cons.labelMissingData: # If one of the randomly selected specified attributes turns out to be a missing data point, generalize instead.
                    self.specifiedAttList.append(target)
                    self.condition.append(self.buildMatch(target, state))
                i += 1
        #-------------------------------------------------------
        # GENERATE MATCHING CONDITION - Without Expert Knowledge Weights
        #-------------------------------------------------------
        else:
            toSpecify = random.randint(1,dataInfo.specLimit) # Pick number of attributes to specify
            potentialSpec = random.sample(range(dataInfo.numAttributes),toSpecify) # List of possible specified attributes
            for attRef in potentialSpec:
                if state[attRef] != cons.labelMissingData: # If one of the randomly selected specified attributes turns out to be a missing data point, generalize instead.
                    self.specifiedAttList.append(attRef)
                    self.condition.append(self.buildMatch(attRef, state))


    def setPhenProb(self):
        """ Calculate the probability that the phenotype of a given instance in the training data will fall withing the phenotype range specified by this rule. """
        count = 0
        ref = 0
#         print cons.env.formatData.phenotypeRanked
#         print self.phenotype
        while ref < len(cons.env.formatData.phenotypeRanked) and cons.env.formatData.phenotypeRanked[ref] <= self.phenotype[1]:
            if cons.env.formatData.phenotypeRanked[ref] >= self.phenotype[0]:
                count += 1
            ref += 1

        self.phenotype_RP = count/float(cons.env.formatData.numTrainInstances)



    def classifierCopy(self, clOld, exploreIter):
        """  Constructs an identical Classifier.  However, the experience of the copy is set to 0 and the numerosity
        is set to 1 since this is indeed a new individual in a population. Used by the genetic algorithm to generate
        offspring based on parent classifiers."""
        self.specifiedAttList = copy.deepcopy(clOld.specifiedAttList)
        self.condition = copy.deepcopy(clOld.condition)
        self.phenotype = copy.deepcopy(clOld.phenotype)
        if cons.env.formatData.discretePhenotype:
            self.phenotype_RP = copy.deepcopy(clOld.phenotype_RP)
        else:
            self.phenotype_RP = None #This will change if phenotype range changes when GA operates.
        self.timeStampGA = exploreIter  #consider starting at 0 instead???
        self.initTimeStamp = exploreIter
        self.lastMatch = exploreIter
        self.aveMatchSetSize = copy.deepcopy(clOld.aveMatchSetSize)
        self.fitness = clOld.fitness #Test removal
        self.accuracy = 1.0
        self.relativeIndFitness = 1.0
        self.indFitness = 1.0
        self.sumIndFitness = 1.0
#         self.sumPreFitnessNEC = 1.0
#         self.sumPreFitnessEC = 1.0
        self.accuracyComponent = 1.0
        self.matchCount = 1             # Known in many LCS implementations as experience i.e. the total number of times this classifier was in a match set
        self.correctCount = 1           # The total number of times this classifier was in a correct set
        self.matchCover = 1             # The total number of times this classifier was in a match set within a single epoch. (value fixed after epochComplete)
        self.correctCover = 1           # The total number of times this classifier was in a correct set within a single epoch. (value fixed after epochComplete)


    def rebootClassifier(self, classifierList):
        """ Rebuilds a saved classifier as part of the population Reboot """
        self.specifiedAttList = ast.literal_eval(classifierList[0])
        self.condition = ast.literal_eval(classifierList[1])
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        if cons.env.formatData.discretePhenotype:
            self.phenotype = str(classifierList[2])
            self.phenotype_RP = cons.env.formatData.classProportions[self.phenotype]
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        #-------------------------------------------------------
        else: #ContinuousCode #########################
            self.phenotype = ast.literal_eval(classifierList[2])
            for i in range(2): #Make values into floats.
                self.phenotype[i] = float(self.phenotype[i])
            self.setPhenProb() #Get the proportion of instances in training data that fall

        self.fitness = float(classifierList[3])
        self.accuracy = float(classifierList[4])
        self.numerosity = int(classifierList[5])
        self.aveMatchSetSize = float(classifierList[6])
        self.timeStampGA = int(classifierList[7])
        self.initTimeStamp = int(classifierList[8])

        if str(classifierList[10]) == 'None':
            self.deletionVote = None
        else:
            self.deletionVote = float(classifierList[10])
        self.correctCount = int(classifierList[11])
        self.matchCount = int(classifierList[12])
        self.correctCover = int(classifierList[13])
        self.matchCover = int(classifierList[14])
        self.epochComplete = bool(classifierList[15])
        #self.freqComponent = float(classifierList[21])
        self.accuracyComponent = float(classifierList[16])
#         print classifierList[16]
#         print classifierList[17]
#         print classifierList[18]
#         print classifierList[19]
        if str(classifierList[17]) == 'None':
            self.coverDiff = None
        else:
            self.coverDiff = float(classifierList[17])
        self.indFitness = float(classifierList[18])
        self.fitness = float(classifierList[19])
        self.lastMatch = 5/0 #fix later


    def selectAttributeRW(self, toSpecify):
        """ Selects attributes to be specified in classifier covering using Expert Knowledge weights, and roulette wheel selection. """
        scoreRefList = copy.deepcopy(cons.EK.refList) #correct set is a list of reference IDs
        selectList = []
        currentCount = 0
        totalSum = copy.deepcopy(cons.EK.EKSum)
        while currentCount < toSpecify:
            choicePoint = random.random() * totalSum
            i=0
            sumScore = cons.EK.scores[scoreRefList[i]]
            while choicePoint > sumScore:
                i=i+1
                sumScore += cons.EK.scores[scoreRefList[i]]
            selectList.append(scoreRefList[i])
            totalSum -= cons.EK.scores[scoreRefList[i]]
            scoreRefList.remove(scoreRefList[i])
            currentCount += 1
        return selectList


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # MATCHING
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def match(self, state):
        """ Returns if the classifier matches in the current situation. """
        for i in range(len(self.condition)):
            attributeInfo = cons.env.formatData.attributeInfo[self.specifiedAttList[i]]
            #-------------------------------------------------------
            # CONTINUOUS ATTRIBUTE
            #-------------------------------------------------------
            if attributeInfo[0]:
                instanceValue = state[self.specifiedAttList[i]]
                if self.condition[i][0] < instanceValue < self.condition[i][1] or instanceValue == cons.labelMissingData:
                    pass
                else:
                    return False
            #-------------------------------------------------------
            # DISCRETE ATTRIBUTE
            #-------------------------------------------------------
            else:
                stateRep = state[self.specifiedAttList[i]]
                if stateRep == self.condition[i] or stateRep == cons.labelMissingData:
                    pass
                else:
                    return False
        return True


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GENETIC ALGORITHM MECHANISMS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def uniformCrossover(self, cl, state, phenotype):
        if cl.isTree:
            cl.uniformCrossover(self, state, phenotype)
            #change this when get crossover to return bool
            return True
        else:
            return self.ruleCrossover(cl, state, phenotype)

    def ruleCrossover(self, cl, state, phenotype):
        """ Applies uniform crossover and returns if the classifiers changed. Handles both discrete and continuous attributes.
        #SWARTZ: self. is where for the better attributes are more likely to be specified
        #DEVITO: cl. is where less useful attribute are more likely to be specified
        """
        if cons.env.formatData.discretePhenotype or random.random() < 0.5: #Always crossover condition if the phenotype is discrete (if continuous phenotype, half the time phenotype crossover is performed instead)
            p_self_specifiedAttList = copy.deepcopy(self.specifiedAttList)
            p_cl_specifiedAttList = copy.deepcopy(cl.specifiedAttList)

            useAT = False
            if cons.doAttributeFeedback and random.random() < cons.AT.percent:
                useAT = True

            #Make list of attribute references appearing in at least one of the parents.-----------------------------
            comboAttList = []
            for i in p_self_specifiedAttList:
                comboAttList.append(i)
            for i in p_cl_specifiedAttList:
                if i not in comboAttList:
                    comboAttList.append(i)
                elif not cons.env.formatData.attributeInfo[i][0]: #Attribute specified in both parents, and the attribute is discrete (then no reason to cross over)
                    comboAttList.remove(i)
            comboAttList.sort()
            #--------------------------------------------------------------------------------------------------------
            changed = False;
            for attRef in comboAttList:
                attributeInfo = cons.env.formatData.attributeInfo[attRef]
                #-------------------------------------------------------
                # ATTRIBUTE CROSSOVER PROBAILITY - ATTRIBUTE FEEDBACK
                #-------------------------------------------------------
                if useAT:
                    probability = cons.AT.getTrackProb()[attRef]
                #-------------------------------------------------------
                # ATTRIBUTE CROSSOVER PROBAILITY - NORMAL CROSSOVER
                #-------------------------------------------------------
                else:
                    probability = 0.5  #Equal probability for attribute alleles to be exchanged.
                #-----------------------------
                ref = 0
                if attRef in p_self_specifiedAttList:
                    ref += 1
                if attRef in p_cl_specifiedAttList:
                    ref += 1

                if ref == 0:    #This should never happen:  All attributes in comboAttList should be specified in at least one classifier.
                    print("Error: UniformCrossover!")
                    pass
                #-------------------------------------------------------
                # CROSSOVER
                #-------------------------------------------------------
                elif ref == 1:  #Attribute specified in only one condition - do probabilistic switch of whole attribute state (Attribute type makes no difference)
                    if attRef in p_self_specifiedAttList and random.random() > probability: # If attribute specified in SWARTZ and high probability of being valuable, then less likely to swap.
                        i = self.specifiedAttList.index(attRef) #reference to the position of the attribute in the rule representation
                        cl.condition.append(self.condition.pop(i)) #Take attribute from self and add to cl
                        cl.specifiedAttList.append(attRef)
                        self.specifiedAttList.remove(attRef)
                        changed = True #Remove att from self and add to cl

                    if attRef in p_cl_specifiedAttList and random.random() < probability: # If attribute specified in DEVITO and high probability of being valuable, then more likely to swap.
                        i = cl.specifiedAttList.index(attRef) #reference to the position of the attribute in the rule representation
                        self.condition.append(cl.condition.pop(i)) #Take attribute from self and add to cl
                        self.specifiedAttList.append(attRef)
                        cl.specifiedAttList.remove(attRef)
                        changed = True #Remove att from cl and add to self.

                else: #Attribute specified in both conditions - do random crossover between state alleles - Notice: Attribute Feedback must not be used to push alleles together within an attribute state.
                    #The same attribute may be specified at different positions within either classifier
                    #-------------------------------------------------------
                    # CONTINUOUS ATTRIBUTE
                    #-------------------------------------------------------
                    if attributeInfo[0]:
                        i_cl1 = self.specifiedAttList.index(attRef) #pairs with self (classifier 1)
                        i_cl2 = cl.specifiedAttList.index(attRef)   #pairs with cl (classifier 2)
                        tempKey = random.randint(0,3) #Make random choice between 4 scenarios, Swap minimums, Swap maximums, Self absorbs cl, or cl absorbs self.
                        if tempKey == 0:    #Swap minimum
                            temp = self.condition[i_cl1][0]
                            self.condition[i_cl1][0] = cl.condition[i_cl2][0]
                            cl.condition[i_cl2][0] = temp
                        elif tempKey == 1:  #Swap maximum
                            temp = self.condition[i_cl1][1]
                            self.condition[i_cl1][1] = cl.condition[i_cl2][1]
                            cl.condition[i_cl2][1] = temp
                        else: #absorb range
                            allList = self.condition[i_cl1] + cl.condition[i_cl2]
                            newMin = min(allList)
                            newMax = max(allList)
                            if tempKey == 2:  #self absorbs cl
                                self.condition[i_cl1] = [newMin,newMax]
                                #Remove cl
                                cl.condition.pop(i_cl2)
                                cl.specifiedAttList.remove(attRef)
                            else:             #cl absorbs self
                                cl.condition[i_cl2] = [newMin,newMax]
                                #Remove self
                                self.condition.pop(i_cl1)
                                self.specifiedAttList.remove(attRef)
                    #-------------------------------------------------------
                    # DISCRETE ATTRIBUTE
                    #-------------------------------------------------------
                    else:
                        pass

            #-------------------------------------------------------
            # SPECIFICATION LIMIT CHECK - return specificity to limit. Note that it is possible for completely general rules to result from crossover - (mutation will ensure that some attribute becomes specified.)
            #-------------------------------------------------------
            if len(self.specifiedAttList) > cons.env.formatData.specLimit:
                self.specLimitFix(self)

            if len(cl.specifiedAttList) > cons.env.formatData.specLimit:
                self.specLimitFix(cl)

            tempList1 = copy.deepcopy(p_self_specifiedAttList)
            tempList2 = copy.deepcopy(cl.specifiedAttList)
            tempList1.sort()
            tempList2.sort()
            if changed and (tempList1 == tempList2):
                changed = False

            v = 0
            while v < len(cl.specifiedAttList)-1:
                attributeInfo = cons.env.formatData.attributeInfo[cl.specifiedAttList[v]]
                if attributeInfo[0]:
                    if cl.condition[v][0] > cl.condition[v][1]:
                        print('crossover error: cl')
                        print(cl.condition)
                        temp = cl.condition[v][0]
                        cl.condition[v][0] = cl.condition[v][1]
                        cl.condition[v][1] = temp
                        #cl.origin += '_fix' #temporary code

                    if state[cl.specifiedAttList[v]] == cons.labelMissingData: #If example is missing, don't attempt range, instead generalize attribute in rule.
                        #print 'removed '+str(cl.specifiedAttList[v])
                        cl.specifiedAttList.pop(v)
                        cl.condition.pop(v) #buildMatch handles both discrete and continuous attributes
                        v -=1
                    else:
                        if not cl.condition[v][0] < state[cl.specifiedAttList[v]] or not cl.condition[v][1] > state[cl.specifiedAttList[v]]:
                            #print 'crossover range error'
                            attRange = attributeInfo[1][1] - attributeInfo[1][0]
                            rangeRadius = random.randint(25,75)*0.01*attRange / 2.0 #Continuous initialization domain radius.
                            Low = state[cl.specifiedAttList[v]] - rangeRadius
                            High = state[cl.specifiedAttList[v]] + rangeRadius
                            condList = [Low,High] #ALKR Representation, Initialization centered around training instance  with a range between 25 and 75% of the domain size.
                            cl.condition[v] = condList
                            #self.origin += '_RangeError' #temporary code
                v += 1

            v = 0
            while v < len(self.specifiedAttList)-1:
                attributeInfo = cons.env.formatData.attributeInfo[self.specifiedAttList[v]]
                if attributeInfo[0]:
                    if self.condition[v][0] > self.condition[v][1]:
                        print('crossover error: self')
                        print(self.condition)
                        temp = self.condition[v][0]
                        self.condition[v][0] = self.condition[v][1]
                        self.condition[v][1] = temp
                        #self.origin += '_fix' #temporary code

                    if state[self.specifiedAttList[v]] == cons.labelMissingData: #If example is missing, don't attempt range, instead generalize attribute in rule.
                        #print 'removed '+str(self.specifiedAttList[v])
                        self.specifiedAttList.pop(v)
                        self.condition.pop(v) #buildMatch handles both discrete and continuous attributes
                        v -=1
                    else:
                        if not self.condition[v][0] < state[self.specifiedAttList[v]] or not self.condition[v][1] > state[self.specifiedAttList[v]]:
                            #print 'crossover range error'
                            attRange = attributeInfo[1][1] - attributeInfo[1][0]
                            rangeRadius = random.randint(25,75)*0.01*attRange / 2.0 #Continuous initialization domain radius.
                            Low = state[self.specifiedAttList[v]] - rangeRadius
                            High = state[self.specifiedAttList[v]] + rangeRadius
                            condList = [Low,High] #ALKR Representation, Initialization centered around training instance  with a range between 25 and 75% of the domain size.
                            self.condition[v] = condList
                            #self.origin += '_RangeError' #temporary code
                v += 1

            return changed
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE CROSSOVER
        #-------------------------------------------------------
        else: #ContinuousCode #########################
            #self.origin += '_CrossoverP' #temporary code
            #cl.origin += '_CrossoverP' #temporary code
            return self.phenotypeCrossover(cl, phenotype)


    #ContinuousCode #########################
    def phenotypeCrossover(self, cl, phenotype):
        """ Crossover a continuous phenotype """
        changed = False
        if self.phenotype[0] == cl.phenotype[0] and self.phenotype[1] == cl.phenotype[1]:
            return changed
        else:
            tempKey = random.random() < 0.5 #Make random choice between 4 scenarios, Swap minimums, Swap maximums, Children preserve parent phenotypes.
            if tempKey: #Swap minimum
                temp = self.phenotype[0]
                self.phenotype[0] = cl.phenotype[0]
                cl.phenotype[0] = temp
                changed = True
            elif tempKey:  #Swap maximum
                temp = self.phenotype[1]
                self.phenotype[1] = cl.phenotype[1]
                cl.phenotype[1] = temp
                changed = True
            if not self.phenotype[0] < phenotype or not self.phenotype[1] > phenotype:
                print('phenotype crossover range error')
            if not cl.phenotype[0] < phenotype or not cl.phenotype[1] > phenotype:
                print('phenotype crossover range error')
        return changed


    def specLimitFix(self,cl):
        """ Lowers classifier specificity to specificity limit. """
        if cons.doAttributeFeedback:
            # Identify 'toRemove' attributes with lowest AT scores
            while len(cl.specifiedAttList) > cons.env.formatData.specLimit:
                minVal = cons.AT.getTrackProb()[cl.specifiedAttList[0]]
                minAtt = cl.specifiedAttList[0]
                for j in cl.specifiedAttList:
                    if cons.AT.getTrackProb()[j] < minVal:
                        minVal = cons.AT.getTrackProb()[j]
                        minAtt = j
                i = cl.specifiedAttList.index(minAtt) #reference to the position of the attribute in the rule representation
                cl.specifiedAttList.remove(minAtt)
                cl.condition.pop(i) #buildMatch handles both discrete and continuous attributes

        else:
            #Randomly pick 'toRemove'attributes to be generalized
            toRemove = len(cl.specifiedAttList) - cons.env.formatData.specLimit
            try:
                genTarget = random.sample(cl.specifiedAttList,toRemove)
            except:
                print("Specified Att List: " + str(cl.specifiedAttList))
                print("SpecLimit: " + str(cons.env.formatData.specLimit))
                print("toRemove: " + str(toRemove))
                raise NameError("Spec Limit Fix error")
            for j in genTarget:
                i = cl.specifiedAttList.index(j) #reference to the position of the attribute in the rule representation
                cl.specifiedAttList.remove(j)
                cl.condition.pop(i) #buildMatch handles both discrete and continuous attributes


    def Mutation(self, state, phenotype):
        """ Mutates the condition of the classifier. Also handles phenotype mutation. This is a niche mutation, which means that the resulting classifier will still match the current instance.  """
        pressureProb = 0.5 #Probability that if EK is activated, it will be applied.
        useAT = False
        if cons.doAttributeFeedback and random.random() < cons.AT.percent:
            useAT = True
        changed = False;
        #-------------------------------------------------------
        # MUTATE CONDITION - mutation rate (upsilon) used to probabilistically determine the number of attributes that will be mutated in the classifier.
        #-------------------------------------------------------

        steps = 0
        keepGoing = True
        while keepGoing:
            if random.random() < cons.upsilon:
                steps += 1
            else:
                keepGoing = False

        #Define Spec Limits
        if (len(self.specifiedAttList) - steps) <= 1:
            lowLim = 1
        else:
            lowLim = len(self.specifiedAttList) - steps
        if (len(self.specifiedAttList) + steps) >= cons.env.formatData.specLimit:
            highLim = cons.env.formatData.specLimit
        else:
            highLim = len(self.specifiedAttList) + steps
        if len(self.specifiedAttList) == 0:
            highLim = 1

        #Get new rule specificity.
        try:
            newRuleSpec = random.randint(lowLim,highLim)
        except Exception as exc:
            print(self.specifiedAttList)
            print(self.condition)
            print(exc)
            print("Steps: " + str(steps))
            print("Low: " + str(lowLim))
            print("High: " + str(highLim))
            print("SpecLimit" + str(cons.env.formatData.specLimit))
            raise NameError('Problem with specificity')

        #-------------------------------------------------------
        # MAINTAIN SPECIFICITY -
        #-------------------------------------------------------
        if newRuleSpec == len(self.specifiedAttList) and random.random() < (1-cons.upsilon): #Pick one attribute to generalize and another to specify.  Keep overall rule specificity the same.
            #Identify Generalizing Target
            if not cons.useExpertKnowledge or random.random() > pressureProb:
                genTarget = random.sample(self.specifiedAttList,1)
            else:
                genTarget = self.selectGeneralizeRW(1)

            attributeInfo = cons.env.formatData.attributeInfo[genTarget[0]]
            if not attributeInfo[0] or random.random() > 0.5: #GEN/SPEC OPTION
                if not useAT or random.random() > cons.AT.getTrackProb()[genTarget[0]]:
                    #Generalize Target
                    i = self.specifiedAttList.index(genTarget[0]) #reference to the position of the attribute in the rule representation
                    self.specifiedAttList.remove(genTarget[0])
                    self.condition.pop(i) #buildMatch handles both discrete and continuous attributes
                    changed = True
            else:
                self.mutateContinuousAttributes(useAT,genTarget[0])

            #Identify Specifying Target
            if len(self.specifiedAttList) >= len(state): #Catch for small datasets - if all attributes already specified at this point.
                pass
            else:
                if not cons.useExpertKnowledge or random.random() > pressureProb:
                    pickList = list(range(cons.env.formatData.numAttributes))
                    for i in self.specifiedAttList: # Make list with all non-specified attributes
                        pickList.remove(i)

                    specTarget = random.sample(pickList,1)
                else:
                    specTarget = self.selectSpecifyRW(1)
                if state[specTarget[0]] != cons.labelMissingData and (not useAT or random.random() < cons.AT.getTrackProb()[specTarget[0]]):
                    #Specify Target
                    self.specifiedAttList.append(specTarget[0])
                    self.condition.append(self.buildMatch(specTarget[0], state)) #buildMatch handles both discrete and continuous attributes
                    changed = True

                if len(self.specifiedAttList) > cons.env.formatData.specLimit:    #Double Check
                    self.specLimitFix(self)
        #-------------------------------------------------------
        # INCREASE SPECIFICITY
        #-------------------------------------------------------
        elif newRuleSpec > len(self.specifiedAttList): #Specify more attributes
            change = newRuleSpec - len(self.specifiedAttList)
            if not cons.useExpertKnowledge or random.random() > pressureProb:
                pickList = list(range(cons.env.formatData.numAttributes))
                for i in self.specifiedAttList: # Make list with all non-specified attributes
                    pickList.remove(i)
                specTarget = random.sample(pickList,change)
            else:
                specTarget = self.selectSpecifyRW(change)
            for j in specTarget:
                if state[j] != cons.labelMissingData and (not useAT or random.random() < cons.AT.getTrackProb()[j]):
                    #Specify Target
                    self.specifiedAttList.append(j)
                    self.condition.append(self.buildMatch(j, state)) #buildMatch handles both discrete and continuous attributes
                    changed = True

        #-------------------------------------------------------
        # DECREASE SPECIFICITY
        #-------------------------------------------------------
        elif newRuleSpec < len(self.specifiedAttList): # Generalize more attributes.
            change = len(self.specifiedAttList) - newRuleSpec
            if not cons.useExpertKnowledge or random.random() > pressureProb:
                genTarget = random.sample(self.specifiedAttList,change)
            else:
                genTarget = self.selectGeneralizeRW(change)

            #-------------------------------------------------------
            # DISCRETE OR CONTINUOUS ATTRIBUTE - remove attribute specification with 50% chance if we have continuous attribute, or 100% if discrete attribute.
            #-------------------------------------------------------
            for j in genTarget:
                attributeInfo = cons.env.formatData.attributeInfo[j]
                if not attributeInfo[0] or random.random() > 0.5: #GEN/SPEC OPTION
                    if not useAT or random.random() > cons.AT.getTrackProb()[j]:
                        i = self.specifiedAttList.index(j) #reference to the position of the attribute in the rule representation
                        self.specifiedAttList.remove(j)
                        self.condition.pop(i) #buildMatch handles both discrete and continuous attributes
                        changed = True
                else:
                    self.mutateContinuousAttributes(useAT,j)
        else:#Neither specify or generalize.
            pass

        #Earlier version added a missing value check as well as an inclusive range check here to avoid possible mutation problems. (see below)
        if changed:
            #self.origin += '_MutationC' #temporary code
            v = 0
            while v < len(self.specifiedAttList)-1:
                attributeInfo = cons.env.formatData.attributeInfo[self.specifiedAttList[v]]
                if attributeInfo[0]:
                    if state[self.specifiedAttList[v]] == cons.labelMissingData: #If example is missing, don't attempt range, instead generalize attribute in rule.
                        self.specifiedAttList.pop(v)
                        self.condition.pop(v) #buildMatch handles both discrete and continuous attributes
                        v -=1
                    else:
                        if not self.condition[v][0] < state[self.specifiedAttList[v]] or not self.condition[v][1] > state[v]:
                            #print 'mutation range error'
                            attRange = attributeInfo[1][1] - attributeInfo[1][0]
                            rangeRadius = random.randint(25,75)*0.01*attRange / 2.0 #Continuous initialization domain radius.
                            #print state[attRef]
                            Low = state[self.specifiedAttList[v]] - rangeRadius
                            High = state[self.specifiedAttList[v]] + rangeRadius
                            condList = [Low,High] #ALKR Representation, Initialization centered around training instance  with a range between 25 and 75% of the domain size.
                            self.condition[v] = condList
                            #self.origin += '_RangeError' #temporary code
                v += 1
        #-------------------------------------------------------
        # MUTATE PHENOTYPE
        #-------------------------------------------------------
        if cons.env.formatData.discretePhenotype:
            pass
        else: #ContinuousCode #########################
            nowChanged = self.continuousPhenotypeMutation(phenotype) #NOTE: Must mutate to still include true current value.
            #if nowChanged:
                #self.origin += '_MutationP' #temporary code
        if changed:# or nowChanged:
            return True


    def continuousPhenotypeMutation(self, phenotype):
        """ Mutate this rule's continuous phenotype. """
        #Continuous Phenotype Crossover------------------------------------
        changed = False
        if random.random() < cons.upsilon: #Mutate continuous phenotype
            phenRange = self.phenotype[1] - self.phenotype[0]
            mutateRange = random.random()*0.5*phenRange
            tempKey = random.randint(0,2) #Make random choice between 3 scenarios, mutate minimums, mutate maximums, mutate both
            if tempKey == 0: #Mutate minimum
                if random.random() > 0.5 or self.phenotype[0] + mutateRange <= phenotype: #Checks that mutated range still contains current phenotype
                    self.phenotype[0] += mutateRange
                else: #Subtract
                    self.phenotype[0] -= mutateRange
                changed = True
            elif tempKey == 1: #Mutate maximum
                if random.random() > 0.5 or self.phenotype[1] - mutateRange >= phenotype: #Checks that mutated range still contains current phenotype
                    self.phenotype[1] -= mutateRange
                else: #Subtract
                    self.phenotype[1] += mutateRange
                changed = True
            else: #mutate both
                if random.random() > 0.5 or self.phenotype[0] + mutateRange <= phenotype: #Checks that mutated range still contains current phenotype
                    self.phenotype[0] += mutateRange
                else: #Subtract
                    self.phenotype[0] -= mutateRange
                if random.random() > 0.5 or self.phenotype[1] - mutateRange >= phenotype: #Checks that mutated range still contains current phenotype
                    self.phenotype[1] -= mutateRange
                else: #Subtract
                    self.phenotype[1] += mutateRange
                changed = True

            #Repair range - such that min specified first, and max second.
            self.phenotype.sort()
        #---------------------------------------------------------------------
        return changed


    def selectGeneralizeRW(self, count):
        """ EK applied to the selection of an attribute to generalize for mutation. """
        EKScoreSum = 0
        selectList = []
        currentCount = 0
        specAttList = copy.deepcopy(self.specifiedAttList)
        for i in self.specifiedAttList:
            #When generalizing, EK is inversely proportional to selection probability
            EKScoreSum += 1 / float(cons.EK.scores[i]+1)

        while currentCount < count:
            choicePoint = random.random() * EKScoreSum
            i=0
            sumScore = 1 / float(cons.EK.scores[specAttList[i]]+1)
            while choicePoint > sumScore:
                i=i+1
                sumScore += 1 / float(cons.EK.scores[specAttList[i]]+1)
            selectList.append(specAttList[i])
            EKScoreSum -= 1 / float(cons.EK.scores[specAttList[i]]+1)
            specAttList.pop(i)
            currentCount += 1
        return selectList


    def selectSpecifyRW(self, count):
        """ EK applied to the selection of an attribute to specify for mutation. """
        pickList = list(range(cons.env.formatData.numAttributes))
        for i in self.specifiedAttList: # Make list with all non-specified attributes
            try:
                pickList.remove(i)
            except:
                print(self.specifiedAttList)
                print(pickList)
                raise NameError("Problem with pick list")

        EKScoreSum = 0
        selectList = []
        currentCount = 0

        for i in pickList:
            #When generalizing, EK is inversely proportional to selection probability
            EKScoreSum += cons.EK.scores[i]

        while currentCount < count:
            choicePoint = random.random() * EKScoreSum
            i=0
            sumScore = cons.EK.scores[pickList[i]]
            while choicePoint > sumScore:
                i=i+1
                sumScore += cons.EK.scores[pickList[i]]
            selectList.append(pickList[i])
            EKScoreSum -= cons.EK.scores[pickList[i]]
            pickList.pop(i)
            currentCount += 1
        return selectList


    def mutateContinuousAttributes(self, useAT, j):
        #-------------------------------------------------------
        # MUTATE CONTINUOUS ATTRIBUTES
        #-------------------------------------------------------
        if useAT:
            if random.random() < cons.AT.getTrackProb()[j]: #High AT probability leads to higher chance of mutation (Dives ExSTraCS to explore new continuous ranges for important attributes)
                #Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
                attRange = float(cons.env.formatData.attributeInfo[j][1][1]) - float(cons.env.formatData.attributeInfo[j][1][0])
                i = self.specifiedAttList.index(j) #reference to the position of the attribute in the rule representation
                mutateRange = random.random()*0.5*attRange
                if random.random() > 0.5: #Mutate minimum
                    if random.random() > 0.5: #Add
                        self.condition[i][0] += mutateRange
                    else: #Subtract
                        self.condition[i][0] -= mutateRange
                else: #Mutate maximum
                    if random.random() > 0.5: #Add
                        self.condition[i][1] += mutateRange
                    else: #Subtract
                        self.condition[i][1] -= mutateRange
                #Repair range - such that min specified first, and max second.
                self.condition[i].sort()
                changed = True
        elif random.random() > 0.5:
                #Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
                attRange = float(cons.env.formatData.attributeInfo[j][1][1]) - float(cons.env.formatData.attributeInfo[j][1][0])
                i = self.specifiedAttList.index(j) #reference to the position of the attribute in the rule representation
                mutateRange = random.random()*0.5*attRange
                if random.random() > 0.5: #Mutate minimum
                    if random.random() > 0.5: #Add
                        self.condition[i][0] += mutateRange
                    else: #Subtract
                        self.condition[i][0] -= mutateRange
                else: #Mutate maximum
                    if random.random() > 0.5: #Add
                        self.condition[i][1] += mutateRange
                    else: #Subtract
                        self.condition[i][1] -= mutateRange
                #Repair range - such that min specified first, and max second.
                self.condition[i].sort()
                changed = True
        else:
            pass


    def rangeCheck(self):
        """ Checks and prevents the scenario where a continuous attributes specified in a rule has a range that fully encloses the training set range for that attribute."""
        for attRef in self.specifiedAttList:
            if cons.env.formatData.attributeInfo[attRef][0]: #Attribute is Continuous
                trueMin = cons.env.formatData.attributeInfo[attRef][1][0]
                trueMax = cons.env.formatData.attributeInfo[attRef][1][1]
                i = self.specifiedAttList.index(attRef)
                valBuffer = (trueMax-trueMin)*0.1
                if self.condition[i][0] <= trueMin and self.condition[i][1] >= trueMax: # Rule range encloses entire training range
                    self.specifiedAttList.remove(attRef)
                    self.condition.pop(i)
                    return
                elif self.condition[i][0]+valBuffer < trueMin:
                    self.condition[i][0] = trueMin - valBuffer
                elif self.condition[i][1]- valBuffer > trueMax:
                    self.condition[i][1] = trueMin + valBuffer
                else:
                    pass

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SUBSUMPTION METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def subsumes(self, cl):
        """ Returns if the classifier (self) subsumes cl """
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        if cl.isTree:
            return False
        if cons.env.formatData.discretePhenotype:
            if cl.phenotype == self.phenotype:
                if self.isSubsumer() and self.isMoreGeneral(cl):
                    return True
            return False
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE -  NOTE: for continuous phenotypes, the subsumption intuition is reversed, i.e. While a subsuming rule condition is more general, a subsuming phenotype is more specific.
        #-------------------------------------------------------
        else: #ContinuousCode #########################
            if self.phenotype[0] >= cl.phenotype[0] and self.phenotype[1] <= cl.phenotype[1]:
                if self.isSubsumer() and self.isMoreGeneral(cl):
                    return True
            return False


    def isSubsumer(self):
        """ Returns if the classifier (self) is a possible subsumer. A classifier must have sufficient experience (one epoch) and it must also be as or more accurate than the classifier it is trying to subsume.  """
        if self.matchCount > cons.theta_sub and self.accuracy > cons.acc_sub: #self.getAccuracy() > 0.99:
            return True
        return False


    def isMoreGeneral(self,cl):
        """ Returns if the classifier (self) is more general than cl. Check that all attributes specified in self are also specified in cl. """
        if cl.isTree:
            return False
        if len(self.specifiedAttList) >= len(cl.specifiedAttList):
            return False

        for i in range(len(self.specifiedAttList)): #Check each attribute specified in self.condition
            attributeInfo = cons.env.formatData.attributeInfo[self.specifiedAttList[i]]
            if self.specifiedAttList[i] not in cl.specifiedAttList:
                return False
            #-------------------------------------------------------
            # CONTINUOUS ATTRIBUTE
            #-------------------------------------------------------
            if attributeInfo[0]:
                otherRef = cl.specifiedAttList.index(self.specifiedAttList[i])
                #If self has a narrower ranger of values than it is a subsumer
                if self.condition[i][0] < cl.condition[otherRef][0]:
                    return False
                if self.condition[i][1] > cl.condition[otherRef][1]:
                    return False

        return True


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # DELETION METHOD
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def getDelProp(self, meanFitness):
        """  Returns the vote for deletion of the classifier. """
        if self.fitness/self.numerosity >= cons.delta*meanFitness or self.matchCount < cons.theta_del:
            self.deletionVote = self.aveMatchSetSize*self.numerosity

        elif self.fitness == 0.0 or self.fitness/self.numerosity == 0.0:
            self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (cons.init_fit/self.numerosity)
        else:
            #print self.fitness
            self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (self.fitness/self.numerosity) #note, numerosity seems redundant (look into theory of deletion in LCS.
        return self.deletionVote


    #NEW############################################################################################################################################################
#     def getDeletionVote(self):
#         """  Returns the vote for deletion of the classifier. """
#         #Preserve Quick delete
#
#         #self.phenotype_RP = cons.env.formatData.classProportions[self.phenotype]
#         if self.epochComplete:# or self.fitness == 0:
#             if (self.accuracy <= self.phenotype_RP) or (self.matchCover < 2 and len(self.specifiedAttList) > 1):# or self.fitness == 0:
#
#                 return [0,True]
#
#         #self.deletionVote = self.aveMatchSetSize * self.aveMatchSetSize *self.numerosity / self.fitness
#         self.deletionVote = self.aveMatchSetSize * self.numerosity / self.fitness
#         if self.numerosity > 2:
#             self.deletionVote *= self.numerosity
#         return [self.deletionVote,False]


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER METHODS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def buildMatch(self, attRef, state):
        """ Builds a matching condition element given an attribute to be specified for the classifierCovering method. """
        attributeInfo = cons.env.formatData.attributeInfo[attRef]
        #-------------------------------------------------------
        # CONTINUOUS ATTRIBUTE
        #-------------------------------------------------------
        if attributeInfo[0]:
            attRange = attributeInfo[1][1] - attributeInfo[1][0]
            rangeRadius = random.randint(25,75)*0.01*attRange / 2.0 #Continuous initialization domain radius.
            Low = state[attRef] - rangeRadius
            High = state[attRef] + rangeRadius
            condList = [Low,High] #ALKR Representation, Initialization centered around training instance  with a range between 25 and 75% of the domain size.
        #-------------------------------------------------------
        # DISCRETE ATTRIBUTE
        #-------------------------------------------------------
        else:
            condList = state[attRef] #State already formatted like GABIL in DataManagement

        return condList


    def equals(self, cl):
        """ Returns if the two classifiers are identical in condition and phenotype. This works for discrete or continuous attributes or phenotypes. """
        if cl.isTree:
            return False
        else:
            if cl.phenotype == self.phenotype and len(cl.specifiedAttList) == len(self.specifiedAttList): #Is phenotype the same and are the same number of attributes specified - quick equality check first.
                clRefs = sorted(cl.specifiedAttList)
                selfRefs = sorted(self.specifiedAttList)
                if clRefs == selfRefs:
                    for i in range(len(cl.specifiedAttList)):
                        tempIndex = self.specifiedAttList.index(cl.specifiedAttList[i])
                        if cl.condition[i] == self.condition[tempIndex]:
                            pass
                        else:
                            return False
                    return True
            return False


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PARAMETER UPDATES
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def updateEpochStatus(self, exploreIter):
        """ Determines when a learning epoch has completed (one cycle through training data). """
        #if not self.epochComplete and (exploreIter - self.initTimeStamp-1) >= cons.env.formatData.numTrainInstances and cons.offlineData:
        if not self.epochComplete and (exploreIter - self.initTimeStamp) >= cons.env.formatData.numTrainInstances and cons.offlineData:
            self.epochComplete = True
            cons.firstEpochComplete = True
            #randomProbClass = cons.env.formatData.classProportions[self.phenotype] #Put this in as a fix - for rules that become epoch complete after having been extrapolated on a previous run.
            #self.usefulDiff = self.correctCover - randomProbClass*self.matchCover
            self.usefulDiff = (self.correctCover - self.phenotype_RP*self.matchCover)#*len(self.specifiedAttList)
            #Pareto Fitness - Epoch Complete Front Construction

            if self.accuracyComponent > 0 and self.usefulDiff > 0:
                objectivePair = [self.accuracyComponent,self.usefulDiff]
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

        if self.epochComplete: #Once epoch Completed, number of matches for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.matchCover += 1


    def updateCorrect(self):
        """ Increases the correct phenotype tracking by one. Once an epoch has completed, rule accuracy can't change."""
        self.correctCount += 1
        if self.epochComplete: #Once epoch Completed, number of correct for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.correctCover += 1

    def updateError(self,trueEndpoint):
        if not self.epochComplete:
            high = self.phenotype[1]
            low = self.phenotype[0]
            #Error caclulations are limited to extremes of observed training data phenotypes in calculating the range centroid.
            if self.phenotype[1] > cons.env.formatData.phenotypeList[1]:
                high = cons.env.formatData.phenotypeList[1]
            if self.phenotype[0] < cons.env.formatData.phenotypeList[0]:
                low = cons.env.formatData.phenotypeList[0]

            rangeCentroid = (high + low) / 2.0
            error = abs(rangeCentroid - trueEndpoint)
            adjustedError = error / (cons.env.formatData.phenotypeList[1] - cons.env.formatData.phenotypeList[0])

            self.errorSum += adjustedError  #Error is fraction of total phenotype range (i.e. maximum possible error)
            self.errorCount += 1


    def updateIncorrectError(self):
        if not self.epochComplete:
            self.errorSum += 1.0
            self.errorCount += 1

    #NEW
    def updateAccuracy(self,exploreIter):
        """ Update the accuracy tracker """
        nonUsefulDiscount = 0.001
        coverOpportunity = 1000
        adjAccuracy = 0
        #-----------------------------------------------------------------------------------
        # CALCULATE ACCURACY
        #-----------------------------------------------------------------------------------
        try:
            if cons.env.formatData.discretePhenotype:
                self.accuracy = self.correctCover / float(self.matchCover)
            else: #ContinuousCode #########################
                self.accuracy = 1 - (self.errorSum/self.matchCover) # 1- average error based on range centroid.  Should be natural pressure to achieve narrow endpoint range.
        except:
            print("CorrectCover: " + str(self.correctCover))
            print("MatchCover: " + str(self.matchCover))
            print("MatchCount: " + str(self.matchCount))
            print("InitTime: " + str(self.initTimeStamp))
            print("EpochComplete: " + str(self.epochComplete))
            raise NameError("Problem with updating accuracy")


        #-----------------------------------------------------------------------------------
        # CALCULATE ADJUSTED ACCURACY
        #-----------------------------------------------------------------------------------
        if self.accuracy > self.phenotype_RP:
            adjAccuracy = self.accuracy - self.phenotype_RP
        elif self.matchCover == 2 and self.correctCover == 1 and not self.epochComplete and (exploreIter - self.timeStampGA) < coverOpportunity:
            adjAccuracy = self.phenotype_RP / 2.0
        else:
            adjAccuracy = self.accuracy * nonUsefulDiscount
        #-----------------------------------------------------------------------------------
        # CALCULATE ACCURACY COMPONENT
        #-----------------------------------------------------------------------------------
        maxAccuracy = 1-self.phenotype_RP
        if maxAccuracy == 0:
            self.accuracyComponent = 0
        else:
            self.accuracyComponent = adjAccuracy / float(maxAccuracy) #Accuracy contribution scaled between 0 and 1 allowing for different maximum accuracies
        self.accuracyComponent = 2*((1/float(1+math.exp(-5*self.accuracyComponent)))-0.5)/float(0.98661429815)
        self.accuracyComponent = math.pow(self.accuracyComponent,1)


    def updateCorrectCoverage(self):
        """ """
        self.coverDiff = self.correctCover - self.phenotype_RP*self.matchCover
#         print self.coverDiff
#         expectedCoverage = cons.env.formatData.numTrainInstances*self.totalFreq
#         self.coverDiff = self.correctCover - expectedCoverage

    #NEW
    def updateIndFitness(self,exploreIter):
        """ Calculates the fitness of an individual rule based on it's accuracy and correct coverage relative to the 'Pareto' front """
        coverOpportunity = 1000
        if self.coverDiff > 0:
#             print 'quality'
            #-----------------------------------------------------------------------------------
            # CALCULATE CORRECT COVER DIFFERENCE COMPONENT
            #-----------------------------------------------------------------------------------
            #NOTES: Coverage is directly comparable when epoch complete, otherwise we want to estimate what coverage might be farther out.
            if self.epochComplete:
                #Get Pareto Metric
                self.indFitness = cons.env.formatData.ecFront.getParetoFitness([self.accuracyComponent,self.coverDiff])

            else: #Rule Not epoch complete
                #EXTRAPOLATE coverDiff up to number of trainin instances (i.e. age at epoch complete)
                ruleAge = exploreIter - self.initTimeStamp+1 #Correct, because we include the current instance we are on.
                self.coverDiff = self.coverDiff*cons.env.formatData.numTrainInstances/float(ruleAge)
                objectivePair = [self.accuracyComponent,self.coverDiff]
                #BEFORE PARETO FRONTS BEGIN TO BE UPDATED
                if len(cons.env.formatData.necFront.paretoFrontAcc) == 0: #Nothing stored yet on incomplete epoch front
                    #Temporarily use accuracy as fitness in this very early stage.
                    #print 'fit path 1'
                    self.indFitness = self.accuracyComponent
                    if ruleAge >= coverOpportunity: #attempt to update front
                        cons.env.formatData.necFront.updateFront(objectivePair)

                #PARETO FRONTS ONLINE
                else:  #Some pareto front established.
                    if len(cons.env.formatData.ecFront.paretoFrontAcc) > 0: #Leave epoch incomplete front behind.
                        self.indFitness = cons.env.formatData.ecFront.getParetoFitness(objectivePair)
                        #print 'fit path 2'
                    else: #Keep updating and evaluating with epoch incomplete front.
                        if ruleAge < coverOpportunity: #Very young rules can not affect bestCoverDiff
                            self.preFitness = cons.env.formatData.necFront.getParetoFitness(objectivePair)
                            #print 'fit path 3'
                        else:
                            cons.env.formatData.necFront.updateFront(objectivePair)
                            self.indFitness = cons.env.formatData.necFront.getParetoFitness(objectivePair)
                            self.matchedAndFrontEstablished = True
                            #print 'fit path 4'
        else:
#             print 'poor'
#             print self.accuracyComponent
            self.indFitness = self.accuracyComponent / float(1000)

        if self.indFitness < 0:
            print("negative fitness error")
        if round(self.indFitness,5) > 1: #rounding added to handle odd division error, where 1.0 was being treated as a very small decimal just above 1.0
            print("big fitness error")

        #self.indFitness = math.pow(self.indFitness,cons.nu)  #Removed 11/25/15 - seems redundant with accuracy version (use one or the other)
        self.lastIndFitness = copy.deepcopy(self.indFitness)



    #NEW
    def updateRelativeIndFitness(self, indFitSum, partOfCorrect, exploreIter):
        """  Updates the relative individual fitness calculation """
        self.sumIndFitness = indFitSum
        self.partOfCorrect = partOfCorrect

        #self.lastRelativeIndFitness = copy.deepcopy(self.relativeIndFitness)   #Is this needed????

        if partOfCorrect:
            self.relativeIndFitness = self.indFitness*self.numerosity /float(self.sumIndFitness)
            #self.relativeIndFitness = self.indFitness/float(self.sumIndFitness) #Treat epoch complete or incomplete equally here.  This will give young rules a boost (in this method, relative fitness can be larger than 1 for NEC rules.
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


    #NEW
    def updateFitness(self, exploreIter):
        """ Update the fitness parameter. """
        if self.epochComplete:
            percRuleExp = 1.0
        else:
            percRuleExp = (exploreIter-self.initTimeStamp+1)/float(cons.env.formatData.numTrainInstances)
        #Consider the useful accuracy cutoff -  -maybe use a less dramatic change or ...na...
        beta = 0.2
        if self.matchCount >= 1.0/beta:
            #print 'fit A'
#                 print self.fitness
#                 print self.relativePreFitness
            self.fitness = self.fitness + beta*percRuleExp*(self.relativeIndFitness-self.fitness)
        elif self.matchCount == 1 or self.aveRelativeIndFitness == None:  #second condition handles special case after GA rule generated, but not has not gone through full matching yet
            #print 'fit B'
            self.fitness = self.relativeIndFitness
            self.aveRelativeIndFitness = self.relativeIndFitness
#                 if self.initTimeStamp == 2:
#                     print self.aveRelativePreFitness
#                     print 5/0
        else:
            #print 'fit C'
            self.fitness = (self.aveRelativeIndFitness*(self.matchCount-1)+self.relativeIndFitness)/self.matchCount  #often, last releative prefitness is 0!!!!!!!!!!!!!!!!!!!!!!!
            self.aveRelativeIndFitness = (self.aveRelativeIndFitness*(self.matchCount-1)+self.relativeIndFitness)/self.matchCount

#         if cons.env.formatData.discretePhenotype:
#             #Consider the useful accuracy cutoff -  -maybe use a less dramatic change or ...na...
#             beta = 0.2
#             if self.matchCount >= 1.0/beta:
#                 #print 'fit A'
# #                 print self.fitness
# #                 print self.relativePreFitness
#                 self.fitness = self.fitness + beta*(self.relativeIndFitness-self.fitness)
#             elif self.matchCount == 1 or self.aveRelativeIndFitness == None:  #second condition handles special case after GA rule generated, but not has not gone through full matching yet
#                 #print 'fit B'
#                 self.fitness = self.relativeIndFitness
#                 self.aveRelativeIndFitness = self.relativeIndFitness
# #                 if self.initTimeStamp == 2:
# #                     print self.aveRelativePreFitness
# #                     print 5/0
#             else:
#                 #print 'fit C'
#                 self.fitness = (self.aveRelativeIndFitness*(self.matchCount-1)+self.relativeIndFitness)/self.matchCount  #often, last releative prefitness is 0!!!!!!!!!!!!!!!!!!!!!!!
#                 self.aveRelativeIndFitness = (self.aveRelativeIndFitness*(self.matchCount-1)+self.relativeIndFitness)/self.matchCount
#             #-----------------------------------------------------------------------------------
#             # NO FITNESS SHARING!!!!!!!!!!!!!!!
#             #-----------------------------------------------------------------------------------
        self.fitness = self.indFitness #TEMPORARY #Effectively turns off fitness sharing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#         else: #ContinuousCode ########################
#             #UPDATE NEEDED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#             if (self.phenotype[1]-self.phenotype[0]) >= cons.env.formatData.phenotypeRange:
#                 self.fitness = pow(0.0001, 5)
#             else:
#                 if self.matchCover < 2 and self.epochComplete:
#                     self.fitness = pow(0.0001, 5)
#                 else:
#                     self.fitness = pow(self.accuracy, cons.nu) #- (self.phenotype[1]-self.phenotype[0])/cons.env.formatData.phenotypeRange)

        self.lastMatchFitness = copy.deepcopy(self.fitness)
        if self.fitness < 0 or round(self.fitness,4) > 1:
            self.reportClassifier('update fitness')
            x = 5/0

        #RULE FITNESS TESTING CODE------------------------------------------------------------------------------------------------------
#         ruleTimeID = 3279
#         if self.initTimeStamp == ruleTimeID:
#             print '-----------------'
#             print str(self.condition) + str(self.specifiedAttList)
#             print 'sumfitness '+str(self.sumIndFitness)
#             print 'correct? '+str(self.partOfCorrect)
#             print 'relIndFit ' +str(self.relativeIndFitness)
#             print 'fitness ' +str(self.fitness)

        #self.isMatchSetFitness = True

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
        
        
    def reportClassifier(self,task):
        print(task)
        #print 'exploreIter= '+str(exploreIter)
        #print 'fitTYPE= ' +str(self.isMatchSetFitness)
        print('fit= '+str(self.fitness))
        print('relativeIndFit= '+str(self.relativeIndFitness))
        print('indFitness= '+str(self.indFitness))
        print('cDiff= '+str(self.coverDiff))
        print('self.correctCover=' +str(self.correctCover))
        print('self.matchCover= '+str(self.matchCover))
        #print 'specAttList=' +str(self.specifiedAttList)
        print('self.accuracyComponent= '+ str(self.accuracyComponent))
        #ruleAge = exploreIter - self.initTimeStamp+1
        print('InitTimestamp+1= ' + str(self.initTimeStamp+1))
        #print 'ruleAge= '+ str(ruleAge)
        print('self.self.aveMatchSetSize= '+str(self.aveMatchSetSize))
        print('epochComplete= '+str(self.epochComplete))
        print('numerosity= '+str(self.numerosity))

        print('-----------------------------------------------')


    #NEW
    def briefUpdateFitness(self, exploreIter):
        #print 'briefupdateFit'
        #Activated durring matchign for all epoch complete rules -
        #Recalculate ENC rule fitness based on progression of algorithm (i.e. recalculate Coverage extrapolation and recheck pareto front - don't need to check for nondominated points
        #this is because points can only have lower extrapolation inbetween being in a match set.  This will effect rules that deal with rare states the most.

        #ALSO we need to adapt exstracs to really large datasets, where epoch complete is rare. ( consider prediction (weight votes by experience),
        #also consider that INcomplete pareto front might be most important (maybe not shift entirely over to EC front

        #Also revisit compaction - for big data - don't want to automatically remove rules that are not EC - want to include a percent data cutoff - could be same as for iterations passing, so that for small datasets, we alway use epoch complete
        #but for big ones we use experienced ENC rules.
        #Also revisit adjustment of all fitness to be no higher than best area ratio when over the front in extrapolation.
        #self.reportClassifier('lastupdate')

        #Recalculate coverDiff (this should not have changed)  This could be stored, so as not to have to recalculate
        #coverDiff = self.correctCover - self.phenotype_RP*self.matchCover
        indFitness = None
        #Criteria for a fitness update:
        #print(self.coverDiff)
        #if self.partOfCorrect and coverDiff > 0 and self.matchedAndFrontEstablished == True and (len(cons.env.formatData.necFront.paretoFrontAcc) > 0 or len(cons.env.formatData.ecFront.paretoFrontAcc) > 0): #fitness only changes if was last part of a correct set - cause otherwise releativePreFitness stays at 0
        if self.coverDiff > 0 and self.matchedAndFrontEstablished == True and (len(cons.env.formatData.necFront.paretoFrontAcc) > 0 or len(cons.env.formatData.ecFront.paretoFrontAcc) > 0): #fitness only changes if was last part of a correct set - cause otherwise releativePreFitness stays at 0
#             print 'NEW NEW NEW NEW NEW NEW NEW NEW'
#             print 'Before correction-----------------------'
#             print 'fitTYPE= ' +str(self.isMatchSetFitness)
#             print 'fit= '+str(self.fitness)
#             print 'pFit= '+str(self.preFitness)
#             print 'uDiff= '+str(self.coverDiff)
            #lastPreFitness = copy.deepcopy(self.indFitness)

            #EXTRAPOLATE coverDiff up to number of training instances (i.e. age at epoch complete) This changes because more iterations have gone by.*******
            ruleAge = exploreIter - self.initTimeStamp+1 #Correct, because we include the current instance we are on.
            coverDiff = self.coverDiff*cons.env.formatData.numTrainInstances/float(ruleAge)
#             if coverDiff > self.coverDiff and self.coverDiff != None:
#                 print 'exploreIter= '+str(exploreIter)
#                 print 'InitTimestamp+1= ' + str(self.initTimeStamp+1)
#                 print 'ruleAge= '+ str(ruleAge)
#                 x = 5/0

            #Get new pareto fitness
            objectivePair = [self.accuracyComponent,coverDiff]
            #BEFORE PARETO FRONTS BEGIN TO BE UPDATED
            if len(cons.env.formatData.ecFront.paretoFrontAcc) > 0: #Leave epoch incomplete front behind.
                indFitness = cons.env.formatData.ecFront.getParetoFitness(objectivePair)
                #print 'EC'
            else: #Keep updating and evaluating with epoch incomplete front.
                indFitness = cons.env.formatData.necFront.getParetoFitness(objectivePair)
                #print 'ENC'
#             print 'pFit= '+str(self.indFitness
            indFitness = math.pow(indFitness,cons.nu)

#             if self.lastPreFitness < self.indFitness or self.lastPreFitness < indFitness:
#                 if self.initTimeStamp == 2:
#                     print "SWITCHING OVER FROM ACCURCY TO PARETO FRONT"
#                     self.reportClassifier('update')
                    #x = 5/0

            #Calculate new adjusted fitness by using last matching update score and recalculating.  (preserve originals so that updates are always based on originals, not on updated estimate)
            tempSumIndFitness = copy.deepcopy(self.sumIndFitness)
            #tempSumIndFitness = tempSumIndFitness - self.indFitness*self.numerosity   #this is not the true sum, but the sum with out the current rule's fitness.
            tempSumIndFitness = tempSumIndFitness - self.indFitness
            if self.lastIndFitness != indFitness: #ok because with many new rules they may still be maxed out at highest fitness possible.
#             if self.epochComplete: #Fitness shared only with other EC rules.
#                 self.relativeIndFitness = self.indFitness*self.numerosity / self.sumPreFitness
#             else:
#                 self.relativeIndFitness = self.indFitness*self.numerosity*(exploreIter-self.initTimeStamp+1) / self.sumPreFitness
                #NOTE - have to re-adjust sumprefitnessnec to account for change in indFitness

                #Readjust sumIndFitness with new indFitness information. (this is an estimate, because the other rules may have changed.
                #tempSumIndFitness = tempSumIndFitness + indFitness*self.numerosity
                tempSumIndFitness = tempSumIndFitness + indFitness
                #self.relativeIndFitness = indFitness*self.numerosity / tempSumIndFitness
                self.relativeIndFitness = indFitness/float(tempSumIndFitness)

                percRuleExp = (exploreIter-self.initTimeStamp+1)/float(cons.env.formatData.numTrainInstances)
                #Consider the useful accuracy cutoff -  -maybe use a less dramatic change or ...na...
                beta = 0.2
                if self.matchCount >= 1.0/beta:

                    self.fitness = self.lastMatchFitness + beta*percRuleExp*(self.relativeIndFitness-self.lastMatchFitness)
                elif self.matchCount == 1 or self.aveRelativeIndFitness == None:  #second condition handles special case after GA rule generated, but not has not gone through full matching yet
                    #print 'fit B'
                    self.fitness = self.relativeIndFitness
                    #self.aveRelativeIndFitness = self.relativeIndFitness
        #                 if self.initTimeStamp == 2:
        #                     print self.aveRelativePreFitness
        #                     print 5/0
                else:
                    #print 'fit C'
                    self.fitness = (self.aveRelativeIndFitness*(self.matchCount-1)+self.relativeIndFitness)/self.matchCount  #often, last releative prefitness is 0!!!!!!!!!!!!!!!!!!!!!!!
                    #self.aveRelativeIndFitness = (self.aveRelativeIndFitness*(self.matchCount-1)+self.relativeIndFitness)/self.matchCount

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

                #self.isMatchSetFitness = False
#                 x= 5/0
#                 if self.initTimeStamp == 2:
#                     self.reportClassifier('update')
            else: #No fitness extrapolation update required
#                 if self.initTimeStamp == 2:
#                     print 'no update required B'
                pass
        else: #No fitness extrapolation update required
#             if self.initTimeStamp == 2:
#                 print 'no update required A'
            pass

        #self.reportClassifier('update')
        if round(self.fitness,5) > 1:
            self.fitness = 1.0
            print('FITNESS ERROR - adjust - too high')
        if self.fitness < 0:
            self.fitness = 0.0
            print('FITNESS ERROR - adjust - too low')
            #print self.fitness

            #x = 5/0

        self.lastIndFitness = copy.deepcopy(indFitness)  # TARGET - won't this cause below to never run?  - also where is lastIndFitness first stored??don't see it above.
        self.fitness = self.indFitness


    def updateNumerosity(self, num):
        """ Alters the numberosity of the classifier.  Notice that num can be negative! """
        self.numerosity += num


#    def updateMatchSetSize(self, matchSetSize):
#        """  Updates the average match set size. """
#        if self.matchCount < 1.0 / cons.beta:
#            self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount-1)+ matchSetSize) / float(self.matchCount)
#        else:
#            self.aveMatchSetSize = self.aveMatchSetSize + cons.beta * (matchSetSize - self.aveMatchSetSize)


    def updateMatchSetSize(self, matchSetSize):
        """  Updates the average match set size. """
        if self.matchCount == 1:
            self.aveMatchSetSize = matchSetSize
        elif self.matchCount < 1.0 / cons.beta: # < 5
            self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount-1)+ matchSetSize) / float(self.matchCount)
            #If matchCount = 2 -- 1 *
        else:
            self.aveMatchSetSize = self.aveMatchSetSize + cons.beta * (matchSetSize - self.aveMatchSetSize)


#    def updateCorrectSetSize(self, correctSetSize):
#        """  Updates the average match set size. """
#        if self.correctCount == 1:
#            self.aveCorrectSetSize = correctSetSize
#        if self.correctCount < 1.0 / cons.beta: # < 5
#            self.aveCorrectSetSize = (self.aveCorrectSetSize * (self.correctCount-1)+ correctSetSize) / float(self.correctCount)
#            #If matchCount = 2 -- 1 *
#        else:
#            self.aveCorrectSetSize = self.aveCorrectSetSize + cons.beta * (correctSetSize - self.aveCorrectSetSize)


    def updateTimeStamp(self, ts):
        """ Sets the time stamp of the classifier. """
        self.timeStampGA = ts


    def setAccuracy(self,acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc


    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit


    def calcClassifierStateFreq(self):
        #Fitness Frequency Calculation ------------------------------------------------------------------
        #-----------------------------------------------------------------------------------
        # CALCULATE STATE FREQUENCY COMPONENT
        #-----------------------------------------------------------------------------------
        self.totalFreq = 1
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
        #print self.totalFreq

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PRINT CLASSIFIER FOR POPULATION OUTPUT FILE
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def printClassifier(self):
        """ Formats and returns an output string describing this classifier. """
        classifierString = ""
        classifierString += str(self.specifiedAttList) + "\t"
        classifierString += str(self.condition) + "\t"
        #-------------------------------------------------------------------------------
        specificity = len(self.condition) / float(cons.env.formatData.numAttributes)
        epoch = 0
        if self.epochComplete:
            epoch = 1
        classifierString += str(self.phenotype)+"\t"
#         if cons.env.formatData.discretePhenotype:
#             classifierString += str(self.phenotype)+"\t"
#         else: #ContinuousCode #########################
#             classifierString += str(self.phenotype[0])+';'+str(self.phenotype[1])+"\t"
        self.globalFitness = self.fitness*self.indFitness
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        classifierString += str(self.fitness)+"\t"+str(self.accuracy)+"\t"+str(self.numerosity)+"\t"+str(self.aveMatchSetSize)+"\t"+str(self.timeStampGA)+"\t"+str(self.initTimeStamp)+"\t"+str(specificity)+"\t"
        classifierString += str(self.deletionVote)+"\t"+str(self.correctCount)+"\t"+str(self.matchCount)+"\t"+str(self.correctCover)+"\t"+str(self.matchCover)+"\t"+str(epoch)+"\t"+str(self.accuracyComponent)+"\t"+str(self.coverDiff)+"\t"+str(self.indFitness)+"\t"+str(self.fitness)+"\t"+str(self.globalFitness)+"\t"+str(self.totalFreq)+"\n"
#         classifierString += str(self.fitness)+"\t"+str(self.accuracy)+"\t"+str(self.scaledUsefulAccuracy)+"\t"+str(self.accuracyComponent)+"\t"+str(self.diffRatio)+"\t"+str(self.diffComponent)+"\t"+str(self.totalFreq)+"\t"+str(self.freqComponent)+"\t"+str(self.numerosity)+"\t"+str(self.aveMatchSetSize)+"\t"+str(self.timeStampGA)+"\t"+str(self.initTimeStamp)+"\t"+str(specificity)+"\t"
#         classifierString += str(self.deletionVote)+"\t"+str(self.correctCount)+"\t"+str(self.matchCount)+"\t"+str(self.correctCover)+"\t"+str(self.matchCover)+"\t"+str(epoch)+"\n"
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return classifierString



