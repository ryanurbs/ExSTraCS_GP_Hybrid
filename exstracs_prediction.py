"""
Name:        ExSTraCS_Prediction.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: Based on a given match set, this module uses a voting scheme to select the phenotype prediction for ExSTraCS.

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

#Import Required Modules-------------------------------
from exstracs_constants import *
import random
#------------------------------------------------------

class Prediction:
    def __init__(self, population, exploreIter):  #now takes in population ( have to reference the match set to do prediction)  pop.matchSet
        """ Constructs the voting array and determines the prediction decision. """
        self.decision = None
        #self.classCount
        #-------------------------------------------------------
        # DISCRETE PHENOTYPE
        #-------------------------------------------------------
        if cons.env.formatData.discretePhenotype:
            self.vote = {}
            self.tieBreak_Numerosity = {}
            self.tieBreak_TimeStamp = {}

            for eachClass in cons.env.formatData.phenotypeList:
                self.vote[eachClass] = 0.0
                self.tieBreak_Numerosity[eachClass] = 0
                self.tieBreak_TimeStamp[eachClass] = 0.0

            for ref in population.matchSet:
                cl = population.popSet[ref]
                if cons.firstEpochComplete and cl.epochComplete:
                    #self.vote[cl.phenotype] += cl.fitness * cl.numerosity
                    #self.vote[cl.phenotype] += cl.fitness * cl.numerosity# * cons.env.formatData.classPredictionWeights[cl.phenotype]
                    self.vote[cl.phenotype] += cl.fitness * cl.indFitness * cl.numerosity# * cons.env.formatData.classPredictionWeights[cl.phenotype]
                    #self.vote[cl.phenotype] += cl.fitness * cons.env.formatData.classPredictionWeights[cl.phenotype]
                    #self.vote[cl.phenotype] += cl.fitness
                    self.tieBreak_Numerosity[cl.phenotype] += cl.numerosity
                    self.tieBreak_TimeStamp[cl.phenotype] += cl.initTimeStamp
                elif not cons.firstEpochComplete:
                    ageDiscount = (exploreIter-cl.initTimeStamp+1.0)/float(cons.env.formatData.numTrainInstances)
                    self.vote[cl.phenotype] += ageDiscount*cl.fitness * cl.indFitness * cl.numerosity# * cons.env.formatData.classPredictionWeights[cl.phenotype]
                    self.tieBreak_Numerosity[cl.phenotype] += ageDiscount*cl.numerosity
                    self.tieBreak_TimeStamp[cl.phenotype] += cl.initTimeStamp

            highVal = 0.0
            bestClass = [] #Prediction is set up to handle best class ties for problems with more than 2 classes
            for thisClass in cons.env.formatData.phenotypeList:
                if self.tieBreak_Numerosity[thisClass] > 0:
                    if self.vote[thisClass]/self.tieBreak_Numerosity[thisClass] >= highVal:
                        highVal = self.vote[thisClass]/self.tieBreak_Numerosity[thisClass]

            for thisClass in cons.env.formatData.phenotypeList:
                if self.tieBreak_Numerosity[thisClass] > 0:
                    if self.vote[thisClass]/self.tieBreak_Numerosity[thisClass] == highVal: #Tie for best class
                        bestClass.append(thisClass)
            #---------------------------
            if highVal == 0.0:
                self.decision = None
            #-----------------------------------------------------------------------
            elif len(bestClass) > 1: #Randomly choose between the best tied classes
                bestNum = 0
                newBestClass = []
                for thisClass in bestClass:
                    if self.tieBreak_Numerosity[thisClass] >= bestNum:
                        bestNum = self.tieBreak_Numerosity[thisClass]

                for thisClass in bestClass:
                    if self.tieBreak_Numerosity[thisClass] == bestNum:
                        newBestClass.append(thisClass)

                #-----------------------------------------------------------------------
                if len(newBestClass) > 1:  #still a tie
                    bestStamp = 0
                    newestBestClass = []
                    for thisClass in newBestClass:
                        if self.tieBreak_TimeStamp[thisClass] >= bestStamp:
                            bestStamp = self.tieBreak_TimeStamp[thisClass]

                    for thisClass in newBestClass:
                        if self.tieBreak_TimeStamp[thisClass] == bestStamp:
                            newestBestClass.append(thisClass)
                    #-----------------------------------------------------------------------
                    if len(newestBestClass) > 1: # Prediction is completely tied - ExSTraCS has no useful information for making a prediction
                        #self.decision = random.choice(newestBestClass)
                        self.decision = 'Tie'
                        #return a tie as a list to choose from (for multi-class - so a more educated guess may be made.
                else:
                    self.decision = newBestClass[0]
            #----------------------------------------------------------------------
            else: #One best class determined by fitness vote
                self.decision = bestClass[0]

        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        #-------------------------------------------------------
        #Grab unique upper and lower bounds from every rule in the match set.
        #Order list to discriminate pools
        #Go through M and add vote to each pool (i.e. fitness*numerosity)
        #Determine which (shortest) span has the largest vote.
        #Quick version is to take the centroid of this 'best' span
        #OR - identify any M rules that cover whole 'best' span, and use centroid voting that includes only these rules.
        else: #ContinuousCode #########################
            if len(population.matchSet) < 1:
                self.decision = None
            else:
                segmentList = []
                for ref in population.matchSet:
                    cl = population.popSet[ref]
                    if not cl.isTree:
                        high = cl.phenotype[1]
                        low = cl.phenotype[0]
                        if not high in segmentList:
                            segmentList.append(high)
                        if not low in segmentList:
                            segmentList.append(low)
                    else:
                        if cl.phenotype == None:
                            raise NameError("phenotype is none")
                        value = cl.phenotype
                        if not value in segmentList:
                            segmentList.append(value)
                segmentList.sort()
                voteList = []
                for i in range(0,len(segmentList)-1):
                    voteList.append(0)
                #PART 2
                for ref in population.matchSet:
                    cl = population.popSet[ref]
                    if not cl.isTree:
                        high = cl.phenotype[1]
                        low = cl.phenotype[0]
                        #j = 0
                        for j in range(len(segmentList)-1):
                            if low <= segmentList[j] and high >= segmentList[j+1]:
                                voteList[j] += cl.numerosity * cl.fitness * cl.indFitness
                    else:
                        for j in range(len(segmentList)-1):
                            if segmentList[j] <= value and value <= segmentList[j+1]:
                                voteList[j] += cl.numerosity * cl.fitness * cl.indFitness
                #PART 3
                bestVote = max(voteList)
                bestRef = voteList.index(bestVote)
                bestlow = segmentList[bestRef]
                besthigh = segmentList[bestRef+1]
                centroid = (bestlow + besthigh) / 2.0

                self.decision = centroid


#     def getFitnessSum(self,population,low,high):
#         """ Get the fitness Sum of rules in the rule-set. For continuous phenotype prediction. """
#         fitSum = 0
#         for ref in population.matchSet:
#             cl = population.popSet[ref]
#             if cl.phenotype[0] <= low and cl.phenotype[1] >= high: #if classifier range subsumes segment range.
#                 fitSum += cl.fitness
#         return fitSum


    def getDecision(self):
        """ Returns prediction decision. """
        return self.decision


    def getSet(self):
        """ Returns prediction decision. """
        return self.vote
