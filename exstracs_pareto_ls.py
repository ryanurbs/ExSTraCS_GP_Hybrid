"""
Name:        exstracs_pareto_ls.py
Authors:     Ryan Urbanowicz - Written at Upenn, Philadelphia, PA, USA
Contact:     ryanurb@upenn.edu
Created:     Dec 21, 2015
Description: Code for defining a pareto front and calculating the pareto fitness landscape.

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
import matplotlib.pyplot as plt
from exstracs_constants import *
from exstracs_pareto import *

class FitnessLandscape:
    def __init__(self, population):
        self.relevantFront = None
        self.population = population
        if len(cons.env.formatData.ecFront.paretoFrontAcc) > 0: #Is EC front filled with some content?
            self.relevantFront = cons.env.formatData.ecFront
        else:
            self.relevantFront = cons.env.formatData.necFront

        self.granularity = 0.01 #Fitness landscape granularity

        print('Gathering fitness scores...')
        fitness_scores = self.calculateFitnessLandscape()
        print('Plotting the fitness landscape...')
        self.plotFitnessLandscape(fitness_scores)


    def calculateFitnessLandscape(self):
        """  """
        fitness_scores = []
    #     for accuracy in np.arange(0, 1.0, granularity):
    #         for coverage in np.arange(0, 1.0, granularity):
        for accuracy in np.arange(0, 1.1, self.granularity):
            for coverage in np.arange(0, 1.1, self.granularity):
                fitness_scores.append(self.relevantFront.getParetoFitness([accuracy, coverage * self.relevantFront.coverMax]))

        fitness_scores = np.array(fitness_scores)
        #fitness_scores.shape = (int(1. / granularity), int(1. / granularity))
        fitness_scores.shape = (int(1.1 / self.granularity), int(1.1 / self.granularity))
        return fitness_scores


    def plotFitnessLandscape(self, fitness_scores):
        #Visualization goals:
        #1. Show the fitness landscape domain (x/y axes each from 0 to 1)
        #2. Show connected points that make up the pareto front
        #3. Based on the specified granularity and a scale of fitness magnitude also from 0 to 1 given as a color or greyscale 2D landscape within the x/y axes 0-1.
        #4. Plot with small black dots or x's the 'target points'  so the the user can see where a specific rule with fall within the fitness landscape.

        plt.figure(figsize=(10, 10))
        plt.imshow(fitness_scores, interpolation='nearest', cmap='viridis')
        plt.ylim(0, 1.1/self.granularity)
        plt.yticks([0, 1./(2*self.granularity), 1./self.granularity], [0, 0.5, 1], fontsize=12)
        plt.xticks([0, 1./(2*self.granularity), 1./self.granularity], [0, 0.5*self.relevantFront.coverMax, self.relevantFront.coverMax], fontsize=12)
        plt.colorbar(shrink=0.8)
        plt.xlabel('Normalized Coverage', fontsize=14)
        plt.ylabel('Useful Accuracy', fontsize=14)

        plt.plot(np.array(self.relevantFront.paretoFrontRawCov) / self.relevantFront.coverMax * 1./self.granularity,
                 np.array(self.relevantFront.paretoFrontAcc) * 1./self.granularity,
                 'o-', ms=10, lw=2, color='black')

        #Go through each rule in the current population, obtain it's accuracy and coverage, and place point on plot
#         targetPointRawCovEC = []
#         targetPointAccEC = []
#         targetPointRawCovNEC = []
#         targetPointAccNEC = []
#         for i in range(len(self.population.popSet)):
#             cl = self.population.popSet[i]
#             if cl.coverDiff > 0:
#                 if cl.epochComplete:
#                     targetPointAccEC.append(cl.accuracyComponent)
#                     targetPointRawCovEC.append(cl.coverDiff)
#                 else:
#                     if cl.coverDiff <= (self.relevantFront.coverMax + self.relevantFront.coverMax*.1): #Limit to graphing
#                         targetPointAccNEC.append(cl.accuracyComponent)
#                         targetPointRawCovNEC.append(cl.coverDiff)
#                     else:
#                         targetPointAccNEC.append(cl.accuracyComponent)
#                         targetPointRawCovNEC.append(self.relevantFront.coverMax + self.relevantFront.coverMax*.1)
#
#         plt.plot(np.array(targetPointRawCovEC) / self.relevantFront.coverMax * 1./self.granularity,
#              np.array(targetPointAccEC) * 1./self.granularity,
#              'o', ms=5, lw=1, color='black')
#
#         plt.plot(np.array(targetPointRawCovNEC) / self.relevantFront.coverMax * 1./self.granularity,
#              np.array(targetPointAccNEC) * 1./self.granularity,
#              'o', ms=5, lw=1, color='blue')

        """

        targetPointRawCov0 = []
        targetPointAcc0 = []
        targetPointRawCov1 = []
        targetPointAcc1 = []
        targetPointRawCov2 = []
        targetPointAcc2 = []
        targetPointRawCov3 = []
        targetPointAcc3 = []
        targetPointRawCov4 = []
        targetPointAcc4 = []
        targetPointRawCov5 = []
        targetPointAcc5 = []
        targetPointRawCov6 = []
        targetPointAcc6 = []
        targetPointRawCov7 = []
        targetPointAcc7 = []
        for i in range(len(self.population.popSet)):
            cl = self.population.popSet[i]
            if cl.coverDiff > 0:
                if cl.isTree:
                    if cl.coverDiff <= (self.relevantFront.coverMax + self.relevantFront.coverMax*.1): #Limit to graphing
                        targetPointAcc0.append(cl.accuracyComponent)
                        targetPointRawCov0.append(cl.coverDiff)
                    else:
                        targetPointAcc0.append(cl.accuracyComponent)
                        targetPointRawCov0.append(self.relevantFront.coverMax + self.relevantFront.coverMax*.1)
                elif len(cl.specifiedAttList) == 1:
                    if cl.coverDiff <= (self.relevantFront.coverMax + self.relevantFront.coverMax*.1): #Limit to graphing
                        targetPointAcc1.append(cl.accuracyComponent)
                        targetPointRawCov1.append(cl.coverDiff)
                    else:
                        targetPointAcc1.append(cl.accuracyComponent)
                        targetPointRawCov1.append(self.relevantFront.coverMax + self.relevantFront.coverMax*.1)
                elif len(cl.specifiedAttList) == 2:
                    if cl.coverDiff <= (self.relevantFront.coverMax + self.relevantFront.coverMax*.1): #Limit to graphing
                        targetPointAcc2.append(cl.accuracyComponent)
                        targetPointRawCov2.append(cl.coverDiff)
                    else:
                        targetPointAcc2.append(cl.accuracyComponent)
                        targetPointRawCov2.append(self.relevantFront.coverMax + self.relevantFront.coverMax*.1)
                elif len(cl.specifiedAttList) == 3:
                    if cl.coverDiff <= (self.relevantFront.coverMax + self.relevantFront.coverMax*.1): #Limit to graphing
                        targetPointAcc3.append(cl.accuracyComponent)
                        targetPointRawCov3.append(cl.coverDiff)
                    else:
                        targetPointAcc3.append(cl.accuracyComponent)
                        targetPointRawCov3.append(self.relevantFront.coverMax + self.relevantFront.coverMax*.1)
                elif len(cl.specifiedAttList) == 4:
                    if cl.coverDiff <= (self.relevantFront.coverMax + self.relevantFront.coverMax*.1): #Limit to graphing
                        targetPointAcc4.append(cl.accuracyComponent)
                        targetPointRawCov4.append(cl.coverDiff)
                    else:
                        targetPointAcc4.append(cl.accuracyComponent)
                        targetPointRawCov4.append(self.relevantFront.coverMax + self.relevantFront.coverMax*.1)
                elif len(cl.specifiedAttList) == 5:
                    if cl.coverDiff <= (self.relevantFront.coverMax + self.relevantFront.coverMax*.1): #Limit to graphing
                        targetPointAcc5.append(cl.accuracyComponent)
                        targetPointRawCov5.append(cl.coverDiff)
                    else:
                        targetPointAcc5.append(cl.accuracyComponent)
                        targetPointRawCov5.append(self.relevantFront.coverMax + self.relevantFront.coverMax*.1)
                elif len(cl.specifiedAttList) == 6:
                    if cl.coverDiff <= (self.relevantFront.coverMax + self.relevantFront.coverMax*.1): #Limit to graphing
                        targetPointAcc6.append(cl.accuracyComponent)
                        targetPointRawCov6.append(cl.coverDiff)
                    else:
                        targetPointAcc6.append(cl.accuracyComponent)
                        targetPointRawCov6.append(self.relevantFront.coverMax + self.relevantFront.coverMax*.1)
                else:
                    if cl.coverDiff <= (self.relevantFront.coverMax + self.relevantFront.coverMax*.1): #Limit to graphing
                        targetPointAcc7.append(cl.accuracyComponent)
                        targetPointRawCov7.append(cl.coverDiff)
                    else:
                        targetPointAcc7.append(cl.accuracyComponent)
                        targetPointRawCov7.append(self.relevantFront.coverMax + self.relevantFront.coverMax*.1)


        plt.plot(np.array(targetPointRawCov0) / self.relevantFront.coverMax * 1./self.granularity,
             np.array(targetPointAcc0) * 1./self.granularity,
             'o', ms=5, lw=1, color='orange')

        plt.plot(np.array(targetPointRawCov1) / self.relevantFront.coverMax * 1./self.granularity,
             np.array(targetPointAcc1) * 1./self.granularity,
             'o', ms=5, lw=1, color='black')

        plt.plot(np.array(targetPointRawCov2) / self.relevantFront.coverMax * 1./self.granularity,
             np.array(targetPointAcc2) * 1./self.granularity,
             'o', ms=5, lw=1, color='blue')

        plt.plot(np.array(targetPointRawCov3) / self.relevantFront.coverMax * 1./self.granularity,
             np.array(targetPointAcc3) * 1./self.granularity,
             'o', ms=5, lw=1, color='green')

        plt.plot(np.array(targetPointRawCov4) / self.relevantFront.coverMax * 1./self.granularity,
             np.array(targetPointAcc4) * 1./self.granularity,
             'o', ms=5, lw=1, color='red')

        plt.plot(np.array(targetPointRawCov5) / self.relevantFront.coverMax * 1./self.granularity,
             np.array(targetPointAcc5) * 1./self.granularity,
             'o', ms=5, lw=1, color='cyan')

        plt.plot(np.array(targetPointRawCov6) / self.relevantFront.coverMax * 1./self.granularity,
             np.array(targetPointAcc6) * 1./self.granularity,
             'o', ms=5, lw=1, color='magenta')

        plt.plot(np.array(targetPointRawCov7) / self.relevantFront.coverMax * 1./self.granularity,
             np.array(targetPointAcc7) * 1./self.granularity,
             'o', ms=5, lw=1, color='white')
    """
        targetPointRawCov0 = []
        targetPointAcc0 = []
        targetPointRawCov1 = []
        targetPointAcc1 = []
        targetPointRawCov2 = []
        targetPointAcc2 = []
        targetPointRawCov3 = []
        targetPointAcc3 = []

        for i in range(len(self.population.popSet)):
            cl = self.population.popSet[i]
            if cl.coverDiff > 0:
                if cl.isTree:
                    print("One: " + str(cl.one_count) + " Zero: " + str(cl.zero_count) + " Total: " + str(cl.zero_count + cl.one_count) + " Epoch: " + str(cl.epochComplete) + "MC: " +str(cl.matchCount))
                    if cl.epochComplete:
                        if cl.coverDiff <= (self.relevantFront.coverMax + self.relevantFront.coverMax*.1): #Limit to graphing
                            targetPointAcc0.append(cl.accuracyComponent)
                            targetPointRawCov0.append(cl.coverDiff)
                        else:
                            targetPointAcc0.append(cl.accuracyComponent)
                            targetPointRawCov0.append(self.relevantFront.coverMax + self.relevantFront.coverMax*.1)
                    else:
                        if cl.coverDiff <= (self.relevantFront.coverMax + self.relevantFront.coverMax*.1): #Limit to graphing
                            targetPointAcc1.append(cl.accuracyComponent)
                            targetPointRawCov1.append(cl.coverDiff)
                        else:
                            targetPointAcc1.append(cl.accuracyComponent)
                            targetPointRawCov1.append(self.relevantFront.coverMax + self.relevantFront.coverMax*.1)

                else:
                    if cl.epochComplete:
                        if cl.coverDiff <= (self.relevantFront.coverMax + self.relevantFront.coverMax*.1): #Limit to graphing
                            targetPointAcc2.append(cl.accuracyComponent)
                            targetPointRawCov2.append(cl.coverDiff)
                        else:
                            targetPointAcc2.append(cl.accuracyComponent)
                            targetPointRawCov2.append(self.relevantFront.coverMax + self.relevantFront.coverMax*.1)
                    else:
                        if cl.coverDiff <= (self.relevantFront.coverMax + self.relevantFront.coverMax*.1): #Limit to graphing
                            targetPointAcc3.append(cl.accuracyComponent)
                            targetPointRawCov3.append(cl.coverDiff)
                        else:
                            targetPointAcc3.append(cl.accuracyComponent)
                            targetPointRawCov3.append(self.relevantFront.coverMax + self.relevantFront.coverMax*.1)


        plt.plot(np.array(targetPointRawCov0) / self.relevantFront.coverMax * 1./self.granularity,
             np.array(targetPointAcc0) * 1./self.granularity,
             'o', ms=5, lw=1, color='red')

        plt.plot(np.array(targetPointRawCov1) / self.relevantFront.coverMax * 1./self.granularity,
             np.array(targetPointAcc1) * 1./self.granularity,
             'o', ms=5, lw=1, color='orange')

        plt.plot(np.array(targetPointRawCov2) / self.relevantFront.coverMax * 1./self.granularity,
             np.array(targetPointAcc2) * 1./self.granularity,
             'o', ms=5, lw=1, color='blue')

        plt.plot(np.array(targetPointRawCov3) / self.relevantFront.coverMax * 1./self.granularity,
             np.array(targetPointAcc3) * 1./self.granularity,
             'o', ms=5, lw=1, color='green')



    #Consider putting together a density plot superimposed on fitness plot.

        #Dotted line from origin to maxCoverage front point
    #     plt.plot([0, paretoFrontRawCov[0] / coverMax * 1./granularity],
    #              [0, paretoFrontAcc[0] * 1./granularity],
    #              '--', lw=2,
    #              color='black')
    #     #Dotted line from origin to maxAcuracy front point
    #     plt.plot([0, paretoFrontRawCov[-1] / coverMax * 1./granularity],
    #              [0, paretoFrontAcc[-1] * 1./granularity],
    #              '--', lw=2,
    #              color='black')
        #Dotted line from maxCoverage front point to zero boundary
    #     plt.plot([paretoFrontRawCov[0] / coverMax * 1./granularity, paretoFrontRawCov[0] / coverMax * 1./granularity],
    #              [0, paretoFrontAcc[0] * 1./granularity],
    #              '--', lw=2,
    #              color='black')

        plt.show()





#     def getParetoFitness(self, objectivePair):
#         """ Determines and returns the pareto fitness based on the proportional distance of a given point"""
#         objectiveCoverVal = objectivePair[1] / self.coverMax
#         #Special Case
#         i = 0#-1
#         foundPerp = False
#         badDist = None
#         while not foundPerp:
#             #print i
#             mainLineSlope = self.calcSlope(0, objectivePair[0], 0, objectiveCoverVal)   #check that we are alwasys storing objective cover val, so there is no error here (recent thought)
#             if self.paretoFrontAcc[i] == objectivePair[0] and self.paretoFrontRawCov[i] == objectivePair[1]: #is point a point on front?
#                 #print "POINT ON FRONT"
#                 goodDist = 1
#                 badDist = 0
#                 foundPerp = True
#             else:
#                 frontPointSlope = self.calcSlope(0, self.paretoFrontAcc[i], 0, self.paretoFrontRawCov[i]/self.coverMax)
#                 if i == 0 and frontPointSlope >= mainLineSlope: #Special Case:  High Coverage boundary case
#                     foundPerp = True
#                     if objectiveCoverVal >= self.paretoFrontRawCov[i]/self.coverMax: #Over front treated like maximum indfitness
#                         if objectivePair[0] >= self.paretoFrontAcc[i]:
#                             goodDist = 1
#                             badDist = 0
#                         else:
#     #                         goodDist = calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#     #                         badDist = calcDistance(0,paretoFrontAcc[i],0,paretoFrontRawCov[i]/coverMax) - goodDist
#                             goodDist = self.calcDistance(0,objectivePair[0],0,1)
#                             badDist = self.calcDistance(0,self.paretoFrontAcc[i],0,1) - goodDist
#     #                         goodDist = objectivePair[0]
#     #                         badDist = paretoFrontAcc[i]-objectivePair[0]
#     #                         goodDist = calcDistance(0,objectivePair[0],paretoFrontRawCov[i]/coverMax,1)
#     #                         badDist = calcDistance(paretoFrontAcc[i],objectivePair[0],paretoFrontRawCov[i]/coverMax,1)
#                     elif objectiveCoverVal == self.paretoFrontRawCov[i]/self.coverMax: #On the boundary Line but not a point on the front. - some more minor penalty.
#                         goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#                         badDist = self.calcDistance(0,self.paretoFrontAcc[i],0,self.paretoFrontRawCov[i]/self.coverMax) - goodDist
#                     else: #Maximum penalty case - point is a boundary case underneath the front.
#                         goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#                         badDist = self.calcDistance(self.paretoFrontAcc[i],objectivePair[0],self.paretoFrontRawCov[i]/self.coverMax, objectiveCoverVal)
#                         goodDist += objectivePair[0]
#                         badDist += self.paretoFrontAcc[i]-objectivePair[0]
#
#                 elif i == len(self.paretoFrontAcc)-1 and frontPointSlope < mainLineSlope: #Special Case:  High Accuracy boundary case
#                     foundPerp = True
#                     if objectivePair[0] > self.paretoFrontAcc[i]: #Over front treated like maximum indfitness
#                         goodDist = 1
#                         badDist = 0
#     #                     if objectiveCoverVal >= paretoFrontRawCov[i]/coverMax:
#     #                         goodDist = 1
#     #                         badDist = 0
#     #                     else:
#     #                         goodDist = calcDistance(0,1,0,objectiveCoverVal)
#     #                         badDist = calcDistance(0,1,0,paretoFrontRawCov[i]/coverMax) - goodDist
#     #                         goodDist = objectiveCoverVal
#     #                         badDist = paretoFrontRawCov[i]/coverMax-objectiveCoverVal
#
#                     elif objectivePair[0] == self.paretoFrontAcc[i]: #On the boundary Line but not a point on the front. - some more minor penalty.
#                         if self.paretoFrontAcc[i] == 1.0:
#                             goodDist = 1
#                             badDist = 0
#                         else:
#                             goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#                             badDist = self.calcDistance(0,self.paretoFrontAcc[i],0,self.paretoFrontRawCov[i]/self.coverMax) - goodDist
#                     else: #Maximum penalty case - point is a boundary case underneath the front.
#                         goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#                         badDist = self.calcDistance(self.paretoFrontAcc[i],objectivePair[0],self.paretoFrontRawCov[i]/self.coverMax, objectiveCoverVal)
#                         goodDist += objectiveCoverVal
#                         badDist += self.paretoFrontRawCov[i]/self.coverMax-objectiveCoverVal
#
#                 elif frontPointSlope >= mainLineSlope: #Normal Middle front situation (we work from high point itself up to low point - but not including)
#                     foundPerp = True
#                     #boundaryCalculation Rule = (x1-x0)(y2-y0)-(x2-x0)(y1-y0), where 0 and 1 represent two points with a line, and 2 represents some other point.
#                     boundaryCalculation = (self.paretoFrontAcc[i]-self.paretoFrontAcc[i-1])*(objectiveCoverVal-self.paretoFrontRawCov[i-1]/self.coverMax) - (objectivePair[0]-self.paretoFrontAcc[i-1])*(self.paretoFrontRawCov[i]/self.coverMax-self.paretoFrontRawCov[i-1]/self.coverMax)
#                     if boundaryCalculation > 0:
#                         goodDist = 1
#                         badDist = 0
#                     else:
#                         frontIntercept =  self.calcIntercept(self.paretoFrontAcc[i], self.paretoFrontAcc[i-1], self.paretoFrontRawCov[i]/self.coverMax, self.paretoFrontRawCov[i-1]/self.coverMax, mainLineSlope)
#                         badDist = self.calcDistance(frontIntercept[0],objectivePair[0],frontIntercept[1],objectiveCoverVal)
#                         goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#
#                 else:
#                     i += 1
#
#         paretoFitness = goodDist / float(goodDist + badDist)
#         return paretoFitness


#     def calcDistance(self, y1, y2, x1, x2):
#         distance = math.sqrt(math.pow(x2-x1,2) + math.pow(y2-y1,2))
#         return distance
#
#
#     def calcIntercept(self, y1a, y2a, x1a, x2a, mainLineSlope):
#         """  Calculates the coordinates at which the two lines 'A' defined by points and 'B', defined by a point and slope, intersect """
#         slopeA = self.calcSlope(y1a, y2a, x1a, x2a)
#         slopeB = mainLineSlope
#         if slopeA == 0:
#             xIntercept = x1a
#             yIntercept = slopeB*xIntercept
#
#         else:
#             xIntercept = (y2a - slopeA*x2a) / float(slopeB-slopeA)
#             yIntercept = slopeB*xIntercept
#         return [yIntercept,xIntercept]
#
#
#     def calcSlope(self, y1, y2, x1, x2):
#         """ Calculate slope between two points """
#         if x2-x1 == 0:
#             slope = 0
#         else:
#             slope = (y2 - y1) / (x2 - x1)
#         return slope
#

