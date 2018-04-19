"""
Name:        ExSTraCS_Pareto.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     May 15, 2015
Modified:    May 15, 2015
Description: This module defines an individual Pareto front which defines the current best multiobjective fitness boundary used in the determination
of rule fitness.

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
import copy
import math

class Pareto:
    def __init__(self):     #def __init__(self, EClass, Epoch):
        #Definte the parts of the Pareto Front
        self.paretoFrontAcc = []
        self.paretoFrontRawCov = [] #will store the actual cover values
        self.coverMax = 0.0
        self.accuracyMax = 0.0

    #After updateing front calculate ans save AOC
    #Change pareto fitness calculator to calculate area under cerve

    def rebootPareto(self, frontAcc, frontRawCov):
        """ Reinitializes pareto front from loaded run. """
        #Reboot Accuracy
        for i in range(len(frontAcc)):
            self.paretoFrontAcc.append(float(frontAcc[i]))
        self.accuracyMax = self.paretoFrontAcc[-1]
        #Reboot Coverage
        for i in range(len(frontRawCov)):
            self.paretoFrontRawCov.append(float(frontRawCov[i]))
        self.coverMax = self.paretoFrontRawCov[0]


    def updateFront(self, objectivePair):
        """  Handles process of checking and adjusting the fitness pareto front. """
        #Update any changes to the maximum Cov - automatically adds point if new cov max is found
        #print self.classID + ' ' + self.epochID + '-------------------------------------'
        #print objectivePair
        changedFront = False
        if len(self.paretoFrontAcc) == 0:
            #print "first point addition"
            self.accuracyMax = objectivePair[0]
            self.paretoFrontAcc.append(objectivePair[0])
            self.coverMax = objectivePair[1]
            self.paretoFrontRawCov.append(objectivePair[1])
            changedFront = True

        elif len(self.paretoFrontAcc) == 1:
            #print "second point addition"
            if objectivePair[1] > self.coverMax:
                #print 'A'
                self.coverMax = objectivePair[1]
                if objectivePair[0] > self.accuracyMax:
                    #print '1*'
                    #Replace Point
                    self.accuracyMax = objectivePair[0]
                    self.paretoFrontAcc[0] = objectivePair[0]
                    self.paretoFrontRawCov[0] = objectivePair[1]
                    changedFront = True
                else:
                    #Add point
                    self.paretoFrontAcc.insert(0,objectivePair[0])
                    self.paretoFrontRawCov.insert(0,objectivePair[1])
                    changedFront = True

            else:
                #print 'B'
                if objectivePair[0] > self.accuracyMax:
                    self.accuracyMax = objectivePair[0]
                    #print '1*'
                    self.paretoFrontAcc.append(objectivePair[0])
                    self.paretoFrontRawCov.append(objectivePair[1])
                    changedFront = True
                else:
                    pass
        else: #Automated check and adjust when there are 2 or more points already on the front.
            #print "LARGER POINT ADDITION"
            oldParetoFrontRawCov = copy.deepcopy(self.paretoFrontRawCov)
            oldParetoFrontAcc = copy.deepcopy(self.paretoFrontAcc)
            self.paretoFrontAcc.append(objectivePair[0])
            self.paretoFrontRawCov.append(objectivePair[1])
            front = self.pareto_frontier(self.paretoFrontRawCov, self.paretoFrontAcc)
            #print front
            self.paretoFrontRawCov = front[0]
            self.paretoFrontAcc = front[1]
            self.coverMax = max(self.paretoFrontRawCov)
            self.accuracyMax = max(self.paretoFrontAcc)

            if oldParetoFrontRawCov != self.paretoFrontRawCov or oldParetoFrontAcc != self.paretoFrontAcc:
                changedFront = True
                #print oldParetoFrontRawCov
#
#         if changedFront:
#             #self.AUC = self.PolygonArea(copy.deepcopy(self.paretoFrontAcc), copy.deepcopy(self.paretoFrontRawCov))
#
#             #self.preFitMax = 0.0 #Reset because with new front, we can have a lower max (in particular 1 point fronts will have a max of 1.0 so otherwise we'd be stuck at 1.
#             #Determine maximum AUC ratio (i.e. maximum prefitness)
#             for i in range(len(self.paretoFrontAcc)):
#                 tempMaxAUC =  self.paretoFrontAcc[i]*self.paretoFrontRawCov[i]
#                 #print tempMaxAUC
#                 if tempMaxAUC > self.maxAUC:
#                     self.maxAUC = tempMaxAUC


                #tempParetoFitness = (self.paretoFrontAcc[i]*self.paretoFrontRawCov[i]/self.coverMax)/self.AUC
#                 tempParetoFitness = self.paretoFrontAcc[i]*self.paretoFrontRawCov[i]/self.AUC
#                 if tempParetoFitness > self.preFitMax:
#                     self.preFitMax = tempParetoFitness
 #NOTE, the best prefitMax might not be on the front - so this should be checked even when rule not updated.!!!!!!!!!!NOT FIXED YET

        self.verifyFront()  #TEMPORARY - DEBUGGING
        return changedFront


    def PolygonArea(self, accList, covList):
        #Shoelace Formula
        accList.insert(0,0)
        covList.insert(0,0)
        accList.insert(1,0)
        covList.insert(1,self.coverMax)
        accList.append(self.accuracyMax)
        covList.append(0)
        #print accList
        #print covList
        n = len(covList) # of corners
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += accList[i] * covList[j]#/self.coverMax
            area -= accList[j] * covList[i]#/self.coverMax
        area = abs(area) / 2.0
        #print area
        return area

#         accList.insert(0,0)
#         covList.insert(0,0)
#         accList.insert(1,0)
#         covList.insert(1,self.coverMax)
#         accList.append(self.accuracyMax)
#         covList.append(0)
#         #print accList
#         #print covList
#         n = len(covList) # of corners
#         area = 0.0
#         for i in range(n):
#             j = (i + 1) % n
#             area += accList[i] * covList[j]
#             area -= accList[j] * covList[i]
#         area = abs(area) / 2.0
#         #print area
#         return area


    def verifyFront(self):
        for i in range(len(self.paretoFrontAcc)-1):
            if self.paretoFrontAcc[i] > self.paretoFrontAcc[i+1]:
                print('ERROR: Accurcy error')
                x = 5/0
            if self.paretoFrontRawCov[i] < self.paretoFrontRawCov[i+1]:
                print('ERROR: Cov error')
                x = 5/0


    def getParetoFitness(self, objectivePair):
        """ Determines and returns the pareto fitness based on the proportional distance of a given point"""
        objectiveCoverVal = objectivePair[1] / self.coverMax
        #Special Case
        i = 0#-1
        foundPerp = False
        badDist = None
        while not foundPerp:
            #print i
            mainLineSlope = self.calcSlope(0, objectivePair[0], 0, objectiveCoverVal)   #check that we are alwasys storing objective cover val, so there is no error here (recent thought)
            if self.paretoFrontAcc[i] == objectivePair[0] and self.paretoFrontRawCov[i] == objectivePair[1]: #is point a point on front?
                #print "POINT ON FRONT"
                goodDist = 1
                badDist = 0
                foundPerp = True
            else:
                frontPointSlope = self.calcSlope(0, self.paretoFrontAcc[i], 0, self.paretoFrontRawCov[i]/self.coverMax)
                if i == 0 and frontPointSlope >= mainLineSlope: #Special Case:  High Coverage boundary case
                    foundPerp = True
                    if objectiveCoverVal >= self.paretoFrontRawCov[i]/self.coverMax: #Over front treated like maximum indfitness
                        if objectivePair[0] >= self.paretoFrontAcc[i]:
                            goodDist = 1
                            badDist = 0
                        else:
    #                         goodDist = calcDistance(0,objectivePair[0],0,objectiveCoverVal)
    #                         badDist = calcDistance(0,paretoFrontAcc[i],0,paretoFrontRawCov[i]/coverMax) - goodDist
                            goodDist = self.calcDistance(0,objectivePair[0],0,1)
                            badDist = self.calcDistance(0,self.paretoFrontAcc[i],0,1) - goodDist
    #                         goodDist = objectivePair[0]
    #                         badDist = paretoFrontAcc[i]-objectivePair[0]
    #                         goodDist = calcDistance(0,objectivePair[0],paretoFrontRawCov[i]/coverMax,1)
    #                         badDist = calcDistance(paretoFrontAcc[i],objectivePair[0],paretoFrontRawCov[i]/coverMax,1)
                    elif objectiveCoverVal == self.paretoFrontRawCov[i]/self.coverMax: #On the boundary Line but not a point on the front. - some more minor penalty.
                        goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                        badDist = self.calcDistance(0,self.paretoFrontAcc[i],0,self.paretoFrontRawCov[i]/self.coverMax) - goodDist
                    else: #Maximum penalty case - point is a boundary case underneath the front.
                        goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                        badDist = self.calcDistance(self.paretoFrontAcc[i],objectivePair[0],self.paretoFrontRawCov[i]/self.coverMax, objectiveCoverVal)
                        goodDist += objectivePair[0]
                        badDist += self.paretoFrontAcc[i]-objectivePair[0]

                elif i == len(self.paretoFrontAcc)-1 and frontPointSlope < mainLineSlope: #Special Case:  High Accuracy boundary case
                    foundPerp = True
                    if objectivePair[0] > self.paretoFrontAcc[i]: #Over front treated like maximum indfitness
                        goodDist = 1
                        badDist = 0

                    elif objectivePair[0] == self.paretoFrontAcc[i]: #On the boundary Line but not a point on the front. - some more minor penalty.
                        if self.paretoFrontAcc[i] == 1.0:
                            goodDist = 1
                            badDist = 0
                        else:
                            goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                            badDist = self.calcDistance(0,self.paretoFrontAcc[i],0,self.paretoFrontRawCov[i]/self.coverMax) - goodDist

#                         if objectiveCoverVal >= self.paretoFrontRawCov[i]/self.coverMax:
#                             goodDist = 1
#                             badDist = 0
#                         else:
#                             goodDist = self.calcDistance(0,1,0,objectiveCoverVal)
#                             badDist = self.calcDistance(0,1,0,self.paretoFrontRawCov[i]/self.coverMax) - goodDist
    #                         goodDist = objectiveCoverVal
    #                         badDist = paretoFrontRawCov[i]/coverMax-objectiveCoverVal

#                     elif objectivePair[0] == self.paretoFrontAcc[i]: #On the boundary Line but not a point on the front. - some more minor penalty.
#                         goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#                         badDist = self.calcDistance(0,self.paretoFrontAcc[i],0,self.paretoFrontRawCov[i]/self.coverMax) - goodDist
                    else: #Maximum penalty case - point is a boundary case underneath the front.
                        goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                        badDist = self.calcDistance(self.paretoFrontAcc[i],objectivePair[0],self.paretoFrontRawCov[i]/self.coverMax, objectiveCoverVal)
                        goodDist += objectiveCoverVal
                        badDist += self.paretoFrontRawCov[i]/self.coverMax-objectiveCoverVal

                elif frontPointSlope >= mainLineSlope: #Normal Middle front situation (we work from high point itself up to low point - but not including)
                    foundPerp = True
                    #boundaryCalculation Rule = (x1-x0)(y2-y0)-(x2-x0)(y1-y0), where 0 and 1 represent two points with a line, and 2 represents some other point.
                    boundaryCalculation = (self.paretoFrontAcc[i]-self.paretoFrontAcc[i-1])*(objectiveCoverVal-self.paretoFrontRawCov[i-1]/self.coverMax) - (objectivePair[0]-self.paretoFrontAcc[i-1])*(self.paretoFrontRawCov[i]/self.coverMax-self.paretoFrontRawCov[i-1]/self.coverMax)
                    if boundaryCalculation > 0:
                        goodDist = 1
                        badDist = 0
                    else:
                        frontIntercept =  self.calcIntercept(self.paretoFrontAcc[i], self.paretoFrontAcc[i-1], self.paretoFrontRawCov[i]/self.coverMax, self.paretoFrontRawCov[i-1]/self.coverMax, mainLineSlope)
                        badDist = self.calcDistance(frontIntercept[0],objectivePair[0],frontIntercept[1],objectiveCoverVal)
                        goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)

                else:
                    i += 1

        paretoFitness = goodDist / float(goodDist + badDist)
        return paretoFitness


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
#                     if objectiveCoverVal > self.paretoFrontRawCov[i]/self.coverMax: #Over front treated like maximum indfitness
#                         goodDist = 1
#                         badDist = 0
#                     elif objectiveCoverVal == self.paretoFrontRawCov[i]/self.coverMax: #On the boundary Line but not a point on the front. - some more minor penalty.
#                         goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#                         badDist = self.calcDistance(0,self.paretoFrontAcc[i],0,self.paretoFrontRawCov[i]/self.coverMax) - goodDist
#                     else: #Maximum penalty case - point is a boundary case underneath the front.
#                         goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#                         badDist = self.calcDistance(self.paretoFrontAcc[i],objectivePair[0],self.paretoFrontRawCov[i]/self.coverMax, objectiveCoverVal)
#
#                 if i == len(self.paretoFrontAcc)-1 and frontPointSlope < mainLineSlope: #Special Case:  High Accuracy boundary case
#                     foundPerp = True
#                     if objectivePair[0] > self.paretoFrontAcc[i]: #Over front treated like maximum indfitness
#                         goodDist = 1
#                         badDist = 0
#                     elif objectivePair[0] == self.paretoFrontAcc[i]: #On the boundary Line but not a point on the front. - some more minor penalty.
#                         goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#                         badDist = self.calcDistance(0,self.paretoFrontAcc[i],0,self.paretoFrontRawCov[i]/self.coverMax) - goodDist
#                     else: #Maximum penalty case - point is a boundary case underneath the front.
#                         goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#                         badDist = self.calcDistance(self.paretoFrontAcc[i],objectivePair[0],self.paretoFrontRawCov[i]/self.coverMax, objectiveCoverVal)
#
#                 elif frontPointSlope >= mainLineSlope: #Normal Middle front situation (we work from high point itself up to low point - but not including)
#                     frontIntercept =  self.calcIntercept(self.paretoFrontAcc[i], self.paretoFrontAcc[i-1], self.paretoFrontRawCov[i]/self.coverMax, self.paretoFrontRawCov[i-1]/self.coverMax, mainLineSlope)
#                     foundPerp = True
#                     badDist = self.calcDistance(frontIntercept[0],objectivePair[0],frontIntercept[1],objectiveCoverVal)
#                     goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#
#                 else:
#                     i += 1
#
#         frontDistanceFitness = goodDist / float(goodDist + badDist)
#         #paretoFitness = (areaFitness + frontDistanceFitness) / float(2)
#
#         paretoFitness = frontDistanceFitness
#
#         return paretoFitness




#         print 'getting pareto'
#         print 'AccFront= '+ str(self.paretoFrontAcc)
#         print 'CovFront= '+str(self.paretoFrontRawCov) #will store the actual cover values
#         print 'pointPair= '+ str(objectivePair)
#         print 'ruleAUC= '+ str(ruleAUC)
#         print 'self.AUC= '+str(self.AUC)
#         print 'paretofit= '+str(paretoFitness)


#         if paretoFitness > self.preFitMax:
#             print 'set to max'
#             print 'AccFront= '+ str(self.paretoFrontAcc)
#             print 'CovFront= '+str(self.paretoFrontRawCov) #will store the actual cover values
#             print 'pointPair= '+ str(objectivePair)
#             print 'ruleAUC= '+ str(ruleAUC)
#             print 'self.AUC= '+str(self.AUC)
#             print 'paretoFitness= '+str(paretoFitness)
#             print 'self.preFitMax= '+ str(self.preFitMax)
#             paretoFitness = self.preFitMax
#         #print paretoFitness
#
#         if paretoFitness < .001:
#             print 'pareto'
#             print 'AccFront= '+ str(self.paretoFrontAcc)
#             print 'CovFront= '+str(self.paretoFrontRawCov) #will store the actual cover values
#             print 'pointPair= '+ str(objectivePair)
#             print 'ruleAUC= '+ str(ruleAUC)
#             print 'self.AUC= '+str(self.AUC)
#             x = 5/0



#         objectiveCoverVal = objectivePair[1] / self.coverMax
#         #Special Case
#         i = -1
#         foundPerp = False
#         badDist = None
#         while not foundPerp:
#             mainLineSlope = self.calcSlope(0, objectivePair[0], 0, objectiveCoverVal)
#             if i == -1:
#                 if self.paretoFrontAcc[i] == 0: #Point is on boundary (thus we can treat this like a middle interval
#                     print 'bound'
#                     i += 1
#                 else:
#                     #print 'close bound'
#                     i += 1
#                     if self.paretoFrontAcc[i] == objectivePair[0] and self.paretoFrontRawCov[i] == objectivePair[1]:
#                         goodDist = 1
#                         badDist = 0
#                         foundPerp = True
#                     else:
#
#                         frontIntercept =  self.calcIntercept(0, self.paretoFrontAcc[i], self.paretoFrontRawCov[i]/self.coverMax, self.paretoFrontRawCov[i]/self.coverMax, mainLineSlope) #Starts on right side of curve
#                         if frontIntercept[0] <= self.paretoFrontAcc[i]:
#                             foundPerp = True
#                             #theoryMaxDist = self.paretoFrontAcc[i]+self.paretoFrontRawCov[i]/self.coverMax
#                             goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#                             badDist = self.calcDistance(self.paretoFrontAcc[i],objectivePair[0],self.paretoFrontRawCov[i]/self.coverMax, objectiveCoverVal) #Handles being on or close to front with lower accuracy
# #                             theoryMaxDist = self.calcDistance(0, self.paretoFrontAcc[i], 0, self.paretoFrontRawCov[i]/self.coverMax)
# #                             goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
# #                             badDist = theoryMaxDist - goodDist #Handles being on or close to front with lower accuracy
#
#             elif i == len(self.paretoFrontAcc)-1: #Special Case
#                 if self.paretoFrontAcc[i] == objectivePair[0] and self.paretoFrontRawCov[i] == objectivePair[1]:
#                     goodDist = 1
#                     badDist = 0
#                     foundPerp = True
#                 else:
#                     #print 'close other bound'
#                     foundPerp = True
#                     #theoryMaxDist = self.paretoFrontAcc[i]+self.paretoFrontRawCov[i]/self.coverMax
#                     goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#                     badDist = self.calcDistance(self.paretoFrontAcc[i],objectivePair[0],self.paretoFrontRawCov[i]/self.coverMax, objectiveCoverVal)
#
# #                     theoryMaxDist = self.calcDistance(self.paretoFrontAcc[i], 0, self.paretoFrontRawCov[i]/self.coverMax, 0)
# #                     goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
# #                     badDist = theoryMaxDist - goodDist #Handles being on or close to front with lower accuracy
#
#             else: # Normal middle pair
#                 #print 'middle'
#                 if self.paretoFrontAcc[i] == objectivePair[0] and self.paretoFrontRawCov[i] == objectivePair[1]:
#                     goodDist = 1
#                     badDist = 0
#                     foundPerp = True
#                 else:
#                     frontIntercept =  self.calcIntercept(self.paretoFrontAcc[i], self.paretoFrontAcc[i+1], self.paretoFrontRawCov[i]/self.coverMax, self.paretoFrontRawCov[i+1]/self.coverMax, mainLineSlope)
#                     #print frontIntercept
#                     if frontIntercept[0] >= self.paretoFrontAcc[i] and frontIntercept[0] <= self.paretoFrontAcc[i+1]:# and frontIntercept[1] <= self.paretoFrontCov[i] and frontIntercept[1] <= self.paretoFrontCov[i]:
#                         foundPerp = True
#                         badDist = self.calcDistance(frontIntercept[0],objectivePair[0],frontIntercept[1],objectiveCoverVal)
#                         goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#                     i += 1
#
#         #print badDist
#         #Calculate distance from point to minFit edge.
#         #print i
#
#         paretoFitness = goodDist / float(goodDist + badDist)
#         if self.epochID == 'ENCFront':
#             if paretoFitness > 1:
#                 paretoFitness = 1.0
#
#         if paretoFitness > 1.0:
#             print "NEW"
#             print self.classID + ' ' + self.epochID + '-------------------------------------'
#             print objectivePair
#             print objectiveCoverVal
#             print self.paretoFrontAcc
#             print self.paretoFrontRawCov
#             print i
#             print self.calcIntercept(self.paretoFrontAcc[1], self.paretoFrontAcc[2], self.paretoFrontRawCov[1]/self.coverMax, self.paretoFrontRawCov[2]/self.coverMax, mainLineSlope)
#             print frontIntercept
#             print mainLineSlope
#             print goodDist
#             print badDist
#             print goodDist+badDist
#             print paretoFitness
#         #print 'end'
#         return paretoFitness


    def calcDistance(self, y1, y2, x1, x2):
        distance = math.sqrt(math.pow(x2-x1,2) + math.pow(y2-y1,2))
        return distance


    def calcIntercept(self, y1a, y2a, x1a, x2a, mainLineSlope):
        """  Calculates the coordinates at which the two lines 'A' defined by points and 'B', defined by a point and slope, intersect """
        slopeA = self.calcSlope(y1a, y2a, x1a, x2a)
        slopeB = mainLineSlope
        if slopeA == 0:
            xIntercept = x1a
            yIntercept = slopeB*xIntercept

        else:
            xIntercept = (y2a - slopeA*x2a) / float(slopeB-slopeA)
            yIntercept = slopeB*xIntercept
        return [yIntercept,xIntercept]


#         slopeA = self.calcSlope(y1a, y2a, x1a, x2a)
#         slopeB = mainLineSlope
#         xIntercept = (slopeA*x1a - y1a) / float(slopeA - slopeB)
#         yIntercept = slopeA*(xIntercept - x1a) + y1a
#         return [yIntercept,xIntercept]


#    def calcIntercept(self, y1a, y2a, x1a, x2a, y1b, x1b):
#        """  Calculates the coordinates at which the two lines 'A' defined by points and 'B', defined by a point and slope, intersect """
#        slopeA = self.calcSlope(y1a, y2a, x1a, x2a)
#        slopeB = -1*slopeA
#        xIntercept = (slopeA*x1a - slopeB*x1b + y1b - y1a) / float(slopeA - slopeB)
#        yIntercept = slopeA*(xIntercept - x1a) + y1a
#        return [yIntercept,xIntercept]


    def calcSlope(self, y1, y2, x1, x2):
        """ Calculate slope between two points """
        if x2-x1 == 0:
            slope = 0
        else:
            slope = (y2 - y1) / (x2 - x1)
        return slope


    def pareto_frontier(self, Xs, Ys, maxX = True, maxY = True):
        """ Code obtained online: http://oco-carbon.com/metrics/find-pareto-frontiers-in-python/"""
        myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
        #print myList
        p_front = [myList[0]]
        #print p_front
        for pair in myList[1:]:
            if maxY:
                #if pair[1] >= p_front[-1][1]:
                if pair[1] > p_front[-1][1]:
                    p_front.append(pair)
            else:
                #if pair[1] <= p_front[-1][1]:
                if pair[1] < p_front[-1][1]:
                    p_front.append(pair)
        #print p_front
        p_frontX = [pair[0] for pair in p_front]
        p_frontY = [pair[1] for pair in p_front]
        return p_frontX, p_frontY







#     def checkFront(self, objectivePair):
#         """ Takes a new pair of accuracy and usefulCover values and checks to see if this pair adds to defining a better non-dominated front"""
#         #Algorithm to check for non-dominated point
#         nonDominated = False
#         self.currRightRef = 0
#         print self.paretoFrontAcc
#         print objectivePair[0]
#         while self.currRightRef < len(self.paretoFrontAcc) and objectivePair[0] <= self.paretoFrontAcc[self.currRightRef] : #if only one point, than currRightRef == 0
#             # define lower bound accuracy of current pareto front
#             self.currRightRef += 1
#         print self.currRightRef
#
#         if self.currRightRef == 0: #special case - Front boundary - accuracy (AUTO INSERT)
#             print 'acc bound'
#             if self.accuracyMax < objectivePair[0]:  #Won't be true for the first point.
#                 self.accuracyMax = objectivePair[0]
#                 nonDominated = True
#                 self.paretoFrontAcc.insert(0,objectivePair[0])
#                 self.paretoFrontRawCov.insert(0,objectivePair[1])
#                 self.paretoFrontCov.insert(0, objectivePair[1]/float(self.coverMax))
#                 #self.adjustCoverList()
#                 print 'inserted new best acc'
#
#         elif self.currRightRef > len(self.paretoFrontAcc)-1: #Special case - Front boundary -cover (AUTO INSERT)
#             print 'cov bound'
#             if self.coverMax < objectivePair[1]:
#                 self.coverMax = objectivePair[1]
#                 nonDominated = True
#                 self.paretoFrontRawCov.append(self.coverMax)
#                 self.paretoFrontCov.append(1.0)
#                 self.adjustCoverList()
#                 self.paretoFrontAcc.append(objectivePair[0])
#
#         else: #Check to add a new point in the middle
#             print 'middle'
#             # i points to lower Acc point and i-1 points to higher acc point.
#             frontSegmentSlope = self.calcSlope(self.paretoFrontAcc[self.currRightRef-1], self.paretoFrontAcc[self.currRightRef], self.paretoFrontCov[self.currRightRef-1], self.paretoFrontCov[self.currRightRef])
#             newSlope = self.calcSlope(self.paretoFrontAcc[self.currRightRef-1], objectivePair[0], self.paretoFrontCov[self.currRightRef-1], objectivePair[1])
#             #check slope to new point from
#             if newSlope >= frontSegmentSlope:
#                 nonDominated = True
#                 self.paretoFrontAcc.insert(self.currRightRef,objectivePair[0])
#                 self.paretoFrontCov.insert(self.currRightRef,objectivePair[1]/float(self.coverMax))
#                 self.paretoFrontRawCov.insert(self.currRightRef, objectivePair[1])
#                 # currRightRef now points to new point
#                 print 'inserted on front'
#         return nonDominated
#
#
#     def adjustFront(self,objectivePair):
#         """ Checks for and potentially removes dominated points from the paretoFront. """
#         #algorithm to recalculate front based on new point (and possibly remove other points
#         #Start from new point and go independently in both directions removing points that have wrong slope.
#         #From new point to right
#
#         #ReWRITE THIS PART assuming accuracy in order, just deleteany further entries that don't follow increaseing Cover trend!!!!!!!!!
#         #PRobleM - THIS DOENS'T TAKE CONVEX SHAPE PRESERVATION INTO CONSIDERATION.
#         #ALSO IN SINGLE POINT VERSION ABOVE, EROR CAN STILL OCCUR.
# #         i = 0
# #         while i < len(self.paretoFrontAcc)-1:
# #             if self.paretoFrontRawCov[i] > self.paretoFrontRawCov[i+1]:
# #                 self.paretoFrontAcc.pop(i+1)
# #                 self.paretoFrontCov.pop(i+1)
# #                 self.paretoFrontRawCov.pop(i+1)
# #             else:
# #                 i += 1
#
#         i = self.currRightRef #now is the new point ref.
#         while i < len(self.paretoFrontAcc)-2:
#             print 'made it'
#             #bigger slope is better
#             firstSlope = self.calcSlope(self.paretoFrontAcc[i], self.paretoFrontAcc[i+1], self.paretoFrontCov[i], self.paretoFrontCov[i+1])
#             secondSlope = self.calcSlope(self.paretoFrontAcc[i], self.paretoFrontAcc[i+2], self.paretoFrontCov[i], self.paretoFrontCov[i+2])
#             print firstSlope
#             print secondSlope
#             deleted = False
#             #if secondSlope >= firstSlope:
#             if secondSlope <= firstSlope: # I was thinking of slope backwards in my head from the ordering of cover, when it was by accuracy (or does this switch??
#                 #Delete point
#                 print i
#                 self.paretoFrontAcc.pop(i+1)
#                 self.paretoFrontCov.pop(i+1)
#                 self.paretoFrontRawCov.pop(i+1)
#                 deleted = True
#             if not deleted:
#                 i += 1
#
#         i = self.currRightRef #now is the new point ref.
#         while i > 1:
#             #bigger slope is better
#             firstSlope = self.calcSlope(self.paretoFrontAcc[i], self.paretoFrontAcc[i-1], self.paretoFrontCov[i], self.paretoFrontCov[i-1])
#             secondSlope = self.calcSlope(self.paretoFrontAcc[i], self.paretoFrontAcc[i-2], self.paretoFrontCov[i], self.paretoFrontCov[i-2])
#             deleted = False
#             #if firstSlope >= secondSlope:
#             if firstSlope <= secondSlope:
#                 #Delete point
#                 self.paretoFrontAcc.pop(i-1)
#                 self.paretoFrontCov.pop(i-1)
#                 self.paretoFrontRawCov.pop(i-1)
#                 deleted = True
#                 self.currRightRef -= 1
#                 i -= 1
#             if not deleted:
#                 i -= 1
#         if i == 1:
#             if self.paretoFrontAcc[i] >= self.paretoFrontAcc[i-1]: #remove point
#                 self.paretoFrontAcc.pop(i-1)
#                 self.paretoFrontCov.pop(i-1)
#                 self.paretoFrontRawCov.pop(i-1)




#
#     def adjustCoverList(self):
#         for i in range(len(self.paretoFrontRawCov)-1):
#             self.paretoFrontCov[i] = self.paretoFrontRawCov[i] / float(self.coverMax)



#    def calcIntercept(self, y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b):
#        """  Calculates the coordinates at which the two lines 'A' and 'B' intersect """
#        slopeA = self.calcSlope(y1a, y2a, x1a, x2a)
#        slopeB = self.calcSlope(y1b, y2b, x1b, x2b)
#        xIntercept = (slopeA*x1a - slopeB*x1b + y1b - y1a) / float(slopeA - slopeB)
#        yIntercept = slopeA*(xIntercept - x1a) + y1a
#        return [yIntercept,xIntercept]

#    def updateFront(self, objectivePair,currentCoverMax):
#        """  Handles process of checking and adjusting the fitness pareto front. """
#        #Update any changes to the maximum Cov
#        if currentCoverMax > self.paretoFrontCov[len(self.paretoFrontCov)-1]:
#            self.paretoFrontCov[len(self.paretoFrontCov)-1] = currentCoverMax
#
#        if self.checkFront(objectivePair):
#            self.adjustFront(objectivePair)
#            print self.paretoFrontAcc
#            print self.paretoFrontCov
#
#
#    def checkFront(self, objectivePair):
#        """ Takes a new pair of accuracy and usefulCover values and checks to see if this pair adds to defining a better non-dominated front"""
#        #Algorithm to check for non-dominated point
#        nonDominated = False
#        self.currRightRef = 0
#        while objectivePair[0] <= self.paretoFrontAcc[self.currRightRef]:
#            # define lower bound accuracy of current pareto front
#            self.currRightRef += 1
#        # I points to lower Acc point and i-1 points to higher acc point.
#        frontSegmentSlope = self.calcSlope(self.paretoFrontAcc[self.currRightRef-1], self.paretoFrontAcc[self.currRightRef], self.paretoFrontCov[self.currRightRef-1], self.paretoFrontCov[self.currRightRef])
#        newSlope = self.calcSlope(self.paretoFrontAcc[self.currRightRef-1], objectivePair[0], self.paretoFrontCov[self.currRightRef-1], objectivePair[1])
#        #check slope to new point from
#        if newSlope >= frontSegmentSlope:
#            nonDominated = True
#            self.paretoFrontAcc.insert(self.currRightRef,objectivePair[0])
#            self.paretoFrontCov.insert(self.currRightRef,objectivePair[1])
#            # currRightRef now points to new point
#            print 'inserted on front'
#        return nonDominated
#
#
#    def calcSlope(self, y1, y2, x1, x2):
#        """ Calculate slope between two points """
##        print y1
##        print y2
##        print x1
##        print x2
#        if x2-x1 == 0:
#            slope = 0
#        else:
#            slope = (y2 - y1) / (x2 - x1)
#        return slope


#    def adjustFront(self,objectivePair):
#        """ Checks for and potentially removes dominated points from the paretoFront. """
#        #algorithm to recalculate front based on new point (and possibly remove other points
#        #Start from new point and go independently in both directions removing points that have wrong slope.
#        #From new point to right
#        i = self.currRightRef #now is the new point ref.
#        while i < len(self.paretoFrontAcc)-2:
#            #bigger slope is better
#            firstSlope = self.calcSlope(self.paretoFrontAcc[i], self.paretoFrontAcc[i+1], self.paretoFrontCov[i], self.paretoFrontCov[i+1])
#            secondSlope = self.calcSlope(self.paretoFrontAcc[i], self.paretoFrontAcc[i+2], self.paretoFrontCov[i], self.paretoFrontCov[i+2])
#            deleted = False
#            if secondSlope >= firstSlope:
#                #Delete point
#                self.paretoFrontAcc.pop[i+1]
#                self.paretoFrontCov.pop[i+1]
#                deleted = True
#            if not deleted:
#                i += 1
#
#        #From new point to left
#        i = self.currRightRef #now is the new point ref.
#        while i > 1:
#            #bigger slope is better
#            firstSlope = self.calcSlope(self.paretoFrontAcc[i], self.paretoFrontAcc[i-1], self.paretoFrontCov[i], self.paretoFrontCov[i-1])
#            secondSlope = self.calcSlope(self.paretoFrontAcc[i], self.paretoFrontAcc[i-2], self.paretoFrontCov[i], self.paretoFrontCov[i-2])
#            deleted = False
#            if firstSlope >= secondSlope:
#                #Delete point
#                self.paretoFrontAcc.pop[i-1]
#                self.paretoFrontCov.pop[i-1]
#                deleted = True
#                self.currRightRef -= 1
#                i -= 1
#            if not deleted:
#                i -= 1
#
#
#    def getParetoFitness(self, objectivePair):
#        """ Determines and returns the pareto fitness based on the proportional distance of a given point"""
#        # Find the relative distance from the front.  Returns 1 (i.e. perfect fitness, if on the front)
#
#        #Calculate distance from point to paretoFront
#        #Step through front intervals and check to see if interval contains intersect if so grab that distance.
#        print 'getting pareto'
#        print objectivePair
#        i = 0
#        foundPerp = False
#        badDist = None
#        while not foundPerp:
#            frontIntercept =  self.calcIntercept(self.paretoFrontAcc[i], self.paretoFrontAcc[i+1], self.paretoFrontCov[i], self.paretoFrontCov[i+1], objectivePair[0], objectivePair[1])
#            if frontIntercept[0] >= self.paretoFrontAcc[i] and frontIntercept[0] <= self.paretoFrontAcc[i+1]:
#                foundPerp = True
#                badDist = self.calcDistance(frontIntercept[0],objectivePair[0],frontIntercept[1],objectivePair[1])
#                print badDist
#            i += 1
#        #Calculate distance from point to minFit edge.
#        lastPoint = len(self.paretoFrontAcc) - 1
#        minIntercept =  self.calcIntercept(self.paretoFrontAcc[0], self.paretoFrontAcc[lastPoint], self.paretoFrontCov[0], self.paretoFrontCov[lastPoint], objectivePair[0], objectivePair[1])
#        goodDist = self.calcDistance(minIntercept[0],objectivePair[0],minIntercept[1],objectivePair[1])
#
#        paretoFitness = goodDist / float(goodDist + badDist)
#        print 'end'
#        return paretoFitness
#
#
#    def calcDistance(self, y1, y2, x1, x2):
#        distance = math.sqrt(math.pow(x2-x1,2) + math.pow(y2-y1,2))
#        return distance
#
#
#    def calcIntercept(self, y1a, y2a, x1a, x2a, y1b, x1b):
#        """  Calculates the coordinates at which the two lines 'A' defined by points and 'B', defined by a point and slope, intersect """
#        slopeA = self.calcSlope(y1a, y2a, x1a, x2a)
#        slopeB = -1*slopeA
#        xIntercept = (slopeA*x1a - slopeB*x1b + y1b - y1a) / float(slopeA - slopeB)
#        yIntercept = slopeA*(xIntercept - x1a) + y1a
#        return [yIntercept,xIntercept]



#     def updateFront(self, objectivePair):
#         """  Handles process of checking and adjusting the fitness pareto front. """
#         #Update any changes to the maximum Cov - automatically adds point if new cov max is found
#         #objectivePair[0] == accuracy
#         #objectivePair[1] == cover
#         if objectivePair[1] > self.coverMax:  #Adds a new Cover Max Point Only (will also cover the first point entry)
#             self.coverMax = objectivePair[1]
#             self.paretoFrontRawCov.append(self.coverMax)
#             self.paretoFrontCov.append(1.0)
#             self.adjustCoverList()
#             self.paretoFrontAcc.append(objectivePair[0])
#             if objectivePair[0] > self.accuracyMax: #Should be true only for the first point.
#                 self.accuracyMax = objectivePair[0]
#
#         if self.checkFront(objectivePair):
#             self.adjustFront(objectivePair)
#             print self.paretoFrontAcc
#             print self.paretoFrontRawCov
#             print self.paretoFrontCov
#
#
#     def adjustCoverList(self):
#         for i in range(len(self.paretoFrontRawCov)-1):
#             self.paretoFrontCov[i] = self.paretoFrontRawCov[i] / float(self.coverMax)
#
#
#     def checkFront(self, objectivePair):
#         """ Takes a new pair of accuracy and usefulCover values and checks to see if this pair adds to defining a better non-dominated front"""
#         #Algorithm to check for non-dominated point
#         nonDominated = False
#         self.currRightRef = 0
#         while objectivePair[0] <= self.paretoFrontAcc[self.currRightRef] and self.currRightRef < len(self.paretoFrontAcc) - 1: #if only one point, than currRightRef == 0
#             # define lower bound accuracy of current pareto front
#             self.currRightRef += 1
#
#         if self.currRightRef == 0: #special case - Front boundary - accuracy (AUTO INSERT)
#             if self.accuracyMax < objectivePair[0]:  #Won't be true for the first point.
#                 self.accuracyMax = objectivePair[0]
#                 nonDominated = True
#                 self.paretoFrontAcc.insert(self.currRightRef,objectivePair[0])
#                 self.paretoFrontRawCov.insert(self.currRightRef,objectivePair[1])
#                 self.paretoFrontCov.insert(self.currRightRef, objectivePair[1]/float(self.coverMax))
#                 #self.adjustCoverList()
#                 print 'inserted new best acc'
#
#         elif self.currRightRef > len(self.paretoFrontAcc)-1: #Special case - Front boundary -cover (AUTO INSERT)
#             if self.coverMax < objectivePair[1]:
#                 self.coverMax = objectivePair[1]
#                 nonDominated = True
#                 self.paretoFrontRawCov.append(self.coverMax)
#                 self.paretoFrontCov.append(1.0)
#                 #self.adjustCoverList()
#                 self.paretoFrontAcc.append(objectivePair[0])
#
#         else: #Check to add a new point in the middle
#             # i points to lower Acc point and i-1 points to higher acc point.
#             frontSegmentSlope = self.calcSlope(self.paretoFrontAcc[self.currRightRef-1], self.paretoFrontAcc[self.currRightRef], self.paretoFrontCov[self.currRightRef-1], self.paretoFrontCov[self.currRightRef])
#             newSlope = self.calcSlope(self.paretoFrontAcc[self.currRightRef-1], objectivePair[0], self.paretoFrontCov[self.currRightRef-1], objectivePair[1])
#             #check slope to new point from
#             if newSlope >= frontSegmentSlope:
#                 nonDominated = True
#                 self.paretoFrontAcc.insert(self.currRightRef,objectivePair[0])
#                 self.paretoFrontCov.insert(self.currRightRef,objectivePair[1])
#                 # currRightRef now points to new point
#                 print 'inserted on front'
#         return nonDominated
