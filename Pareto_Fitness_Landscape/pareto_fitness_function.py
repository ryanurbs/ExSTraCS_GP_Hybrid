#from __future__ import print_function
"""
Name:        pareto_fitness_function.py
Authors:     Ryan Urbanowicz - Written at Upenn, Philadelphia, PA, USA
Contact:     ryanurb@upenn.edu
Created:     Dec 17, 2015
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
#from Pareto_Fitness_Landscape.plot_fitness_landscape import fitness_scores


def main():   
    #Take the front values and using them to construct a fitness landscape using the function below, and then place the target points on that landscape
    #As a holding place for this task, I will take the target values and output their fitness scores
    #
    fitnessList = []
    for i in range(len(targetPointAcc)): # For each target point
        objectivePair = [targetPointAcc[i], targetPointRawCov[i]]
        fitnessList.append(getParetoFitness(objectivePair))
        
    print("Rule\tAccuracy\tCover\tRawCover\tFitness")
    for i in range(len(targetPointAcc)): # For each target point
        print(str(i)+'\t'+str(targetPointAcc[i])+'\t'+str(targetPointRawCov[i]/coverMax)+'\t'+str(targetPointRawCov[i])+'\t'+str(fitnessList[i]))
          
          
    print('Gathering fitness scores...')
    fitness_scores = calculateFitnessLandscape()
    print('Plotting the fitness landscape...')
    plotFitnessLandscape(fitness_scores)
    
    
def plotFitnessLandscape(fitness_scores):
    #Visualization goals:
    #1. Show the fitness landscape domain (x/y axes each from 0 to 1)
    #2. Show connected points that make up the pareto front
    #3. Based on the specified granularity and a scale of fitness magnitude also from 0 to 1 given as a color or greyscale 2D landscape within the x/y axes 0-1.
    #4. Plot with small black dots or x's the 'target points'  so the the user can see where a specific rule with fall within the fitness landscape.
    
    plt.figure(figsize=(10, 10))
    plt.imshow(fitness_scores, interpolation='nearest', cmap='viridis')
    plt.ylim(0, 1.1/granularity)
    plt.yticks([0, 1./(2*granularity), 1./granularity], [0, 0.5, 1], fontsize=12)
    plt.xticks([0, 1./(2*granularity), 1./granularity], [0, 0.5, 1], fontsize=12)
    plt.colorbar(shrink=0.8)
    plt.xlabel('Normalized Coverage', fontsize=14)
    plt.ylabel('Useful Accuracy', fontsize=14)
    
    plt.plot(np.array(paretoFrontRawCov) / coverMax * 1./granularity,
             np.array(paretoFrontAcc) * 1./granularity,
             'o-', ms=10, lw=2, color='black')
    
    plt.plot(np.array(targetPointRawCov) / coverMax * 1./granularity,
         np.array(targetPointAcc) * 1./granularity,
         'x', ms=10, lw=2, color='black')

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

    
def calculateFitnessLandscape():
    """  """
    fitness_scores = []
#     for accuracy in np.arange(0, 1.0, granularity):
#         for coverage in np.arange(0, 1.0, granularity):
    for accuracy in np.arange(0, 1.1, granularity):
        for coverage in np.arange(0, 1.1, granularity):
            fitness_scores.append(getParetoFitness([accuracy, coverage * coverMax]))
    
    fitness_scores = np.array(fitness_scores)
    #fitness_scores.shape = (int(1. / granularity), int(1. / granularity))
    fitness_scores.shape = (int(1.1 / granularity), int(1.1 / granularity))
    return fitness_scores
    
    
def getParetoFitness(objectivePair):
    """ Determines and returns the pareto fitness based on the proportional distance of a given point"""
    objectiveCoverVal = objectivePair[1] / coverMax
    #Special Case
    i = 0#-1
    foundPerp = False
    badDist = None
    while not foundPerp:
        #print i
        mainLineSlope = calcSlope(0, objectivePair[0], 0, objectiveCoverVal)   #check that we are alwasys storing objective cover val, so there is no error here (recent thought)
        if paretoFrontAcc[i] == objectivePair[0] and paretoFrontRawCov[i] == objectivePair[1]: #is point a point on front?
            #print "POINT ON FRONT"
            goodDist = 1
            badDist = 0
            foundPerp = True
        else:
            frontPointSlope = calcSlope(0, paretoFrontAcc[i], 0, paretoFrontRawCov[i]/coverMax) 
            if i == 0 and frontPointSlope >= mainLineSlope: #Special Case:  High Coverage boundary case
                foundPerp = True
                if objectiveCoverVal >= paretoFrontRawCov[i]/coverMax: #Over front treated like maximum indfitness
                    if objectivePair[0] >= paretoFrontAcc[i]:
                        goodDist = 1
                        badDist = 0
                    else:
#                         goodDist = calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#                         badDist = calcDistance(0,paretoFrontAcc[i],0,paretoFrontRawCov[i]/coverMax) - goodDist
                        goodDist = calcDistance(0,objectivePair[0],0,1)
                        badDist = calcDistance(0,paretoFrontAcc[i],0,1) - goodDist
#                         goodDist = objectivePair[0]
#                         badDist = paretoFrontAcc[i]-objectivePair[0]
#                         goodDist = calcDistance(0,objectivePair[0],paretoFrontRawCov[i]/coverMax,1)
#                         badDist = calcDistance(paretoFrontAcc[i],objectivePair[0],paretoFrontRawCov[i]/coverMax,1)
                elif objectiveCoverVal == paretoFrontRawCov[i]/coverMax: #On the boundary Line but not a point on the front. - some more minor penalty.
                    goodDist = calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                    badDist = calcDistance(0,paretoFrontAcc[i],0,paretoFrontRawCov[i]/coverMax) - goodDist
                else: #Maximum penalty case - point is a boundary case underneath the front.
                    goodDist = calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                    badDist = calcDistance(paretoFrontAcc[i],objectivePair[0],paretoFrontRawCov[i]/coverMax, objectiveCoverVal)
                    goodDist += objectivePair[0]
                    badDist += paretoFrontAcc[i]-objectivePair[0]
                    
            elif i == len(paretoFrontAcc)-1 and frontPointSlope < mainLineSlope: #Special Case:  High Accuracy boundary case
                foundPerp = True
                if objectivePair[0] > paretoFrontAcc[i]: #Over front treated like maximum indfitness
                    goodDist = 1
                    badDist = 0
#                     if objectiveCoverVal >= paretoFrontRawCov[i]/coverMax:
#                         goodDist = 1
#                         badDist = 0
#                     else:
#                         goodDist = calcDistance(0,1,0,objectiveCoverVal)
#                         badDist = calcDistance(0,1,0,paretoFrontRawCov[i]/coverMax) - goodDist
#                         goodDist = objectiveCoverVal
#                         badDist = paretoFrontRawCov[i]/coverMax-objectiveCoverVal        
                                  
                elif objectivePair[0] == paretoFrontAcc[i]: #On the boundary Line but not a point on the front. - some more minor penalty.
                    if paretoFrontAcc[i] == 1.0:
                        goodDist = 1
                        badDist = 0
                    else:
                        goodDist = calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                        badDist = calcDistance(0,paretoFrontAcc[i],0,paretoFrontRawCov[i]/coverMax) - goodDist
                else: #Maximum penalty case - point is a boundary case underneath the front.
                    goodDist = calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                    badDist = calcDistance(paretoFrontAcc[i],objectivePair[0],paretoFrontRawCov[i]/coverMax, objectiveCoverVal)
                    goodDist += objectiveCoverVal
                    badDist += paretoFrontRawCov[i]/coverMax-objectiveCoverVal   
                    
            elif frontPointSlope >= mainLineSlope: #Normal Middle front situation (we work from high point itself up to low point - but not including)
                foundPerp = True
                #boundaryCalculation Rule = (x1-x0)(y2-y0)-(x2-x0)(y1-y0), where 0 and 1 represent two points with a line, and 2 represents some other point.
                boundaryCalculation = (paretoFrontAcc[i]-paretoFrontAcc[i-1])*(objectiveCoverVal-paretoFrontRawCov[i-1]/coverMax) - (objectivePair[0]-paretoFrontAcc[i-1])*(paretoFrontRawCov[i]/coverMax-paretoFrontRawCov[i-1]/coverMax)
                if boundaryCalculation > 0:
                    goodDist = 1
                    badDist = 0
                else:
                    frontIntercept =  calcIntercept(paretoFrontAcc[i], paretoFrontAcc[i-1], paretoFrontRawCov[i]/coverMax, paretoFrontRawCov[i-1]/coverMax, mainLineSlope)
                    badDist = calcDistance(frontIntercept[0],objectivePair[0],frontIntercept[1],objectiveCoverVal)
                    goodDist = calcDistance(0,objectivePair[0],0,objectiveCoverVal)

            else:
                i += 1
        
    paretoFitness = goodDist / float(goodDist + badDist)
    return paretoFitness  


def calcDistance(y1, y2, x1, x2):
    distance = math.sqrt(math.pow(x2-x1,2) + math.pow(y2-y1,2))
    return distance
        
        
def calcIntercept(y1a, y2a, x1a, x2a, mainLineSlope):
    """  Calculates the coordinates at which the two lines 'A' defined by points and 'B', defined by a point and slope, intersect """
    slopeA = calcSlope(y1a, y2a, x1a, x2a)
    slopeB = mainLineSlope
    if slopeA == 0:
        xIntercept = x1a
        yIntercept = slopeB*xIntercept
        
    else:
        xIntercept = (y2a - slopeA*x2a) / float(slopeB-slopeA)
        yIntercept = slopeB*xIntercept
    return [yIntercept,xIntercept]


def calcSlope(y1, y2, x1, x2):
    """ Calculate slope between two points """
    if x2-x1 == 0:
        slope = 0
    else:
        slope = (y2 - y1) / (x2 - x1)
    return slope
    
         
if __name__=="__main__":
    
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    
    granularity = 0.001 #Fitness landscape granularity
    paretoFrontAcc = [0.409368404876,0.692582281546,0.902030447688,1.0]  
    paretoFrontRawCov = [210.452,204.723,180.634,165.33]
    coverMax = paretoFrontRawCov[0]
    
    #Points to specifically mark on fitness landscape heatmap.
    targetPointAcc = [0.198883, 0.6]
    targetPointRawCov = [199.473, 170.4] 
        
    main()