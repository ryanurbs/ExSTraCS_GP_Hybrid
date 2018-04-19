
import math
import numpy as np
import matplotlib.pyplot as plt

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
                if objectiveCoverVal > paretoFrontRawCov[i]/coverMax: #Over front treated like maximum indfitness
                    goodDist = 1
                    badDist = 0
                elif objectiveCoverVal == paretoFrontRawCov[i]/coverMax: #On the boundary Line but not a point on the front. - some more minor penalty.
                    goodDist = calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                    badDist = calcDistance(0,paretoFrontAcc[i],0,paretoFrontRawCov[i]/coverMax) - goodDist
                else: #Maximum penalty case - point is a boundary case underneath the front.
                    goodDist = calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                    badDist = calcDistance(paretoFrontAcc[i],objectivePair[0],paretoFrontRawCov[i]/coverMax, objectiveCoverVal)
                    
            if i == len(paretoFrontAcc)-1 and frontPointSlope < mainLineSlope: #Special Case:  High Accuracy boundary case
                foundPerp = True
                if objectivePair[0] > paretoFrontAcc[i]: #Over front treated like maximum indfitness
                    goodDist = 1
                    badDist = 0
                elif objectivePair[0] == paretoFrontAcc[i]: #On the boundary Line but not a point on the front. - some more minor penalty.
                    goodDist = calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                    badDist = calcDistance(0,paretoFrontAcc[i],0,paretoFrontRawCov[i]/coverMax) - goodDist
                else: #Maximum penalty case - point is a boundary case underneath the front.
                    goodDist = calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                    badDist = calcDistance(paretoFrontAcc[i],objectivePair[0],paretoFrontRawCov[i]/coverMax, objectiveCoverVal)
                    
            elif frontPointSlope >= mainLineSlope: #Normal Middle front situation (we work from high point itself up to low point - but not including)
                frontIntercept =  calcIntercept(paretoFrontAcc[i], paretoFrontAcc[i-1], paretoFrontRawCov[i]/coverMax, paretoFrontRawCov[i-1]/coverMax, mainLineSlope)
                foundPerp = True
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
    
granularity = 0.001 #Fitness landscape granularity
    
paretoFrontAcc = [0.409368404876,0.692582281546,0.902030447688,1.0]  
paretoFrontRawCov = [210.452,204.723,180.634,165.33]
coverMax = paretoFrontRawCov[0]

print('Gathering fitness scores...')

fitness_scores = []
for accuracy in np.arange(0, 1.0, granularity):
    for coverage in np.arange(0, 1.0, granularity):
        fitness_scores.append(getParetoFitness([accuracy, coverage * coverMax]))

fitness_scores = np.array(fitness_scores)
fitness_scores.shape = (int(1. / granularity), int(1. / granularity))

print('Plotting the fitness landscape...')

plt.figure(figsize=(10, 10))
plt.imshow(fitness_scores, interpolation='nearest', cmap='viridis')
plt.ylim(0, 1000)
plt.yticks([0, 500, 1000], [0, 0.5, 1], fontsize=12)
plt.xticks([0, 500, 1000], [0, 0.5, 1], fontsize=12)
plt.colorbar(shrink=0.8)
plt.xlabel('Coverage', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)

plt.plot(np.array(paretoFrontRawCov) / coverMax * 1000.,
         np.array(paretoFrontAcc) * 1000.,
         'o-', ms=10, lw=2, color='black')

plt.plot([0, paretoFrontRawCov[0] / coverMax * 1000.],
         [0, paretoFrontAcc[0] * 1000.],
         '--', lw=2,
         color='black')

plt.plot([0, paretoFrontRawCov[-1] / coverMax * 1000.],
         [0, paretoFrontAcc[-1] * 1000.],
         '--', lw=2,
         color='black')

plt.show()