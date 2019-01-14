import random
import matplotlib.pyplot as plt
import numpy as np

class KMeans:
    def __init__(self, k, name, xData=[], yData=[]):
        self.K = k
        self.name = name
        if not(self.checkXYValidity(xData, yData)):
            xData, yData = [], []
        self.loadData(xData, yData)

    def loadData(self, x, y):
        self.setXData(x)
        self.setYData(y)
        self.initializeStructures()

    def initializeStructures(self):
        self.centroids = self.getEmptyCentroidCoordinates()
        self.distances = self.getEmptyDistanceStructure()
        self.assignCentroid = self.getEmptyCentroidAssignmentStructure()
        self.iterationNumber = 0

    def getEmptyCentroidCoordinates(self):
        centroids = []
        for i in range(0, self.K):
            emptyTuple = (-1, -1)
            centroids.append(emptyTuple)
        return centroids

    def getEmptyDistanceStructure(self):
        distances = []
        for i in range(0, len(self.X)):
            kTuple = []
            for i in range(0, self.K):
                kTuple.append(-1)
            distances.append(kTuple)
        return distances

    def getEmptyCentroidAssignmentStructure(self):
        assignments = []
        for i in range(0, len(self.X)):
            assignments.append(-1)
        return assignments

    def setXData(self, x):
        self.X = x

    def setYData(self, y):
        self.Y = y

    def checkXYValidity(self, x, y):
        return len(x)==len(y)

    def initializeCentroids(self):
        self.centroids = self.pickKRandomCentroids()

    def pickKRandomCentroids(self):
        centroidPairs = []
        idCentroids = random.sample(range(0, len(self.X)), self.K)
        for idC in idCentroids:
            x = self.X[idC]
            y = self.Y[idC]
            centroidPairs.append( (x, y) )
        return centroidPairs

    def computeDistances(self):
        for i, (x, y) in enumerate(zip(self.X, self.Y)):
            for c, (xc, yc) in enumerate(self.centroids):
                d = self.euclideanDistance(x, y, xc, yc)
                self.distances[i][c] = d

    def updateCentroids(self):
        for c in range(0, self.K):
            clusterX, clusterY = self.getClusterDataXY(c)
            x, y = self.getMeanPoint(clusterX, clusterY)
            self.centroids[c] = (x, y)

    def getMeanPoint(self, xPoints, yPoints):
        x = np.mean(xPoints)
        y = np.mean(yPoints)
        return x, y

    def getClusterDataXY(self, c):
        clusterID = self.getClusterElementsID(c)
        xCluster = [self.X[i] for i in clusterID]
        yCluster = [self.Y[i] for i in clusterID]
        return xCluster, yCluster

    def getClusterElementsID(self, c):
        cluster = []
        for i in range(0, len(self.X)):
            if self.assignCentroid[i] == c:
                cluster.append(i)
        return cluster

    def assignCentroidToEveryPoint(self):
        for i in range(0, len(self.X)):
            idCentroid = np.argmin([self.distances[i][c] for c in range(0, self.K)])
            self.assignCentroid[i] = idCentroid

    def iterate(self):
        if not(self.isConverged()):
            if self.iterationNumber == 0:
                self.initializeCentroids()
            self.computeDistances()
            self.assignCentroidToEveryPoint()
            self.updateCentroids()
            self.iterationNumber = self.iterationNumber + 1

    def getClusters(self):
        allClusters = []
        for c in range(0, self.K):
            cluster = self.getClusterByID(c)
            allClusters.append( cluster )
        return allClusters

    def getClusterByID(self, c):
        cluster = []
        for i, (x, y) in enumerate(zip(self.X, self.Y)):
            if self.assignCentroid[i]==c:
                cluster.append( (x,y) )
        return cluster

    def isConverged(self):
        return False    #TODO

    def euclideanDistance(self, xa, ya, xb, yb):
        dx = xa - xb
        dy = ya - yb
        return np.sqrt(dx**2 + dy**2)

    def computeSSE(self):
        SSE = 0
        for i, (x, y) in enumerate(zip(self.X, self.Y)):
            distance = self.getDistanceByCentroid(i)
            SSE = SSE + distance**2
        return SSE

    def getDistanceByCentroid(self, i):
        (x, y) = self.X[i], self.Y[i]
        (xc, yc) = self.getCentroidOfPointByItsID(i)
        distance = self.euclideanDistance(x, y, xc, yc)
        return distance

    def getCentroidOfPointByItsID(self, i):
        centroidID = self.assignCentroid[i]
        return self.centroids[centroidID]

    def plotClusters(self):
        clusters = self.getClusters()
        availableColors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'black'] #Then max number of clusters is 7
        plotTitle = "Clusters after {} iterations".format(self.iterationNumber)
        for c, cluster in enumerate(clusters):
            x = [ x for x, y in cluster ]
            y = [ y for x, y in cluster ]
            colorIndex = c%len(availableColors)
            plt.plot(x, y, 'ro', color=availableColors[colorIndex])
        plt.title(plotTitle)
        plt.show()
