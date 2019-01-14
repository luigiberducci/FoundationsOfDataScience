import numpy as np
import statistics as stat
import matplotlib.pyplot as plt

class SimpleLR:
    #Attributes
    # modelName
    # X
    # Y
    # m
    # q

    #Methods
    def __init__(self, name, X=[], Y=[]):
        self.setModelName(name)
        if len(X)>0 and len(Y)>0:
            self.setTrainingData(X, Y)
        else:
            self.X = []
            self.Y = []
        self.m = None
        self.q = None

    def predictAll(self, X):
        predictions = []
        for i,x in enumerate(X):
            y = self.predict(x)
            if not(y==None):
                predictions.append(y)
        return predictions

    def predict(self, x):
        answer = None
        if self.isTrained():
            answer = self.m * x + self.q
        return answer

    def train(self):
        print(len(self.X))
        (meanX, meanY) = self.getXYmeans()
        SSxy = self.getXYCrossDeviations()
        SSxx = self.getSumXDeviation()
        self.m = SSxy/SSxx
        self.q = meanY - self.m*meanX

    def isTrained(self):
        answer = False
        if not(self.m == None) and not(self.q == None):
            answer = True
        return answer

    def setModelName(self, name):
        self.modelName = name

    def setTrainingData(self, X, Y):
        self.setX(X)
        self.setY(Y)

    def setX(self, X):
        self.X = X

    def setY(self, Y):
        self.Y = Y

    def getSumXDeviation(self):
        x = self.X
        (meanX, meanY) = self.getXYmeans()
        return sum( (x[i] - meanX)**2 for i in range(0, len(x)) )

    def getXYmeans(self):
        meanX = stat.mean(self.X)
        meanY = stat.mean(self.Y)
        return (meanX, meanY)

    def getXYCrossDeviations(self):
        x = self.X
        y = self.Y
        (meanX, meanY) = self.getXYmeans()
        return sum( (x[i] - meanX)*(y[i] - meanY) for i in range(0, len(x)))

    def summary(self):
        content =  "Linear Regression Model\n"
        content += "\tName:\t{}\n".format(self.modelName)
        content += "\tX-data:\t{}\n".format(len(self.X))
        content += "\tY-data:\t{}\n".format(len(self.Y))
        if self.isTrained():
            content += "Learning: y = {}x + {}".format(self.m, self.q)
        print(content);

    def plot(self):
        self.plotData()
        if self.isTrained():
            self.plotLearning()
        plt.show()

    def plotData(self):
        plt.plot(self.X, self.Y, 'ro')

    def plotLearning(self):
        (minX, maxX) = (min(self.X), max(self.X))
        minY = self.m * minX + self.q
        maxY = self.m * maxX + self.q
        plt.plot([minX, maxX], [minY, maxY])
