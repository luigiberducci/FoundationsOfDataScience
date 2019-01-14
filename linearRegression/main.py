import sys
import numpy as np
import SimpleLR
import matplotlib.pyplot as plt

def main(argv):
    (trainX, trainY) = extractData("data/train.csv")
    model = trainModel(trainX, trainY)
    model.summary()

    (testX, testY) = extractData("data/test.csv")
    predictions = model.predictAll(testX)

    plotPredictionsComparison(testX, predictions, testY)
    plotErrors(testX, predictions, testY)

def plotErrors(x, yPredicted, yTruthValue):
    errors = computeErrors(yPredicted, yTruthValue)
    plotPoints(x, errors, 'red')
    plt.title("Error distribution")
    plt.show()

def computeErrors(predictions, truthValue):
    return np.subtract(truthValue, predictions)

def plotPredictionsComparison(x, yPredicted, yTruthValue):
    plotPoints(x, yPredicted, 'blue')
    plotPoints(x, yTruthValue, 'green')
    plt.title("Predictions (blue) VS Truth-Values (green)")
    plt.show()

def plotPoints(x, y, colorname):
    plt.plot(x, y, 'ro', color=colorname)

def plotLines(x, y, colorname):
    plt.plot(x, y, color=colorname)

def trainModel(X, Y):
    model = SimpleLR.SimpleLR("Simple Linear Regression", X, Y);
    model.train()
    return model

def extractData(filepath):
    (X, Y) = initializeXYEmpties()
    try:
        (X, Y) = scanFileToGatherData(filepath)
    except IOError:
        msg = "[Error] Cannot read file {}".format(filepath)
        printErrorAndExit(msg)
    return X, Y

def scanFileToGatherData(filepath):
    (X, Y) = initializeXYEmpties()
    with open(filepath) as input:
        for i, line in enumerate(input.readlines()):
            if isFirstLine(i):
                continue
            try:
                (x, y) = extractDataFromLine(line)
                X.append(x)
                Y.append(y)
            except:
                pass
    return X, Y

def isFirstLine(i):
    return i==0

def extractDataFromLine(line):
    newLine= cleanLine(line)
    (x, y) = newLine.split(",")
    return float(x), float(y)

def cleanLine(line):
    return line.replace("\n", "")

def initializeXYEmpties():
    return ([], [])

def printErrorAndExit(msg):
    print(msg)
    exitWithError()

def exitWithError():
    ERROR_CODE = -1
    exit(ERROR_CODE)

if __name__=="__main__":
    main(sys.argv)
