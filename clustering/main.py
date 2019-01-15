import sys
import KMeans

def main(argv):
    clusteringK = 5
    x, y = extractData("data/train.csv")
    clustering = createClustering(clusteringK, x, y)

    oldSSE = -1
    SSE    = -1
    percentageReduction = 100000;
    i = 0
    while i==0 or checkConvergence(oldSSE, SSE):
        i = i + 1
        oldSSE = SSE
        clustering.iterate()
        clustering.plotClusters()
        SSE = clustering.computeSSE()
        printIterationInformation(i, oldSSE, SSE)
        percentageReduction = computePercentageReduction(oldSSE, SSE)

def checkConvergence(oldCost, newCost):
    tolerance = 0.0001
    perc = computePercentageReduction(oldCost, newCost)
    return perc>tolerance

def printIterationInformation(i, oldCost, newCost):
    percentageReduction = 0.00
    if not(isFirstIteration(i)):
        percentageReduction = computePercentageReduction(oldCost, newCost)
    string = "Iteration: {} | Reduction (%): {}".format(i, percentageReduction)
    print(string)

def isFirstIteration(i):
    return i==1

def computePercentageReduction(oldCost, newCost):
    absDiff = abs(newCost - oldCost)
    return abs(absDiff/oldCost)

def createClustering(k, x, y):
    name = "First Attempt of Clustering"
    return KMeans.KMeans(k, name, x, y)

def extractData(filepath):
    (X, Y) = initializeXYEmpties()
    try:
        (X, Y) = scanFileToGatherData(filepath)
    except IOError:
        msg = "[Error] Cannot read file {}". format(filepath)
        printErrorAndExit(msg)
    return X, Y

def scanFileToGatherData(filepath):
    (X, Y) = initializeXYEmpties()
    with open(filepath) as input:
        for i, line in enumerate(input.      readlines()):
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
