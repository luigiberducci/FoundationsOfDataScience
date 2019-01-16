import NaiveBayes
import sys
import os

def main(argv):
    testNaiveBayesOnDiabetesDataset()

def testNaiveBayesOnDiabetesDataset():
    observation = formatObservation("1,85,66,29,0,26.6,0.351,31")
    classifier = getTrainedClassifierOnDiabetesDataset()
    pred, prob = classifier.predict(observation, "Outcome")
    printPredictionResult(pred, prob)

def getTrainedClassifierOnDiabetesDataset():
    data = getDataFromDiabetesDataset()
    classifier = createNaiveBayesClassifier(data)
    classifier.train()
    return classifier

def getDataFromDiabetesDataset():
    diabetesDataset = os.path.join("data", "diabetes.csv")
    return extractDataFromFile(diabetesDataset)

def testNaiveBayesOnTennisDataset():
    observation = formatObservation("Sunny,Cool,High,Strong")
    classifier = getTrainedClassifierOnTennisDataset()
    prediction, probability = classifier.predict(observation, "play")
    printPredictionResult(prediction, probability)

def printPredictionResult(pred, prob):
    result = "Prediction {} with probability {}".format(pred, prob)
    print(result)

def getTrainedClassifierOnTennisDataset():
    data = getDataFromTennisDataset()
    classifier = createNaiveBayesClassifier(data)
    classifier.train()
    return classifier

def formatObservation(observationString):
    return observationString.split(",")

def getDataFromTennisDataset():
    tennisDataset = os.path.join("data", "tennis.csv")
    return extractDataFromFile(tennisDataset)

def createNaiveBayesClassifier(data):
    name = "My Naive Bayes Classifier"
    return NaiveBayes.NaiveBayes(name, data)

def extractDataFromFile(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        listOfData = scanLinesToGetList(lines)
    return listOfData

def scanLinesToGetList(lines):
    lineList = []
    for i, line in enumerate(lines):
        fieldList = convertLineToList(line)
        lineList.append(fieldList)
    return lineList

def convertLineToList(line):
    cleaned = cleanLine(line)
    splitted = cleaned.split(",")
    return splitted

def filterFeatures(lineAsList):
    return lineAsList[1:]

def cleanLine(line):
    return line.replace("\n", "")

if __name__=="__main__":
    main(sys.argv)
