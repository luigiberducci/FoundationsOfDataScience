import NaiveBayes
import sys

def main(argv):
    data = extractDataFromFile("data/tennis.csv")
    classifier = createNaiveBayesClassifier(data)
    classifier.train()

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
    return cleaned.split(",")

def cleanLine(line):
    return line.replace("\n", "")

if __name__=="__main__":
    main(sys.argv)
