class NaiveBayes:
    # Attributes
    ## modelName
    ## dataTable
    ## prior prob table
    # probabilityTable
    ## domains
    ## fields

    # Methods
    def __init__(self, name, dataTable = []):
        self.modelName = name
        self.setFields(dataTable)
        self.setDataTable(dataTable)

    def setDataTable(self, dataTable):
        self.dataTable = dataTable[1:]
        self.defineDomains()
        self.priorProbabilities = []

    def setFields(self, dataTable):
        self.fields = dataTable[0]

    def defineDomains(self):
        self.domains = []
        for i, f in enumerate(self.fields):
            dom = self.collectDomainOfField(i)
            self.domains.append(dom)

    def collectDomainOfField(self, fieldID):
        dom = set()
        for i, line in enumerate(self.dataTable):
            fieldValue = line[fieldID]
            dom.add(fieldValue)
        return sorted(list(dom))

    def train(self):
        self.computeAllPriorProbabilities()
        self.computeAllLikelihoodOfData()

    def computeAllLikelihoodOfData(self):
        likelihood = []
        for f, field in enumerate(self.fields):
            fieldLikelihoods = []
            domain = self.getFieldDomain(f)
            for v, value in enumerate(domain):
                likelihoodWithHypotesis = self.computeLikelihoodOfDataKnowingHypotesis(f, v)
                fieldLikelihoods.append(likelihoodWithHypotesis)
            likelihood.append(fieldLikelihoods)
        self.likelihoodTable = likelihood

    def computeLikelihoodOfDataKnowingHypotesis(self, hypotesisField, hypotesisValue):
        subsetOfInstances = self.getLinesWhereFieldHasValue(hypotesisField, hypotesisValue)
        numberOfInstances = len(subsetOfInstances)

        dataLikelihood = []
        for f, field in enumerate(self.fields):
            fieldDistribution = []
            for v, value in enumerate(self.getFieldDomain(f)):
                valueOccurrences = self.countTimesThatFieldHasValueInDataSubset(f, v, subsetOfInstances)
                valueLikelihood = valueOccurrences/numberOfInstances
                fieldDistribution.append(valueLikelihood)
            dataLikelihood.append(fieldDistribution)
        return dataLikelihood

    def getFieldDomain(self, fieldID):
        return self.domains[fieldID]


    def computeAllPriorProbabilities(self):
        allPriorProbs = []
        for i, field in enumerate(self.fields):
            priorDistributionOfField = self.computeSinglePriorProbability(i)
            allPriorProbs.append(priorDistributionOfField)
        self.priorProbabilities = allPriorProbs

    def computeSinglePriorProbability(self, fieldID):
        priorDistributionOfField = []
        for j, val in enumerate(self.getFieldDomain(fieldID)):
            probThatFieldHasVal = self.computeProbThatFieldHasValue(fieldID, j)
            priorDistributionOfField.append(probThatFieldHasVal)
        return priorDistributionOfField

    def computeProbThatFieldHasValue(self, i, j):
        value = self.getIthValueInDomain(i, j)
        valueOccurrences = self.countTimesThatFieldHasValue(i, value)
        numOfIstances = self.getNumberOfIstances()
        return valueOccurrences/numOfIstances

    def getIthValueInDomain(self, fieldID, valueID):
        dom = self.getFieldDomain(fieldID)
        return dom[valueID]

    def countTimesThatFieldHasValue(self, i, value):
        return self.countTimesThatFieldHasValueInDataSubset(i, value, self.dataTable)

    def countTimesThatFieldHasValueInDataSubset(self, i, value, lines):
        countOccurrences = 0
        for line in lines:
            currentValue = line[i]
            if currentValue == value:
                countOccurrences = countOccurrences + 1
        return countOccurrences

    def getLinesWhereFieldHasValue(self, fieldID, valueID):
        linesSubset = []
        value = self.getIthValueInDomain(fieldID, valueID)
        for i, line in enumerate(self.dataTable):
            currentValue = line[fieldID]
            if currentValue == value:
                linesSubset.append(line)
        return linesSubset

    def getNumberOfIstances(self):
        return self.getNumberOfIstancesInDataSubset(self.dataTable)

    def getNumberOfIstancesInDataSubset(self, lines):
        return len(lines)
