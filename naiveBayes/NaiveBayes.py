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

    def predict(self, observation, predictField):
        fieldID = self.getFieldIDByName(predictField)
        hMAP, prob = self.getMaxLikelihoodValue(observation, predictField)
        predictedValue = self.getIthValueInDomain(fieldID, hMAP)
        return predictedValue, prob

    def getMaxLikelihoodValue(self, observation, predictField):
        allProbabilities = self.getProbDistributionForEachValue(observation, predictField)
        maxProb = max(allProbabilities)
        sumUpProbs = sum(prob for prob in allProbabilities)

        hMAP = allProbabilities.index(maxProb)
        probHMAP = maxProb/sumUpProbs
        return hMAP, probHMAP

    def getProbDistributionForEachValue(self, observation, predictField):
        predFieldID = self.getFieldIDByName(predictField)
        probHypotesis = []
        for valueID, hypValue in enumerate(self.getFieldDomain(predFieldID)):
            prob = self.getProbabilityOfHypotesis(observation, predFieldID, valueID)
            probHypotesis.append(prob)
        return probHypotesis

    def getProbabilityOfHypotesis(self, observation, hypFieldID, hypValueID):
        product = self.getProductOfCondIndipendentObservation(observation, hypFieldID, hypValueID)
        prior = self.getPriorProbability(hypFieldID, hypValueID)
        return product*prior

    def getProductOfCondIndipendentObservation(self, observation, hypFieldID, hypValueID):
        hypDistributionAllFields = self.getProbDistributionOfHypotesis(hypFieldID, hypValueID)
        product = 1
        for fieldID, obsValue in enumerate(observation):
            obsValID = self.getValueIDByName(fieldID, obsValue)
            hypDistrCurrentField = hypDistributionAllFields[fieldID]
            product = product * hypDistrCurrentField[obsValID]
        return product

    def getProbDistributionOfHypotesis(self, fieldID, valueID):
        hypotesis = (fieldID, valueID)
        return self.likelihoodTable.get(hypotesis)

    def getPriorProbability(self, fieldID, valueID):
        return self.priorProbabilities[fieldID][valueID]

    def getFieldIDByName(self, field):
        return self.fields.index(field)

    def getValueIDByName(self, fieldID, value):
        domain = self.getFieldDomain(fieldID)
        return domain.index(value)

    def train(self):
        self.computeAllPriorProbabilities()
        self.computeAllLikelihoodOfData()

    def computeAllLikelihoodOfData(self):
        likelihood = dict()
        for f, field in enumerate(self.fields):
            domain = self.getFieldDomain(f)
            for v, value in enumerate(domain):
                likelihoodWithHypotesis = self.computeLikelihoodOfDataKnowingHypotesis(f, v)
                likelihood[(f,v)] = likelihoodWithHypotesis
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

    def computeProbThatFieldHasValue(self, fieldID, valueID):
        valueOccurrences = self.countTimesThatFieldHasValue(fieldID, valueID)
        numOfIstances = self.getNumberOfIstances()
        return valueOccurrences/numOfIstances

    def getIthValueInDomain(self, fieldID, valueID):
        dom = self.getFieldDomain(fieldID)
        return dom[valueID]

    def countTimesThatFieldHasValue(self, i, value):
        return self.countTimesThatFieldHasValueInDataSubset(i, value, self.dataTable)

    def countTimesThatFieldHasValueInDataSubset(self, fieldID, valueID, lines):
        countOccurrences = 0
        value = self.getIthValueInDomain(fieldID, valueID)
        for line in lines:
            currentValue = line[fieldID]
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
