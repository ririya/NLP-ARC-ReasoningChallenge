import re
from conceptnet5.db.query import AssertionFinder
import requests
import time

cnfinder = AssertionFinder()
weightThreshold = 1.0

class Word:
    def __init__(self, word,index):
        self.word = word
        self.count = 1
        self.sentenceCount = 1
        self.index = index

def splitSentence(sentence):
    sentence = sentence.lower()
    sentence = replaceNonAlphanumeric(sentence)
    wordsInSentence = sentence.split()
    return wordsInSentence

def replaceNonAlphanumeric(s):
    s = re.sub('[^0-9a-zA-Z]+', ' ', s)
    return s

def printRemainingTime(start_time,numSentences,indSentence,printEvery):

    if indSentence % printEvery == 0:

        elapsed_time = time.time() - start_time

        remainingSentences = numSentences - indSentence
        remainingTime = remainingSentences / printEvery * elapsed_time / 60

        print(str(indSentence + 1) + ' of ' + str(numSentences) + ' elapsed_time = ' + str(
            elapsed_time) + ' seconds, estimated remaining time = ' + str(remainingTime) + ' minutes')

        # filehandler = open(fileName, "wb")
        # pickle.dump([sentenceDictList, lastReadSentenceInd], filehandler)
        # filehandler.close()

        start_time = time.time()

    return start_time


def getConceptNetRelatedWords(word):

    relatedWords = dict()

    try:
        # query = 'http://api.conceptnet.io/related/c/en/' + word + '/?filter=/c/en'
        # # query = 'http://localhost:8084/related/c/en/' + s + '/?filter=/c/en'
        # obj = requests.get(query)
        # obj = obj.json()
        # relatedObj = obj['related']
        # if len(relatedObj) > 0:
        #     for rW in relatedObj:
        #         id = rW['@id']
        #         id = id.replace('/c/en/', '')
        #         weight = rW['weight']
        #
        #         if id in relatedWords:
        #             relatedWords[id] = max(relatedWords[id],weight)
        #         else:
        #             relatedWords[id] = weight

        # start_time1 = time.time()

        obj = cnfinder.lookup('/c/en/' + word, limit=300)

        for edge in obj:
            id = edge['@id']
            idSplit = id.split(',')
            if '/en/' not in idSplit[1] or '/en/' not in idSplit[2]:
                continue

            start =  idSplit[1].replace('/c/en/','')
            indSlash = start.find('/')
            start = start[:indSlash]

            end = idSplit[2].replace('/c/en/','')
            indSlash = end.find('/')
            end = end[:indSlash]

            if start == word:
                rW = end
            else:
                rW = start

            weight = edge['weight']
            if rW in relatedWords and weight >= weightThreshold:
                relatedWords[rW] = max(relatedWords[rW],weight)
            else:
                relatedWords[rW] = weight

            pass

        # elapsed_time1 = time.time() - start_time1

        # start_time2 = time.time()

        # query = 'http://api.conceptnet.io/related/c/en/' + word
        # # query = 'http://localhost:8084/related/c/en/' + s + '/?filter=/c/en'
        # obj = requests.get(query).json()
        # obj = obj.json()
        #
        # elapsed_time2 = time.time() - start_time1
        #
        # print('elapsed_time1=' + str(elapsed_time1) + 'elapsed_time2=' + str(elapsed_time2))

    except:
        pass

    return relatedWords

def updateSparseWithRelatedWords(word, corpusVocabulary, rows, cols, occurrences, indRow):

    relatedWords = getConceptNetRelatedWords(word)

    for rW in relatedWords:

        if rW in corpusVocabulary:
            indWord = corpusVocabulary[rW].index
            rows.append(indRow)
            cols.append(indWord)
            occurrences.append(1)

    return (rows, cols, occurrences,relatedWords)



