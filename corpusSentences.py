import time
import util
import pickle
from scipy.sparse import csr_matrix
import numpy as np

def buildSentenceDictionaries(sentences,sentenceDictList,lastReadSentenceInd, corpusVocabulary, fileName):

    start_time_all = time.time()
    start_time = start_time_all

    numSentences = len(sentences)

    for indSentence, sentence in enumerate(sentences):

        start_time = util.printRemainingTime(start_time, numSentences, indSentence, 10000)

        if indSentence <= lastReadSentenceInd:
            continue

        sentenceDict = dict()
        wordsAlreadyRead = set()

        wordsInSentence = util.splitSentence(sentence)

        for word in wordsInSentence:

            if word not in wordsAlreadyRead:

                sentenceDict[word] = 1 / corpusVocabulary[word].sentenceCount

            wordsAlreadyRead.add(word)

        sentenceDictList.append(sentenceDict)

        lastReadSentenceInd = indSentence

    filehandler = open(fileName, "wb")
    pickle.dump([sentenceDictList, lastReadSentenceInd], filehandler)
    filehandler.close()

    elapsed_time = time.time() - start_time_all
    print('total time for building sentence list = ' + str(elapsed_time/60) + ' minutes ')

    return sentenceDictList


def buildSentenceMatrix(sentences,rows,cols,weights,lastReadSentenceInd, corpusVocabulary, fileName):

    start_time_all = time.time()
    start_time = start_time_all

    numSentences = len(sentences)

    numberWordsPerSentence = np.zeros(numSentences)

    for indSentence, sentence in enumerate(sentences):

        start_time = util.printRemainingTime(start_time, numSentences, indSentence, 10000)

        if indSentence <= lastReadSentenceInd:
            continue

        wordsAlreadyRead = set()

        wordsInSentence = util.splitSentence(sentence)

        for word in wordsInSentence:

            if word not in wordsAlreadyRead:

                indWord = corpusVocabulary[word].index
                rows.append(indSentence)
                cols.append(indWord)
                weights.append(1 / corpusVocabulary[word].sentenceCount)
                numberWordsPerSentence[indSentence] += 1

            wordsAlreadyRead.add(word)

        lastReadSentenceInd = indSentence

    sentenceMatrix = csr_matrix((weights, (rows, cols)), shape=(numSentences, len(corpusVocabulary)))

    filehandler = open(fileName, "wb")
    pickle.dump([sentenceMatrix, numberWordsPerSentence, lastReadSentenceInd], filehandler)
    filehandler.close()

    elapsed_time = time.time() - start_time_all
    print('total time for building sentence matrix = ' + str(elapsed_time/60) + ' minutes ')

    return (sentenceMatrix,numberWordsPerSentence)