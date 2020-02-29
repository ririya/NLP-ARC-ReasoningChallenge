import time
import json
import util
import pickle
import numpy as np
from scipy.sparse import csr_matrix

def buildQuestionDictionaries(f, fileName, lastReadQuestionInd, questionList, corpusVocabulary, wordCountThreshold):
    printEvery = 100

    questions = f.readlines()
    numQuestions = len(questions)

    start_time = time.time()
    start_time_print = start_time

    for indQ, q in enumerate(questions):

        if indQ % printEvery == 0:
            elapsed_time = time.time() - start_time_print

            remainingSentences = numQuestions - indQ
            remainingTime = remainingSentences / printEvery * elapsed_time / 60

            print(str(indQ + 1) + ' of ' + str(numQuestions) + ' elapsed_time = ' + str(
                elapsed_time) + ' seconds, estimated remaining time = ' + str(remainingTime) + ' minutes')

            # filehandler = open(fileName, "wb")
            # pickle.dump([questionWordMatrix, lastReadQuestionInd], filehandler)
            # filehandler.close()

            start_time_print = time.time()

        if indQ <= lastReadQuestionInd:
            continue

        j = json.loads(q)
        question = j['question']
        stem = question['stem']
        wordsInQuestion = util.splitSentence(stem)

        choices = question['choices']
        for c in choices:
            choiceText = c['text']
            wordsInChoice = util.splitSentence(choiceText)
            wordsInQuestion.extend(wordsInChoice)

        wordsAlreadyRead = set()

        wordVec = dict()

        for word in wordsInQuestion:

            if word not in wordsAlreadyRead:

                if word not in corpusVocabulary:

                    relatedWords = util.getConceptNetRelatedWords(word)
                    wordVec[word] = relatedWords

                else:
                    if corpusVocabulary[word].sentenceCount < wordCountThreshold:

                        # st = time.time()

                        relatedWords = util.getConceptNetRelatedWords(word)
                        wordVec[word] = relatedWords

                        # el = time.time() - st
                        # print('elapsed_time_getConceptNetWords = ' + str(el))

                    else:
                        wordVec[word] = []

            wordsAlreadyRead.add(word)

        questionList.append(wordVec)

        lastReadQuestionInd = indQ

    filehandler = open(fileName, "wb")
    pickle.dump([questionList, lastReadQuestionInd], filehandler)
    filehandler.close()

    elapsed_time = time.time() - start_time

    print('total time for building question matrix = ' + str(elapsed_time/60) + ' minutes ')

    return questionList

def buildQuestionMatrix(questions, numQuestions, fileName, lastReadQuestionInd, rows, cols,occurrences, corpusVocabulary, wordCountThreshold):

    start_time = time.time()
    start_time_print = start_time

    for indQ, q in enumerate(questions):

        start_time_print = util.printRemainingTime(start_time_print, numQuestions, indQ, 100)

        if indQ <= lastReadQuestionInd:
            continue

        j = json.loads(q)
        question = j['question']
        stem = question['stem']
        wordsInQuestion = util.splitSentence(stem)

        choices = question['choices']
        for c in choices:
            choiceText = c['text']
            wordsInChoice = util.splitSentence(choiceText)
            wordsInQuestion.extend(wordsInChoice)

        wordsAlreadyRead = set()

        for word in wordsInQuestion:

            if word not in wordsAlreadyRead:

                if word not in corpusVocabulary:  #word not in the vocabulary, so it's a rare word: look for similar words

                    (rows, cols, occurrences) = util.updateSparseWithRelatedWords(word, corpusVocabulary, rows, cols, occurrences, indQ)

                else:

                    indWord = corpusVocabulary[word].index
                    rows.append(indQ)
                    cols.append(indWord)
                    occurrences.append(1)

                    if corpusVocabulary[word].sentenceCount < wordCountThreshold:   #only use conceptnet for rare words

                        (rows, cols, occurrences) = util.updateSparseWithRelatedWords(word, corpusVocabulary, rows, cols, occurrences, indQ)

            wordsAlreadyRead.add(word)

        lastReadQuestionInd = indQ

    questionMatrix = csr_matrix((occurrences, (rows, cols)), shape=(numQuestions, len(corpusVocabulary)))

    filehandler = open(fileName, "wb")
    pickle.dump([questionMatrix, lastReadQuestionInd], filehandler)
    filehandler.close()

    elapsed_time = time.time() - start_time

    print('total time for building question matrix = ' + str(elapsed_time/60) + ' minutes ')

    return questionMatrix



def addWordsToSparseMatrix(wordsInSentence, corpusVocabulary, wordCountThreshold):

    wordsAlreadyRead = set()

    rows = []
    cols = []
    occurrences = []


    rowsCN = []
    colsCN = []
    occurrencesCN = []

    for word in wordsInSentence:

        if word not in wordsAlreadyRead:

            if word not in corpusVocabulary:

                (rowsCN, colsCN, occurrencesCN) = util.updateSparseWithRelatedWords(word, corpusVocabulary, rowsCN, colsCN, occurrencesCN, 0)

            else:

                indWord = corpusVocabulary[word].index
                rows.append(0)
                cols.append(indWord)
                occurrences.append(1)

                if corpusVocabulary[word].sentenceCount < wordCountThreshold:  # only use conceptnet for somewhat rare words

                    (rowsCN, colsCN, occurrencesCN) = util.updateSparseWithRelatedWords(word, corpusVocabulary, rowsCN, colsCN,  occurrencesCN, 0)

        wordsAlreadyRead.add(word)

    sentenceVector = csr_matrix((occurrences, (rows, cols)), shape=(1, len(corpusVocabulary)),dtype=float)
    relatedVector = csr_matrix((occurrencesCN, (rowsCN, colsCN)), shape=(1, len(corpusVocabulary)), dtype = float)

    return (sentenceVector,relatedVector)

def getRelatedWordsBothways(wordsInQuestion,wordsInChoices, questionVector, choicesVector,wordCountThreshold,corpusVocabulary,inverseDictionary):
    relatedWords = []
    relatedWords = getRelatedWords(wordsInQuestion, choicesVector, wordCountThreshold, corpusVocabulary,inverseDictionary, relatedWords)
    relatedWords = getRelatedWords(wordsInChoices, questionVector, wordCountThreshold, corpusVocabulary,inverseDictionary, relatedWords)
    return relatedWords

def getRelatedWords(wordsInQuestion,choicesVector,wordCountThreshold,corpusVocabulary,inverseDictionary, relatedWords):

    wordsAlreadyRead = set()

    for word in wordsInQuestion:

        if word not in wordsAlreadyRead:

            if word in corpusVocabulary: #ignore words not present in vocabulary

                if corpusVocabulary[word].sentenceCount < wordCountThreshold:  # only use conceptnet for somewhat rare words

                    rowsCN = []
                    colsCN = []
                    occurrencesCN = []

                    (rowsCN, colsCN, occurrencesCN,CN_relatedWords) = util.updateSparseWithRelatedWords(word, corpusVocabulary, rowsCN, colsCN, occurrencesCN, 0)

                    relatedVector = csr_matrix((occurrencesCN, (rowsCN, colsCN)), shape=(1, len(corpusVocabulary)), dtype=float)

                    commonWords = relatedVector.toarray()[0]*choicesVector.toarray()[0]

                    commonInd = np.where(commonWords>0)[0]

                    # choiceWords = []

                    # choicesVector2 = choicesVector.toarray()[0]
                    # nonzero = np.where(choicesVector2 > 0)[0]
                    # for c in nonzero:
                    #     choiceWords.append(inverseDictionary[c])

                    for cI in commonInd:

                        cIWord = inverseDictionary[cI]

                        pair = sorted([corpusVocabulary[word].index,cI])

                        if pair not in relatedWords:
                            relatedWords.append(pair)

    return relatedWords





def getSentenceVector(wordsInSentence, corpusVocabulary):


    wordsAlreadyRead = set()

    rows = []
    cols = []
    occurrences = []

    for word in wordsInSentence:

        if word not in wordsAlreadyRead:

            if word in corpusVocabulary:  # if word not in vocabulary, ignore it

                indWord = corpusVocabulary[word].index
                rows.append(0)
                cols.append(indWord)
                occurrences.append(1)

        wordsAlreadyRead.add(word)

    sentenceVector = csr_matrix((occurrences, (rows, cols)), shape=(1, len(corpusVocabulary)),dtype=float)

    return (sentenceVector)


def buildQuestionMatrix2(testQuestionIndices,questionMatrix,choiceMatrix,relatedMatrix,questions, numQuestions, fileName, lastReadQuestionInd, corpusVocabulary, inverseDictionary, wordCountThreshold):

    start_time = time.time()
    start_time_print = start_time

    usedQuestions = []

    if len(testQuestionIndices)>0:
        for q in testQuestionIndices:
            usedQuestions.append(questions[testQuestionIndices[q]])
    else:
        usedQuestions = questions

    for indQ, q in enumerate(usedQuestions):

        start_time_print = util.printRemainingTime(start_time_print, numQuestions, indQ, 100)

        if indQ <= lastReadQuestionInd:
            continue

        j = json.loads(q)
        question = j['question']
        stem = question['stem']
        wordsInQuestion = util.splitSentence(stem)

        choices = question['choices']

        wordsInChoices = []

        for c in choices:
            choiceText = c['text']
            wordsInCurrChoice = util.splitSentence(choiceText)
            wordsInChoices.extend(wordsInCurrChoice)

        # questionVector, relatedQuestionVector = addWordsToSparseMatrix(wordsInQuestion, corpusVocabulary, wordCountThreshold)
        # choicesVector, relatedChoicesVector = addWordsToSparseMatrix(wordsInChoices, corpusVocabulary, wordCountThreshold)

        questionVector = getSentenceVector(wordsInQuestion, corpusVocabulary)
        choicesVector = getSentenceVector(wordsInChoices, corpusVocabulary)

        # questionWords = []
        # choiceWords = []
        # questionVector2 = questionVector.toarray()[0]
        # nonzero = np.where(questionVector2 > 0)[0]
        #
        # for c in nonzero:
        #    questionWords.append(inverseDictionary[c])
        #
        # choicesVector2 = choicesVector.toarray()[0]
        # nonzero = np.where(choicesVector2 > 0)[0]
        #
        # for c in nonzero:
        #     choiceWords.append(inverseDictionary[c])

        relatedWords = getRelatedWordsBothways(wordsInQuestion,wordsInChoices, questionVector, choicesVector,wordCountThreshold,corpusVocabulary,inverseDictionary)


        questionMatrix.append(questionVector)
        choiceMatrix.append(choicesVector)
        relatedMatrix.append(relatedWords)

        lastReadQuestionInd = indQ

    filehandler = open(fileName, "wb")
    pickle.dump([questionMatrix,choiceMatrix,relatedMatrix, lastReadQuestionInd], filehandler)
    filehandler.close()

    elapsed_time = time.time() - start_time

    print('total time for building question matrix = ' + str(elapsed_time/60) + ' minutes ')

    return (questionMatrix,choiceMatrix, relatedMatrix)
