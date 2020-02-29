import time
import numpy as np
import util
import os
import json
import gc
import pickle
from heapq import nlargest
from questions import*

class BestSentences:
    def __init__(self, maxSentences):
        self.indices = -1*np.ones(maxSentences)
        self.scores = np.zeros(maxSentences)

    def update(self,newScore, newInd):

        if newScore > np.min(self.scores):
            indToReplace = np.argmin(self.scores)
            self.scores[indToReplace] = newScore
            self.indices[indToReplace] = newInd

    def findBestScores(self, scores):
        for ind,score in enumerate(scores):
            if score > np.min(self.scores):
                indToReplace = np.argmin(self.scores)
                self.scores[indToReplace] = score
                self.indices[indToReplace] = ind


def findSentencesForQuestion(question,sentenceDictList,maxSentences,useConceptNet):

    start_time = time.time()

    bestSentences =  BestSentences(maxSentences)

    for indSentence, sentence in enumerate(sentenceDictList):
        sentenceScore = 0
        for word in question:
            if word in sentence:
                sentenceScore += sentence[word]
            # if len(question[word]) > 0 and useConceptNet:
            #     for relatedWord in question[word]:
            #         if relatedWord in sentence:
            #             sentenceScore += sentence[relatedWord]
        bestSentences.update(sentenceScore, indSentence)

    elapsed_time = time.time() - start_time

    print('elapsed time to find relevant sentences = ' + str(elapsed_time))

    return bestSentences

def findSentences(originalQuestions, originalSentences,numberWordsPerSentence,questionMatrix,choiceMatrix,relatedMatrix,numQuestions,sentences, maxSentences, useConceptNet,saveDir, relatedWordsWeight=0.01):

    # allBestSentences = []

    start_time_all = time.time()
    start_time = start_time_all

    for indQ, question in enumerate(questionMatrix):
        # bestSentences = findSentencesForQuestion(question,sentences,maxSentences,useConceptNet)

        choice = choiceMatrix[indQ]
        related = relatedMatrix[indQ]

        savePath = saveDir +  str(indQ) + '.txt'
        picklePath = saveDir +  str(indQ) + '.pkl'

        if not os.path.exists(savePath):

            bestSentences = findSentencesForQuestion_SparseMatrices(originalSentences,question,choice,related, sentences, numberWordsPerSentence, maxSentences, useConceptNet, relatedWordsWeight)

            f = open(savePath,'w')
            f.write('Question:\n')
            j = json.loads(originalQuestions[indQ])
            question = j['question']
            stem = question['stem']
            f.write(stem + '\n')

            choices = question['choices']
            for c in choices:
                choiceText = c['text']
                label = c['label']
                f.write(label + ') ' +  choiceText + '\n')

            f.write('\n')

            f.write('Relevant Sentences:' + '\n')

            for indSentence in bestSentences.indices:
                f.write(originalSentences[indSentence]  + '\n')

            f.close()

            bestSentencesPickelArray = []

            for ind in bestSentences.indices:
                bestSentencesPickelArray.append(originalSentences[ind])
            filehandler = open(picklePath, "wb")
            pickle.dump([bestSentencesPickelArray,bestSentences.indices], filehandler)
            filehandler.close()

            gc.collect()

            # allBestSentences.append(bestSentences)

        start_time = util.printRemainingTime(start_time, numQuestions, indQ, 1)

    elapsed_time = time.time() - start_time

    print('elapsed time to find relevant sentences = ' + str(elapsed_time))

    # return allBestSentences

def findSentencesForQuestion_SparseMatrices(originalSentences,question,choice,related,sentenceMatrix,numberWordsPerSentence,maxSentences,useConceptNet, relatedWordsWeight,normalize = False):

    bestSentences = BestSentences(maxSentences)

    question = question.toarray()[0]
    choice = choice.toarray()[0]

    scoresQuestion = sentenceMatrix.dot(question)
    scoresChoice = sentenceMatrix.dot(choice)

    scoreMask = (scoresQuestion*scoresChoice).astype(bool)

    scores = (scoresQuestion + scoresChoice)*scoreMask

    if normalize:
        scores = scores / numberWordsPerSentence

    if useConceptNet:

        priority = []

        for pair in related:

            s0 = sentenceMatrix[:,pair[0]]
            s1 = sentenceMatrix[:,pair[1]]

            containsBoth = s0.toarray()*s1.toarray()

            indContainsBoth = np.where(containsBoth > 0)[0]

            priority.extend(indContainsBoth.tolist())

        # related = related.toarray()[0]
        # scoreRelated = sentenceMatrix.dot(related)
        # scores += scoreRelated*relatedWordsWeight

        priorityScores =np.zeros(len(scores))
        priorityScores[priority] = scores[priority]
        scores[priority] = 0

        sortedScoresPriorityIndices = np.argsort(-priorityScores)
        sortedPriorityScores = priorityScores[sortedScoresPriorityIndices]
        indZero = np.where(sortedPriorityScores == 0)[0]
        sortedScoresPriorityIndices = np.delete(sortedScoresPriorityIndices, indZero)

        sortedScoresIndices = np.argsort(-scores)
        sortedScoresIndices = np.concatenate((sortedScoresPriorityIndices,sortedScoresIndices))

        # del scoreRelated

    else:



        # bestIndices = nlargest(maxSentences, range(len(scores)), key=lambda idx: scores[idx])
        #
        sortedScoresIndices = np.argsort(-scores)

    # bestSentences.findBestScores(scores)


    bestSentences.indices = sortedScoresIndices[:maxSentences]
    bestSentences.scores = scores[bestSentences.indices]

    del question
    del choice
    del related
    del scoreMask
    del scoresQuestion
    del scoresChoice
    del scores
    del sortedScoresIndices


    return bestSentences

def testQuestions(testQuestionIndices,filename, jsonFile,corpusVocabulary,inverseDictionary,wordCountThreshold,sentences,numberWordsPerSentence,sentenceMatrix,numberOfRelevantSentences,simpleIRSaveDir,conceptnetSaveDir):
    if os.path.exists(filename):
        filehandler = open(filename, "rb")
        object_file = pickle.load(filehandler)
        questionMatrix = object_file[0]
        choiceMatrix = object_file[1]
        relatedMatrix = object_file[2]
        lastReadQuestionInd = object_file[3]

    else:
        questionMatrix = []
        choiceMatrix = []
        relatedMatrix = []
        rows = []
        cols = []
        occurrences = []
        lastReadQuestionInd = -1

    questions = jsonFile.readlines()
    numQuestions = len(questions)

    if lastReadQuestionInd < numQuestions - 1:
        # questionList = buildQuestionDictionaries(f_Easy, easyTestFileName, lastReadQuestionInd, questionList, corpusVocabulary, wordCountThreshold)
        questionMatrix,choiceMatrix,relatedMatrix  = buildQuestionMatrix2(testQuestionIndices,questionMatrix,choiceMatrix,relatedMatrix,questions, numQuestions, filename, lastReadQuestionInd,corpusVocabulary,inverseDictionary, wordCountThreshold)

    if not os.path.exists(simpleIRSaveDir):
        os.mkdir(simpleIRSaveDir)

    if not os.path.exists(conceptnetSaveDir):
        os.mkdir(conceptnetSaveDir)

    findSentences(questions, sentences, numberWordsPerSentence, questionMatrix,choiceMatrix,relatedMatrix, numQuestions, sentenceMatrix,
                  numberOfRelevantSentences, True, conceptnetSaveDir)

    findSentences(questions, sentences, numberWordsPerSentence, questionMatrix,choiceMatrix,relatedMatrix, numQuestions, sentenceMatrix,
                  numberOfRelevantSentences, False, simpleIRSaveDir)










