import time
import pickle
import os
import numpy as np
import json

from questions import*
from corpusSentences import*
from getWordCounts import*
from findSentences import*
from scipy.sparse import csr_matrix

wordCountThreshold = 100000
numberOfRelevantSentences = 10
corpusDictionaryFileName = "corpusDictionary.pkl"

if os.path.exists(corpusDictionaryFileName):
    filehandler = open(corpusDictionaryFileName, "rb")
    object_file = pickle.load(filehandler)
    wordCount = object_file[0]
    wordCountSentence = object_file[1]
    corpusVocabulary = object_file[2]
    inverseDictionary = object_file[3]
    lastReadSentenceInd = object_file[4]
    #
    # sorted_wordCount = sorted(wordCount.items(),reverse = True, key=lambda kv: kv[1])
    # sorted_wordCountSentence = sorted(wordCountSentence.items(),reverse = True, key=lambda kv: kv[1])

    # counts = np.asarray(list(wordCountSentence.values()))
    # mean_count = np.mean(counts)
    # std_count = np.std(counts)
    # median_count = np.median(counts)
    # perc_count = np.percentile(counts,0.25)
    # perc_count = np.percentile(counts, 0.2)
    # perc_count = np.percentile(counts, 0.1)


else:
    wordCount = dict()
    wordCountSentence = dict()
    corpusVocabulary = dict()
    inverseDictionary = dict()
    lastReadSentenceInd = -1

f = open("ARC-V1-Feb2018-2/ARC_Corpus.txt", "r")

sentences = f.readlines()

numSentences = len(sentences)

if lastReadSentenceInd < numSentences - 1:
#
    (wordCount,wordCountSentence, corpusVocabulary, lastReadSentenceInd) = getWordCounts(sentences,numSentences,corpusDictionaryFileName,corpusVocabulary,inverseDictionary, wordCount,wordCountSentence,lastReadSentenceInd)

sentenceMatrixFileName = 'sentenceMatrix.pkl'

if os.path.exists(sentenceMatrixFileName):
    filehandler = open(sentenceMatrixFileName, "rb")
    object_file = pickle.load(filehandler)
    sentenceMatrix = object_file[0]
    numberWordsPerSentence = object_file[1]
    lastReadSentenceInd = object_file[2]

else:
    # sentenceMatrix = np.zeros((numSentences,len(corpusVocabulary)))
    # sentenceMatrix = csr_matrix((numSentences,len(corpusVocabulary)), dtype=np.float)
    # a = np.ones(len(corpusVocabulary))
    # b = sentenceMatrix.dot(a)
    rows = []
    cols = []
    weights = []
    # sentenceDictList = []
    lastReadSentenceInd = -1

if lastReadSentenceInd < numSentences - 1:

    [sentenceMatrix,numberWordsPerSentence] = buildSentenceMatrix(sentences,rows,cols,weights,lastReadSentenceInd, corpusVocabulary, sentenceMatrixFileName)

# testQuestionIndices = []
testQuestionIndices = range(20)

jsonF = open("ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Train.jsonl")
pklFile = "easyTrainMatrix.pkl"
simpleIRSaveDir = 'easyTrain-sentencesSimpleIR/'
conceptnetSaveDir = 'easyTrain-sentencesConceptnet/'
testQuestions(testQuestionIndices,pklFile, jsonF,corpusVocabulary,inverseDictionary,wordCountThreshold,sentences,numberWordsPerSentence,sentenceMatrix,numberOfRelevantSentences,simpleIRSaveDir,conceptnetSaveDir)

jsonF = open("ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl")
pklFile = "easyTestMatrix.pkl"
simpleIRSaveDir = 'easyTest-sentencesSimpleIR/'
conceptnetSaveDir = 'easyTest-sentencesConceptnet/'
testQuestions(testQuestionIndices,pklFile, jsonF,corpusVocabulary,inverseDictionary,wordCountThreshold,sentences,numberWordsPerSentence,sentenceMatrix,numberOfRelevantSentences,simpleIRSaveDir,conceptnetSaveDir)

#
jsonF = open("ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl")
pklFile = "challengeTestMatrix.pkl"
simpleIRSaveDir = 'challengeTest-sentencesSimpleIR/'
conceptnetSaveDir = 'challengeTest-sentencesConceptnet/'

testQuestions(testQuestionIndices,pklFile, jsonF,corpusVocabulary,inverseDictionary,wordCountThreshold,sentences,numberWordsPerSentence,sentenceMatrix,numberOfRelevantSentences,simpleIRSaveDir,conceptnetSaveDir)


jsonF = open("ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Train.jsonl")
pklFile = "challengeTrainMatrix.pkl"
simpleIRSaveDir = 'challengeTrain-sentencesSimpleIR/'
conceptnetSaveDir = 'challengeTrain-sentencesConceptnet/'

testQuestions(testQuestionIndices,pklFile, jsonF,corpusVocabulary,inverseDictionary,wordCountThreshold,sentences,numberWordsPerSentence,sentenceMatrix,numberOfRelevantSentences,simpleIRSaveDir,conceptnetSaveDir)





