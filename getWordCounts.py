import time
import util
import pickle

def getWordCounts(sentences,numSentences,fileName,corpusVocabulary,inverseDictionary, wordCount,wordCountSentence,lastReadSentenceInd):

    start_time_all = time.time()
    start_time = start_time_all

    for indSentence, sentence in enumerate(sentences):

        start_time = util.printRemainingTime(start_time, numSentences, indSentence, 10000)

        if indSentence <= lastReadSentenceInd:
            continue

        wordsInSentence = util.splitSentence(sentence)

        wordsAlreadyRead = set()

        for word in wordsInSentence:

            if word in wordCount:
                # corpusDictionary[word].count += 1
                wordCount[word] += 1
                corpusVocabulary[word].count += 1

                if word not in wordsAlreadyRead:
                    wordCountSentence[word] += 1
                    corpusVocabulary[word].sentenceCount += 1

                wordsAlreadyRead.add(word)

            else:
                # relatedWords = getConceptNetRelatedWords(word)
                # corpusDictionary[word] = Word(relatedWords)
                wordCount[word] = 1
                wordCountSentence[word] = 1
                inverseDictionary[len(corpusVocabulary)] = word
                corpusVocabulary[word] = util.Word(word,len(corpusVocabulary))


        lastReadSentenceInd = indSentence

        # filehandler = open(fileName, "wb")
        # # pickle.dump([corpusDictionary, lastReadSentenceInd],filehandler)
        # pickle.dump([wordCount,wordCountSentence, lastReadSentenceInd], filehandler)
        # filehandler.close()

    filehandler = open(fileName, "wb")
    # pickle.dump([corpusDictionary, lastReadSentenceInd],filehandler)
    pickle.dump([wordCount, wordCountSentence, corpusVocabulary,inverseDictionary, lastReadSentenceInd], filehandler)
    filehandler.close()

    elapsed_time = time.time() - start_time_all
    print('total time for word count = ' + str(elapsed_time/60) + ' minutes ')

    return(wordCount,wordCountSentence, corpusVocabulary, lastReadSentenceInd)
