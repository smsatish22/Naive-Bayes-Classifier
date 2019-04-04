# use this file to learn naive-bayes classifier
# Expected: generate nbmodel.txt

import sys
import os
import numpy as np
from collections import Counter

stopWords = ["a", "im", "am", "an", "all", "also", "again", "as", "and", "anniversary", "ask", "at", "be", "are",
             "betsy", "by", "can", "chicago", "city", "could", "did", "do", "door", "everything", "from", "for", "have",
             "had", "has", "he", "her", "here", "hers", "his", "home", "hotel", "hotels", "how", "husband", "is", "in",
             "i", "if", "it", "its", "james", "just", "knew", "ky", "place", "location", "louisville", "m", "made",
             "me", "mine", "my", "on", "one", "of", "our", "out", "outside", "pier", "room", "rooms", "she", "so",
             "sofitel", "stay", "stayed", "staying", "service", "soon", "supper", "th", "that", "the", "to", "then",
             "them", "this", "there", "their", "they", "those", "time", "trip", "tower", "us", "very", "water", "we",
             "was", "weekend", "were", "wifes", "with", "what", "where", "which", "when", "would", "you", "your"]


def diff(li1, li2):
    return [item for item in li1 if item not in li2]


def extractWords(filePath):
    fileContent = None
    with open(filePath) as fp:
        fileContent = fp.read().replace('\n', ' ').lower()
    return filter(None, map(lambda word: ''.join(filter(str.isalpha, word)), fileContent.split()))


def dirWalk(fileRoot):
    result = []
    for root, dirs, files in os.walk(fileRoot, topdown=True):
        files = [f for f in files if not f[0] == '.']
        for f in files:
            result += extractWords(os.path.join(root, f))
    return result


def findProbability(freq, uniqueWords, totalWordsInGroup):
    return np.float(freq + 1) / (np.float(uniqueWords) + np.float(totalWordsInGroup))


def writeNModel(wordDetails, priorProbabilities):
    with open("nbmodel.txt", "w+") as f:
        f.write("%s %s %s %s" % (
            priorProbabilities['priorPos'], priorProbabilities['priorNeg'], priorProbabilities['priorDec'],
            priorProbabilities['priorTru']))
        f.write("\n")
        for word in wordDetails.keys():
            wordDetail = wordDetails[word]
            f.write("%s %s %s %s %s %s %s %s %s %s" % (
                word, wordDetail['total'], wordDetail['positive'], wordDetail['negative'], wordDetail['deceptive'],
                wordDetail['truthful'], wordDetail['posProbability'], wordDetail['negProbability'],
                wordDetail['deceptiveProbability'], wordDetail['truthfulProbability']))
            f.write("\n")


def readpath(inputPath):
    positive_p = inputPath + '/positive_polarity'
    negative_p = inputPath + '/negative_polarity'
    positiveDeceptive = positive_p + '/deceptive_from_MTurk'
    positiveTruthful = positive_p + '/truthful_from_TripAdvisor'
    negativeDeceptive = negative_p + '/deceptive_from_MTurk'
    negativeTruthful = negative_p + '/truthful_from_Web'
    positiveWords = []
    negativeWords = []
    deceptiveWords = []
    truthfulWords = []

    positiveDeceptiveWords = diff(dirWalk(positiveDeceptive), stopWords)
    positiveWords += positiveDeceptiveWords
    deceptiveWords += positiveDeceptiveWords
    # print positiveDeceptiveWords, len(positiveDeceptiveWords)

    positiveTruthfulWords = diff(dirWalk(positiveTruthful), stopWords)
    positiveWords += positiveTruthfulWords
    truthfulWords += positiveTruthfulWords
    # print positiveTruthfulWords, len(positiveTruthfulWords)

    negativeDeceptiveWords = diff(dirWalk(negativeDeceptive), stopWords)
    negativeWords += negativeDeceptiveWords
    deceptiveWords += negativeDeceptiveWords
    # print negativeDeceptiveWords, len(negativeDeceptiveWords)

    negativeTruthfulWords = diff(dirWalk(negativeTruthful), stopWords)
    negativeWords += negativeTruthfulWords
    truthfulWords += negativeTruthfulWords
    # print negativeTruthfulWords, len(negativeTruthfulWords)

    positiveWordToCount = Counter(positiveWords)
    negativeWordCount = Counter(negativeWords)

    uniqueWordCount = positiveWordToCount + negativeWordCount

    deceptiveWordToCount = Counter(deceptiveWords)
    truthfulWordToCount = Counter(truthfulWords)

    positive_docs = list()
    for (root, dirs, files) in os.walk(positive_p, topdown=True):
        positive_docs += [os.path.join(root, file) for file in files]
        # positive_docs.remove(['.DS_Store'])

    negative_docs = list()
    for (root, dirs, files) in os.walk(negative_p, topdown=True):
        negative_docs += [os.path.join(root, file) for file in files]
        # negative_docs.remove(['.DS_Store'])
    # print (negative_words)

    positive_deceptive_docs = list()
    for (root, dirs, files) in os.walk(positiveDeceptive, topdown=True):
        positive_deceptive_docs += [os.path.join(root, file) for file in files]
        ##positive_deceptive_words.remove(['.DS_Store'])

    negative_deceptive_docs = list()
    for (root, dirs, files) in os.walk(negativeDeceptive, topdown=True):
        negative_deceptive_docs += [os.path.join(root, file) for file in files]
        ##negative_deceptive_words.remove(['.DS_Store'])

    positive_truthful_docs = list()
    for (root, dirs, files) in os.walk(positiveTruthful, topdown=True):
        positive_truthful_docs += [os.path.join(root, file) for file in files]
        ##positive_truthful_words.remove(['.DS_Store'])

    negative_truthful_docs = list()
    for (root, dirs, files) in os.walk(negativeTruthful, topdown=True):
        negative_truthful_docs += [os.path.join(root, file) for file in files]
        ##negative_truthful_words.remove(['.DS_Store'])
    # print(negative_truthful_words)

    deceptive_docs = list()
    deceptive_docs = positive_deceptive_docs + negative_deceptive_docs
    # print(deceptive)

    truthful_docs = list()
    truthful_docs = positive_truthful_docs + negative_truthful_docs
    # print(truthful)

    prior_positive = len(positive_docs) / np.float(len(positive_docs) + len(negative_docs))
    # print (prior_positive)

    prior_negative = len(negative_docs) / np.float(len(positive_docs) + len(negative_docs))
    # print (prior_negative)

    prior_deceptive = len(deceptive_docs) / np.float(len(truthful_docs) + len(deceptive_docs))
    # print (prior_deceptive)

    prior_truthful = len(truthful_docs) / np.float(len(truthful_docs) + len(deceptive_docs))
    # print (prior_truthful)

    priorProbabilities = {
        'priorPos': np.float(len(positive_docs)) / np.float(len(positive_docs) + len(negative_docs)),
        'priorNeg': np.float(len(negative_docs)) / np.float(len(positive_docs) + len(negative_docs)),
        'priorDec': np.float(len(deceptive_docs)) / np.float(len(truthful_docs) + len(deceptive_docs)),
        'priorTru': np.float(len(truthful_docs)) / np.float(len(truthful_docs) + len(deceptive_docs))
    }

    wordDetails = {}
    for word in uniqueWordCount.keys():
        total = uniqueWordCount[word]
        positive = positiveWordToCount[word] if word in positiveWords else 0
        negative = negativeWordCount[word] if word in negativeWords else 0
        deceptive = deceptiveWordToCount[word] if word in deceptiveWords else 0
        truthful = truthfulWordToCount[word] if word in truthfulWords else 0

        wordDetails[word] = {'total': total,
                             'positive': positive,
                             'negative': negative,
                             'deceptive': deceptive,
                             'truthful': truthful,
                             'posProbability': findProbability(positive, len(uniqueWordCount), len(positiveWords)),
                             'negProbability': findProbability(negative, len(uniqueWordCount), len(negativeWords)),
                             'deceptiveProbability': findProbability(deceptive, len(uniqueWordCount),
                                                                     len(deceptiveWords)),
                             'truthfulProbability': findProbability(truthful, len(uniqueWordCount), len(truthfulWords))
                             }
    return wordDetails, priorProbabilities


if __name__ == "__main__":
    modelFile = "nbmodel.txt"
    inputPath = str(sys.argv[1])
    wordDetails, priorProbabilities = readpath(inputPath)
    writeNModel(wordDetails, priorProbabilities)

