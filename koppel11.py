"""
Jakob Koehler & Tolga Buz
Reproduction of Koppel11
"""
# --- Imports:
from math import sqrt
import jsonhandler
import random
import argparse

# --- Parameters:
# n-gram size
n = 4
# length of feature list
featureLength = 20000
# Score threshold (needed for open set)
threshold = 0
# number of k repetitions
repetitions = 100
# minimum size of document
# (increases precision, but deteriorates recall,
# if there are many small documents)
min_len = 0
# candidates with less than this amount of words in training data are not
# attributed to
min_train_len = 500


def create_vector(s):
    """
    gets a string (e.g. Book), splits it into and returns a vector
    with all possible n-grams/features
    """

    vector = {}
    word_list = s.split()
    for word in word_list:
        if len(word) <= n:
            add(vector, word)
        else:
            for k in range(len(word) - n + 1):
                add(vector, word[k:k + n])
    return vector


def add(vector, ngram):
    """
    adds n-grams to our feature list-vector, if is not included yet
    (containing all possible n-grams/features)
    """

    if ngram in vector:
        vector[ngram] += 1
    else:
        vector[ngram] = 1


def select_features(vector):
    """
    selects the x most frequent n-grams/features (x=featureLength)
    to avoid a (possibly) too big feature list
    """
    return sorted(vector, key=vector.get, reverse=True)[:min(len(vector), featureLength)]


def create_feature_map(s, features):
    """
    creates Feature Map that only saves
    the features that actually appear more frequently than 0.
    Thus, the feature list needs less memory and can work faster
    """

    feature_map = {}
    vec = create_vector(s)
    for ngram in features:
        if ngram in vec:
            feature_map[ngram] = vec[ngram]
    return feature_map


def cos_sim(v1, v2):
    """
    calculates cosine similarity of two vectors v1 and v2.
    -> cosine(X, Y) = (X * Y)/(|X|*|Y|)
    """

    sp = float(0)
    len1 = 0
    len2 = 0
    for ngram in v1:
        len1 += v1[ngram] ** 2
    for ngram in v2:
        len2 += v2[ngram] ** 2
    len1 = sqrt(len1)
    len2 = sqrt(len2)
    for ngram in v1:
        if ngram in v2:
            sp += v1[ngram] * v2[ngram]
    return sp / (len1 * len2)


def minmax(v1, v2):
    """
    calculates minmax similarity of two vectors v1 and v2.
    -> minmax(X, Y) = sum(min(Xi, Yi))/sum(max(Xi, Yi))

    This baseline method will be used for further evaluation.
    """

    min_sum = 0
    max_sum = 0
    for ngram in v1:
        if ngram in v2:
            # ngram is in both vectors
            min_sum += min(v1[ngram], v2[ngram])
            max_sum += max(v1[ngram], v2[ngram])
        else:
            # ngram only in v1
            max_sum += v1[ngram]
    for ngram in v2:
        if ngram not in v1:
            # ngram only in v2
            max_sum += v2[ngram]
    if max_sum == 0:
        return 0
    return float(min_sum) / max_sum


def training(s):
    """
    Turns a given string into a n-gram vector
    and returns its feature list.
    """

    print("training...")
    vec = create_vector(s)
    print("selecting features...")
    feature_list = select_features(vec)
    print("done")
    return feature_list


def test_sim(x, y, feature_list, func):
    """
    args: two vectors, a feature list
    and func(to decide whether to use cosine or minmax similarity).

    uses create_feature_map and cos_sim or minmax
    and returns the similarity value of the two vectors
    """

    fx = create_feature_map(x, feature_list)
    fy = create_feature_map(y, feature_list)
    if func == 0:
        return cos_sim(fx, fy)
    else:
        return minmax(fx, fy)


def get_random_string(s, length):
    """
    Returns a random part of a string s
    that has a given length
    """

    word_list = s.split()
    r = random.randint(0, len(word_list) - length)
    return "".join(word_list[r:r + length])


if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser(
        description="Tira submission for PPM approach (koppel11)")
    parser.add_argument("-i", action="store", help="path to corpus directory")
    parser.add_argument("-o", action="store", help="path to output directory")
    args = vars(parser.parse_args())

    corpus_dir = args["i"]
    output_dir = args["o"]
    if corpus_dir is None or output_dir is None:
        parser.print_help()
        exit()

    candidates = jsonhandler.candidates
    unknowns = jsonhandler.unknowns
    jsonhandler.loadJson(corpus_dir)
    jsonhandler.loadTraining()

    texts = {}
    # texts = frozenset() would this work??
    corpus = ""
    print("loading texts for training")
    deletes = []
    for can in candidates:
        texts[can] = ""
        for file in jsonhandler.trainings[can]:
            texts[can] += jsonhandler.getTrainingText(can, file)
            # if frozenset() is used:
            # texts.add(jsonhandler.getTrainingText(can, file))
            print("text " + file + " read")
        if len(texts[can].split()) < min_train_len:
            del texts[can]
            deletes.append(can)
        else:
            corpus += texts[can]

    new_cans = []
    for can in candidates:
        if can not in deletes:
            new_cans.append(can)
    candidates = new_cans
    words = [len(texts[can].split()) for can in texts]
    min_words = min(words)
    print(min_words)

    fl = training(corpus)
    authors = []
    scores = []

    for file in unknowns:
        print("testing " + file)
        u_text = jsonhandler.getUnknownText(file)
        u_len = len(u_text.split())
        if u_len < min_len:
            authors.append("None")
            scores.append(0)
        else:
            wins = [0] * len(candidates)
            text_len = min(u_len, min_words)
            print(text_len)
            ustring = "".join(u_text.split()[:text_len])
            for i in range(repetitions):
                rfl = random.sample(fl, len(fl) // 2)
                sims = []
                for can in candidates:
                    can_string = get_random_string(texts[can], text_len)
                    sims.append(test_sim(can_string, ustring, rfl, 1))
                wins[sims.index(max(sims))] += 1
            score = max(wins) / float(repetitions)
            if score >= threshold:
                authors.append(candidates[wins.index(max(wins))])
                scores.append(score)
            else:
                authors.append("None")
                scores.append(score)

    print("storing answers")
    jsonhandler.storeJson(output_dir, unknowns, authors, scores)
