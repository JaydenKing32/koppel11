# META_filename - name of the meta-file.json
# GT_filename - name of the ground-truth.json
# OUT_filename - file to write the output in (answers.json)
# encoding - encoding of the texts (from json)
# language - language of the texts (from json)
# u_path - path of the 'unknown' dir in the corpus (from json)
# candidates - list of candidate author names (from json)
# unknowns - list of unknown filenames (from json)
# trainings - dictionary with lists of filenames of training texts for each author
#     {"candidate2":["file1.txt", "file2.txt", ...], "candidate2":["file1.txt", ...] ...}
# trueAuthors - list of true authors of the texts (from GT_filename json)
# corresponding to 'unknowns'

"""
EXAMPLE:

import jsonhandler

candidates = jsonhandler.candidates
unknowns = jsonhandler.unknowns
jsonhandler.load_json("test_corpus")

# If you want to do training:
jsonhandler.load_training()
for can in candidates:
    for file in jsonhandler.trainings[can]:
        # Get content of training file 'file' of candidate 'can' as a string with:
        # jsonhandler.get_training_text(can, file)

# Create lists for your answers (and scores)
authors = []
scores = []

# Get Parameters from json-file:
l = jsonhandler.language
e = jsonhandler.encoding

for file in unknowns:
    # Get content of unknown file 'file' as a string with:
    # jsonhandler.get_unknown_text(file)
    # Determine author of the file, and score (optional)
    author = "oneAuthor"
    score = 0.5
    authors.append(author)
    scores.append(score)

# Save results to json-file out.json (passing 'scores' is optional)
jsonhandler.store_json(unknowns, authors, scores)

# If you want to evaluate the ground-truth file
load_ground_truth()
# find out true author of document unknowns[i]:
# trueAuthors[i]
"""

import os
import json
import codecs

META_FILENAME = "meta-file.json"
OUT_FILENAME = "answers.json"
GT_FILENAME = "ground-truth.json"

# initialization of global variables
encoding = ""
language = ""
corpus_dir = ""
u_path = ""
candidates = []
unknowns = []
trainings = {}
trueAuthors = []


def load_json(corpus):
    """
    always run this method first to evaluate the meta json file. Pass the
    directory of the corpus (where meta-file.json is situated)
    """
    global corpus_dir, u_path, candidates, unknowns, encoding, language
    corpus_dir += corpus
    m_file = open(os.path.join(corpus_dir, META_FILENAME), "r")
    meta_json = json.load(m_file)
    m_file.close()

    u_path += os.path.join(corpus_dir, meta_json["folder"])
    encoding += meta_json["encoding"]
    language += meta_json["language"]
    candidates += [author["author-name"]
                   for author in meta_json["candidate-authors"]]
    unknowns += [text["unknown-text"] for text in meta_json["unknown-texts"]]


def load_training():
    """
    run this method next, if you want to do training (read training files etc)
    """
    for can in candidates:
        trainings[can] = []
        for subdir, dirs, files in os.walk(os.path.join(corpus_dir, can)):
            for doc in files:
                trainings[can].append(doc)


def get_training_text(can, filename):
    """
    get training text 'filename' from candidate 'can' (obtain values from
    'trainings', see example above)
    """
    d_file = codecs.open(os.path.join(corpus_dir, can, filename), "r", "utf-8")
    s = d_file.read()
    d_file.close()
    return s


def get_training_bytes(can, filename):
    """
    get training file as bytearray
    """
    d_file = open(os.path.join(corpus_dir, can, filename), "rb")
    b = bytearray(d_file.read())
    d_file.close()
    return b


def get_unknown_text(filename):
    """
    get unknown text 'filename' (obtain values from 'unknowns', see example above)
    """
    d_file = codecs.open(os.path.join(u_path, filename), "r", "utf-8")
    s = d_file.read()
    d_file.close()
    return s


def get_unknown_bytes(filename):
    """
    get unknown file as bytearray
    """
    d_file = open(os.path.join(u_path, filename), "rb")
    b = bytearray(d_file.read())
    d_file.close()
    return b


def store_json(path, texts, cans, scores=None):
    """
    run this method in the end to store the output in the 'path' directory as OUT_filename
    pass a list of filenames (you can use 'unknowns'), a list of your
    predicted authors and optionally a list of the scores (both must of
    course be in the same order as the 'texts' list)
    """
    answers = []
    if scores is None:
        scores = [1 for _ in texts]
    for i in range(len(texts)):
        answers.append(
            {"unknown_text": texts[i], "author": cans[i], "score": scores[i]})
    f = open(os.path.join(path, OUT_FILENAME), "w")
    json.dump({"answers": answers}, f, indent=2)
    f.close()


def load_ground_truth():
    """
    if you want to evaluate your answers using the ground-truth.json, load
    the true authors in 'trueAuthors' using this function
    """
    t_file = open(os.path.join(corpus_dir, GT_FILENAME), "r")
    t_json = json.load(t_file)
    t_file.close()

    global trueAuthors
    for i in range(len(t_json["ground-truth"])):
        trueAuthors.append(t_json["ground-truth"][i]["true-author"])
