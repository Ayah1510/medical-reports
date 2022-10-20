import collections
import os
import porter as port


def remove_punctuation(text):
    # Replace punctuation symbols with spaces.
    punctuation_ = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for p in punctuation_:
        text = text.replace(p, " ")
    return text


def read_document(filename):
    # Read the file and returns a list of words.
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    words = []
    stemmer = port.PorterStemmer()
    text = remove_punctuation(text.lower())
    # separate the document in words
    for w in text.split():
        if len(w) > 2:
            w = stemmer.stem(w, 0, len(w) - 1)
            words.append(w)
    return words


def read_first_line_document(filename):
    # Read the file and returns a list of words.
    f = open(filename, encoding="utf8")
    text = f.readline().rstrip()
    f.close()
    words = []
    stemmer = port.PorterStemmer()
    text = remove_punctuation(text.lower())
    # separate the document in words
    for w in text.split():
        if len(w) > 2:
            w = stemmer.stem(w, 0, len(w) - 1)
            words.append(w)
    return words


def read_stopwords():
    # Read the file and returns a list of words.
    f = open("stopwords.txt", encoding="utf8")
    text = f.read()
    f.close()
    words = []
    # separate the document in words
    for w in text.split():
        words.append(w)
    return words


def write_vocabulary(voc, filename, n):
    # Write the n most frequent words to a file.
    stopwords = read_stopwords()
    f = open(filename, "w")
    for word, count in sorted(voc.most_common(n)):
        if word not in stopwords:
            print(word, file=f)
    f.close()


def detailed_description():
    # The script reads all the documents in the train directory,
    # then forms a vocabulary, writes it to the 'vocabulary.txt' file.
    voc = collections.Counter()
    for f in os.listdir("train"):
        voc.update(read_document("train/" + f))
    write_vocabulary(voc, "vocabulary_train.txt", 2500)

    voc = collections.Counter()
    for f in os.listdir("test"):
        voc.update(read_document("test/" + f))
    write_vocabulary(voc, "vocabulary_test.txt", 2500)

    voc = collections.Counter()
    for f in os.listdir("validation"):
        voc.update(read_document("validation/" + f))
    write_vocabulary(voc, "vocabulary_validation.txt", 2500)


def short_description():
    # The script reads all the documents in the train directory,
    # then forms a vocabulary, writes it to the 'vocabulary.txt' file.
    voc = collections.Counter()
    for f in os.listdir("train"):
        voc.update(read_first_line_document("train/" + f))
    write_vocabulary(voc, "vocabulary_train_short.txt", 1500)

    voc = collections.Counter()
    for f in os.listdir("test"):
        voc.update(read_first_line_document("test/" + f))
    write_vocabulary(voc, "vocabulary_test_short.txt", 1500)

    voc = collections.Counter()
    for f in os.listdir("validation"):
        voc.update(read_first_line_document("validation/" + f))
    write_vocabulary(voc, "vocabulary_validation_short.txt", 1500)


def main():
    # to read the detailed description, as in the whole document
    # detailed_description()

    # to read only the short description, as in only the first line
    short_description()


if __name__ == "__main__":
    main()
