import numpy as np
import os
import porter as port


def load_vocabulary(filename):
    # Load the vocabulary and returns a dictionary mapping words to numerical indices.
    f = open(filename)
    n = 0
    voc = {}
    for w in f.read().split():
        voc[w] = n
        n += 1
    f.close()
    return voc


def get_classes():
    f = open("classes.txt")
    classes = f.read().split("\n")
    classes.sort()
    f.close()
    print(classes)
    return classes


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


def remove_punctuation(text):
    # Replace punctuation symbols with spaces.
    punctuation_ = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for p in punctuation_:
        text = text.replace(p, " ")
    return text


def clean_data(text):
    # Read the file and returns a list of words.
    words = []
    stemmer = port.PorterStemmer()
    stopwords = read_stopwords()
    text = remove_punctuation(text.lower())

    # separate the document in words
    for w in text.split():
        if len(w) > 2 and w not in stopwords:
            w = stemmer.stem(w, 0, len(w) - 1)
            words.append(w)
    return words


def process_directory(path, voc):
    all_features = []
    all_labels = []
    classes = get_classes()
    for label, class_ in enumerate(classes):
        medical_files = os.listdir(path)

        for f in medical_files:
            if class_ in f:
                file_name = os.path.join(path, f)
                f = open(file_name, encoding="utf8")
                text = f.read()
                f.close()
                new_text = clean_data(text)
                # Start with all zeros
                bow = np.zeros(len(voc))
                for w in text.split():
                    # If the word is the vocabulary...
                    if w in voc:
                        # ...increment the proper counter.
                        index = voc[w]
                        bow[index] += 1
                all_features.append(bow)
                all_labels.append(label)

    X = np.stack(all_features, 0)
    Y = np.array(all_labels)
    return X, Y


def detailed_description():
    # The script compute the BoW representation of all the documents
    voc = load_vocabulary("vocabulary_train.txt")

    X, Y = process_directory("train", voc)
    X = X[:, :2000]
    print("train", X.shape, Y.shape)
    data = np.concatenate([X, Y[:, None]], 1)
    np.savetxt("train.txt.gz", data)

    voc = load_vocabulary("vocabulary_test.txt")
    X, Y = process_directory("test", voc)
    X = X[:, :2000]
    print("test", X.shape, Y.shape)
    data = np.concatenate([X, Y[:, None]], 1)
    np.savetxt("test.txt.gz", data)

    voc = load_vocabulary("vocabulary_validation.txt")
    X, Y = process_directory("validation", voc)
    X = X[:, :2000]
    print("validation", X.shape, Y.shape)
    data = np.concatenate([X, Y[:, None]], 1)
    np.savetxt("validation.txt.gz", data)


def short_description():
    # The script compute the BoW representation of all the documents
    voc = load_vocabulary("vocabulary_train_short.txt")

    X, Y = process_directory("train", voc)
    X = X[:, :1000]
    print("train", X.shape, Y.shape)
    data = np.concatenate([X, Y[:, None]], 1)
    np.savetxt("train_short.txt.gz", data)

    voc = load_vocabulary("vocabulary_test_short.txt")
    X, Y = process_directory("test", voc)
    X = X[:, :1000]
    print("test", X.shape, Y.shape)
    data = np.concatenate([X, Y[:, None]], 1)
    np.savetxt("test_short.txt.gz", data)

    voc = load_vocabulary("vocabulary_validation_short.txt")
    X, Y = process_directory("validation", voc)
    X = X[:, :1000]
    print("validation", X.shape, Y.shape)
    data = np.concatenate([X, Y[:, None]], 1)
    np.savetxt("validation_short.txt.gz", data)


def main():
    # to read the detailed description, as in the whole document
    # detailed_description()

    # to read only the short description, as in only the first line
    short_description()


if __name__ == "__main__":
    main()
