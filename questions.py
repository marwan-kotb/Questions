import nltk
import sys, os
from nltk.tokenize import word_tokenize
import string, math
import operator


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    
    data = dict()
    for file in os.listdir(directory):
        D = os.path.join(directory,file)
        with open(D) as f:
            data[file] = f.read().replace('\n', ' ')
    
    return data


def tokenize(document):
    

    w = nltk.word_tokenize(document)
    
    words = []
    
    for j in range(len(w)):
        w[j] = w[j].lower()
    

    for word in w:
        
        if word in string.punctuation or word in nltk.corpus.stopwords.words("english"):
            continue

        words.append(word)

    

    return words


def compute_idfs(documents):
    
    
    idfs = dict()
    for value in documents.values():
        for word in value:
            f = sum([word in value for value in documents.values()])
            idf = math.log(len(documents) / f)
            idfs[word] = idf
    
    return idfs
   


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    
    tfidfs = dict()
    
    for filename in files.keys():
        tfidfs[filename] = 0
        for word in query:
            if word in files[filename]:
                tf = files[filename].count(word)
                tfidfs[filename] +=  (tf * idfs[word])
        
    e = dict(sorted(tfidfs.items(), key=operator.itemgetter(1),reverse=True))
    q = list(e.keys())
    result = q[:n]
    return result

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    d = dict()
    for sentence, words in sentences.items():
        count = 0
        l = len(words)
        IDF = 0
        for word in query:
            if word in words:
                count += 1
                IDF += idfs[word]
        
        try:
            density = count/l
        except:
            density = 0

        d[sentence] = (IDF, density)
    e = dict(sorted(d.items(), key=operator.itemgetter(1),reverse=True))
    result = list(e.keys())[:n]
    return result



    
   
    



if __name__ == "__main__":
    main()
