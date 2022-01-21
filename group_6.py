import nltk

# import ssl
# try:
#    _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#    pass
# else:
#    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
from starter import my_corpus

print(nltk.__version__)
from nltk.tokenize import TreebankWordTokenizer
import re
import math
from nltk.corpus import stopwords


# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('inaugural')
class encocde_decode():
    def __init__(self, params):
        super().__init__()

        self.params = params
        print('setting parameters')

    def int_finder(self, dictionary, word):
        for idx, val in dictionary.items():
            if word == val:
                return idx

    def encode_as_ints(self, req_dictionary, sequence):

        int_represent = []
        to_encode = sequence.split()
        print('encode this sequence: %s' % sequence)
        print('as a list of integers.')
        for word in to_encode:
            if self.int_finder(req_dictionary, word) is None:
                int_represent.append(self.int_finder(req_dictionary,"<UNK>"))
            else:
                int_represent.append(self.int_finder(req_dictionary, word))
        return int_represent

    def encode_as_text(self,req_dictionary, int_represent):

        text = ''

        print('encode this list', int_represent)
        print('as a text sequence.')
        for i in int_represent:
            text += req_dictionary[i] + ' '
        return text

def main():
    corpus = my_corpus(None)
    my_corpus.corpus_construction(corpus, "source_text.txt")
    unn_data, dictionary = my_corpus.summary_statistics(corpus, 'train_corpus.txt', True)
    my_corpus.print_stats(corpus)
    c = list(dictionary.keys())
    req_dictionary = dict(enumerate(c))
    a = encocde_decode(None)
    text = input('Please enter a test sequence to encode and recover: ')
    print(' ')
    ints = a.encode_as_ints(req_dictionary,text)
    print(' ')
    print('integer encoding: ', ints)

    print(' ')
    text = a.encode_as_text(req_dictionary,ints)
    print(' ')
    print('this is the encoded text: %s' % text)


if __name__ == "__main__":
    main()