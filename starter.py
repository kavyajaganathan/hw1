import nltk

# import ssl
# try:
#    _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#    pass
# else:
#    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
print(nltk.__version__)
from nltk.tokenize import TreebankWordTokenizer
import re
import math
from nltk.corpus import stopwords
from collections import Counter
from nltk import pos_tag
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('inaugural')


class my_corpus():
    def __init__(self, params):
        super().__init__()

        self.params = params
        print('setting parameters')

    def corpus_construction(self, data_file):
        with open(data_file, 'r') as file:
            lines = file.read().replace('\n', '')
        # convert all text to lowercase
        lines = lines.lower()
        # replace Replace (i) years, (ii) decimals, (iii) date days, (iv) integers and
        # (v) all other numbers with <year>, <decimal>,<days>,<integer>and<other>tags in this lower text
        regex_year = re.compile(r'(?:(?:18|19|20|21)[0-9]{2})')
        regex_decimal = re.compile(r'\d*\.\d+')
        regex_integer = re.compile(r'[+-]?\b[0-9]+')
        regex_date_1 = re.compile(r'<integer>\s\w+\s<year>')
        regex_date_2 = re.compile(r'<integer>\s\w+,\s<year>')
        regex_date_3 = re.compile(r'<integer>\s\w+\s\s\[o.s.\s<integer>\s\w+\s\]\s<year>')
        regex_date_4 = re.compile(r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)')
        regex_other = re.compile(r'(?=(?:\D*\d){10}(?:(?:\D*\d){3})?$)[\d-]+')
        lines = regex_year.sub("<year>", lines)
        print("Years done")
        lines = regex_decimal.sub("<decimal>", lines)
        print("Decimals done")
        lines = regex_integer.sub("<integer>", lines)
        print("Integers done")
        lines = regex_date_1.sub("<day>", lines)
        lines = regex_date_2.sub("<day>", lines)
        lines = regex_date_3.sub("<day>", lines)
        lines = regex_date_4.sub("<day>", lines)
        print("Dates done")
        #lines = regex_other.sub("<other>", lines)
        #print("Other done")
        # splitting data
        file_length = len(lines)
        train_len = math.floor(0.80 * file_length)
        val_len = math.ceil(0.10 * file_length)

        train = lines[0:train_len]
        val = lines[train_len:train_len + val_len]
        test = lines[train_len + val_len:]

        # tokenize the data
        tokenizer = TreebankWordTokenizer()
        tokenizer.PARENS_BRACKETS = [re.compile(r'[\]\[\(\)\{\}]'), r' \g<0> ']
        train_file = tokenizer.tokenize(train)
        val_file = tokenizer.tokenize(val)
        test_file = tokenizer.tokenize(test)
        print("Tokenized")

        # train, text and val file generation
        textfile = open("train_corpus.txt", "w")
        for element in train_file:
            textfile.write(element + "\n")
        textfile.close()
        print("Train file created")
        textfile = open("val_corpus.txt", "w")
        for element in val_file:
            textfile.write(element + "\n")
        textfile.close()
        print("Val file created")
        textfile = open("test_corpus.txt", "w")
        for element in test_file:
            textfile.write(element + "\n")
        textfile.close()
        print("Test file created")
        return (train_file, test_file, val_file)

    def summary_statistics(self,train_file,test_file,val_file):
        with open('train_corpus.txt') as f:
            lines = [line.rstrip() for line in f]
        # term-frequency dictionary creation
        frequency_dict = dict()
        for token in lines:
            if token in frequency_dict:
                frequency_dict[token] = frequency_dict[token] + 1
            else:
                frequency_dict[token] = 1
        # thresholding to 3. All below 3 are replaced as <unk>
        to_remove = []
        no_of_out_of_vocab = 0
        tot_no_of_unk = 0
        original_size = len(frequency_dict.keys())
        for key, value in frequency_dict.items():
            if value < 3:
                tot_no_of_unk += value
                no_of_out_of_vocab += 1
                to_remove.append(key)
        #print(tot_no_of_unk)
        for k in to_remove:
            frequency_dict.pop(k)
        #print(tot_no_of_unk)
        frequency_dict['<UNK>'] = tot_no_of_unk
        with open('test_corpus.txt') as f:
            test_lines = [line.rstrip() for line in f]
        no_of_types_unk = 0
        for token in test_lines:
            if token not in frequency_dict.keys():
                no_of_types_unk += 1
        #print(no_of_types_unk)
        no_of_stop_words = 0
        stop_words = set(stopwords.words('english'))
        for words in lines:
            if words in stop_words:
                no_of_stop_words += 1
        #Custom Metrics
        # Custom Metrics 1:
        from nltk import pos_tag
        tokens_tag = pos_tag(lines)
        from collections import Counter
        counts = Counter(tag for word, tag in tokens_tag)
        #print(counts)
        # Custom Metrics 2:
        from nltk.corpus import inaugural
        total_lens = 0
        for i, sentence in enumerate(inaugural.sents(fileids='/Users/kavyajaganathan/Downloads/hw1/source_text.txt')):
            total_lens += len(sentence)
        #print(total_lens / i)
        print("Summary Statistics:")
        print("Number of Tokens in each split:")
        print("Train:",len(train_file))
        print("Validation:",len(val_file))
        print("Test:",len(test_file))
        print("\n")
        print("Vocabulary Size:",original_size)
        print("Number of <UNK> tokens:",tot_no_of_unk)
        print("Number of out of vocabulary words:",no_of_out_of_vocab)
        print("Number of types mapped to UNK:",no_of_types_unk)
        print("Number of Stop Words:",no_of_stop_words)
        print("\n")
        print("Custom Metric 1: POS Tagging:")
        for k, v in counts.items():
            print(k, v)
        print("Custom Metric 2: Average Sentence Length:")
        print(total_lens / i)

    def encode_as_ints(self, sequence):

        int_represent = []

        print('encode this sequence: %s' % sequence)
        print('as a list of integers.')

        return (int_represent)

    def encode_as_text(self, int_represent):

        text = ''

        print('encode this list', int_represent)
        print('as a text sequence.')

        return (text)


def main():
    corpus = my_corpus(None)
    train_corpus_file,test_corpus_file,val_corpus_file = my_corpus.corpus_construction(corpus, "source_text.txt")
    my_corpus.summary_statistics(corpus,train_corpus_file,test_corpus_file,val_corpus_file)
    text = input('Please enter a test sequence to encode and recover: ')
    print(' ')
    ints = corpus.encode_as_ints(text)
    print(' ')
    print('integer encoding: ', ints)

    print(' ')
    text = corpus.encode_as_text(ints)
    print(' ')
    print('this is the encoded text: %s' % text)


if __name__ == "__main__":
    main()

    # word_tokenize("Hi this is Kavya.")
