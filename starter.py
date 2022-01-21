import nltk
from tabulate import tabulate

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


# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('inaugural')


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
        regex_other = re.compile(r'(?=(?:\D*\d){10}(?:(?:\D*\d){3})?$)[\d-]+')
        regex_year = re.compile(r'(?:(?:18|19|20|21)[0-9]{2})')
        regex_decimal = re.compile(r'\d*\.\d+')
        regex_integer = re.compile(r'[+-]?\b[0-9]+')
        regex_date_1 = re.compile(r'<integer>\s\w+\s<year>')
        regex_date_2 = re.compile(r'<integer>\s\w+,\s<year>')
        regex_date_3 = re.compile(r'<integer>\s\w+\s\s\[o.s.\s<integer>\s\w+\s\]\s<year>')
        regex_date_4 = re.compile(r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)')
        lines = regex_other.sub("<other>", lines)
        print("Other done")
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

    def summary_statistics(self, data_file, train=False):
        with open(data_file) as f:
            lines = [line.rstrip() for line in f]
        # term-frequency dictionary creation
        # lines - holds keys of the dictionary you've created clarissa
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
        for k in to_remove:
            frequency_dict.pop(k)
        frequency_dict['<UNK>'] = tot_no_of_unk
        if not train:
            no_of_types_unk = 0
            for token in lines:
                if token not in frequency_dict.keys():
                    no_of_types_unk += 1
        no_of_stop_words = 0
        stop_words = set(stopwords.words('english'))
        for words in lines:
            if words in stop_words:
                no_of_stop_words += 1
        # Custom Metrics
        # Custom Metrics 1:
        from nltk import pos_tag
        tokens_tag = pos_tag(lines)
        from collections import Counter
        counts = Counter(tag for word, tag in tokens_tag)
        # Custom Metrics 2:
        text = ' '.join(lines)
        sentences = text.split(".")
        words = text.split(" ")
        if (sentences[len(sentences) - 1] == ""):
            avg = len(words) / len(sentences) - 1
        else:
            avg = len(words) / len(sentences)
        data = []
        data.append(len(lines))
        # Vocabulary Size
        data.append(original_size)
        # Num of UNK tokens
        data.append(tot_no_of_unk)
        # Number of OOV
        data.append(no_of_out_of_vocab)
        # Number of types mapped to UNK
        if train:
            data.append('x')
        else:
            data.append(no_of_types_unk)
        # Number of Stop Words
        data.append(no_of_stop_words)
        # Avg sentence length
        data.append(avg)
        # POS Tagging
        data.append(counts)

        return data, frequency_dict

    def print_stats(self):
        data_train, d = my_corpus.summary_statistics(self, 'train_corpus.txt', True)
        d_test, dt1 = my_corpus.summary_statistics(self, 'test_corpus.txt', False)
        d_val, dt_val = my_corpus.summary_statistics(self, 'val_corpus.txt', False)
        table = [['data', 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'Custom Metric 1- Average Sentence Length'],
                 ['Train'] + data_train[0:7], ['Validation'] + d_val[0:7],
                 ['Test'] + d_test[0:7]]
        print(tabulate(table, headers='firstrow', tablefmt='grid'))
        keys_train = []
        val_train = []
        keys_val = []
        val_val = []
        keys_test = []
        val_test = []
        for k, v in data_train[7].items():
            keys_train.append(k)
            val_train.append(v)
        for k, v in d_val[7].items():
            keys_val.append(k)
            val_val.append(v)
        for k, v in d_val[7].items():
            keys_test.append(k)
            val_test.append(v)
        table_train = [keys_train[0:10], val_train[0:10]]
        table_val = [keys_val[0:10], val_val[0:10]]
        table_test = [keys_test[0:10], val_test[0:10]]
        print('vii - Train')
        print(tabulate(table_train, headers='firstrow', tablefmt='grid'))
        print('vii - Validation')
        print(tabulate(table_val, headers='firstrow', tablefmt='grid'))
        print('vii - Test')
        print(tabulate(table_test, headers='firstrow', tablefmt='grid'))

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
    my_corpus.corpus_construction(corpus, "source_text.txt")
    my_corpus.print_stats(corpus)
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
