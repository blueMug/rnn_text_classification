import mxnet as mx
import util
import numpy as np
from collections import Counter
import itertools
import logging
logger = logging.getLogger(__name__)


class Corpus:
    def __init__(self, train_file, validate_file, config, vocab_path, max_length):

        logger.info('Loading data...')

        x_train, x_train_len, y_train, vocab, vocab_inv, self.n_class = \
            self.load_data(train_file, config, max_length, None)
        self.sentence_size = x_train.shape[1]
        self.vocab_size = len(vocab)
        util.save_to_pickle(vocab_path, vocab)

        x_dev, x_dev_len, y_dev, _, _, _ = self.load_data(validate_file, config, max_length, vocab)

        # randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        x_train = x_train[shuffle_indices]
        x_train_len = x_train_len[shuffle_indices]
        y_train = y_train[shuffle_indices]

        # replicating random examples from pre-data
        # rest = batch_size - len(x_train) % batch_size
        # random_indices = np.random.randint(x_train.shape[0], size=rest)
        #
        # x_train = np.concatenate((x_train, x_train[random_indices, :]), axis=0)
        # x_train_len = np.concatenate((x_train_len, x_train_len[random_indices]), axis=0)
        # y_train = np.concatenate((y_train, y_train[random_indices]), axis=0)

        self.x_train = mx.nd.array(x_train)
        self.x_train_len = mx.nd.array(x_train_len)
        self.y_train = mx.nd.array(y_train)

        self.x_dev = mx.nd.array(x_dev)
        self.x_dev_len = mx.nd.array(x_dev_len)
        self.y_dev = mx.nd.array(y_dev)

        logger.info('Train/Valid split: %d/%d' % (len(y_train), len(y_dev)))
        logger.info('train shape: %(shape)s', {'shape': x_train.shape})
        logger.info('valid shape: %(shape)s', {'shape': x_dev.shape})

    def load_data(self, train_file, config, max_length, vocabulary=None):
        """
        Loads and preprocessed data for the MR dataset.
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        sentences, sen_lens, labels, n_class = self.load_data_and_labels(train_file, config, max_length)
        sentences_padded = util.pad_sentences(sentences, max_length)
        vocabulary_inv = None
        if vocabulary is None:
            vocabulary, vocabulary_inv = self.build_vocab(sentences_padded)
        x, y = self.build_input_data(sentences_padded, labels, vocabulary)
        x_len = np.array(sen_lens)
        return [x, x_len, y, vocabulary, vocabulary_inv, n_class]

    def load_data_and_labels(self, data_file, config, max_length):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        :param data_file:  training set, label <> sententce
        :param config:  describe label, it denotes a one-hot vector, the dimension == number of label
            one label a line. Example:   sports\neducation\nculture   sports:[1,0,0], education:[0,1,0], culture:[0,0,1]
        :param max_length: max sentence length
        :return:
        """
        trains = util.read_txt(data_file)
        label_dict = util.read_txt_to_dict(config)
        #
        n_class = len(label_dict)
        x_text = []
        x_len = []
        y_text = []
        for t in trains:
            line = t.split(' <> ')
            if len(line) < 2:
                continue
            cur_text = line[1].split()[:max_length]
            x_text.append(cur_text)
            x_len.append(len(cur_text))
            label_num = label_dict[line[0].strip()]
            y_text.append(label_num)

        return [x_text, x_len, y_text, n_class]

    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv.insert(0, '<unk>')
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def build_input_data(self, sentences, labels, vocabulary):
        """
        Maps sentencs and labels to vectors based on a vocabulary.
        """
        x = np.array([[vocabulary.get(word, 0) for word in sentence] for sentence in sentences])
        y = np.array(labels)
        return [x, y]


class CorpusIter(mx.io.DataIter):
    """
    An iterator that returns the a batch of sequence each time
    """
    def __init__(self, source, source_len, label, batch_size, max_seq_len, config=True):
        super(CorpusIter, self).__init__()
        self.config = config
        self.batch_size = batch_size
        self.data_names = ['sequence', 'sequence_len']
        self.label_names = ['label']
        self.provide_data = [mx.io.DataDesc(name='sequence', shape=(self.batch_size, max_seq_len), layout='NTC'),
                             mx.io.DataDesc(name='sequence_len', shape=(self.batch_size,), layout='NTC')]
        self._index = 0
        self._source = source
        self._source_len = source_len
        self._next_data = None
        self._next_data_len = None

        if self.config:
            self._label = label
            self._next_label = None
            self.provide_label = [mx.io.DataDesc(name='label', shape=(self.batch_size,), layout='NTC')]

    def iter_next(self):
        i = self._index
        interval = self.batch_size
        if i+interval > self._source.shape[0] - 1:
            return False
        self._next_data = self._source[i:i+interval]
        self._next_data_len = self._source_len[i:i+interval]
        if self.config:
            self._next_label = self._label[i:i+interval]
        self._index += self.batch_size
        return True

    def next(self):
        if self.iter_next():
            if self.config:
                return mx.io.DataBatch([self._next_data, self._next_data_len], [self._next_label],
                                       pad=0, index=None, provide_data=self.provide_data, provide_label=self.provide_label)
            else:
                return mx.io.DataBatch([self._next_data, self._next_data_len],
                                       pad=0, index=None, provide_data=self.provide_data)
        else:
            raise StopIteration

    def reset(self):
        self._index = 0
        self._next_data = None
        self._next_data_len = None
        if self.config:
            self._next_label = None


class TestCorpus:
    def __init__(self, test_file, vocab_path, max_length, config=None):

        logger.info('Loading data...')

        x_test, x_test_len, self.contents, labels, y_test = self.load_test_data(test_file, max_length, vocab_path, config)
        self.sentence_size = x_test.shape[1]

        self.x_test = mx.nd.array(x_test)
        self.x_test_len = mx.nd.array(x_test_len)
        if config:
            self.y_test = mx.nd.array(y_test)

    def load_test_data(self, test_file, max_length, vocabulary=None, config=None):
        """
        Loads and preprocessed data for the MR dataset.
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        contents = util.read_txt(test_file)
        lines = [line for line in contents]
        labels = []
        x_text = []
        x_text_len = []
        y_text = None
        if config is None:
            for s in lines:
                cur_line = s.split()[:max_length]
                x_text.append(cur_line)
                x_text_len.append(len(cur_line))
        else:
            y = []
            label_dict = util.read_txt_to_dict(config)
            for line in lines:
                line = line.split(' <> ')
                cur_line = line[1].split()[:max_length]
                x_text.append(cur_line)
                x_text_len.append(len(cur_line))
                labels.append(line[0])
                label_num = label_dict[line[0].strip()]
                y.append(label_num)
            y_text = np.array(y)

        sentences_padded = util.pad_sentences(x_text, max_length)
        vocabulary = util.read_pickle(vocabulary)
        x = np.array([[vocabulary.get(word, 0) for word in sentence] for sentence in sentences_padded])
        x_text_len = np.array(x_text_len)

        return x, x_text_len, contents, labels, y_text
