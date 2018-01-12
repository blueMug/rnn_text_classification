import mxnet as mx
import argparse
import logging
from data_io import TestCorpus, CorpusIter
import util

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="CNN for text classification evaluation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test', type=str, default='',
                    help='test set')
parser.add_argument('--output', type=str, default='',
                    help='output file, if null, output at the console')
parser.add_argument('--evaluation', action='store_true',
                    help='if calculate precision rate, if True, test set file must contain labels, like train set')
parser.add_argument('--config', type=str, default='',
                    help='config file, denote labels')
parser.add_argument('--vocab', type=str, default='./data/vocab.pkl',
                    help='vocab file path for generation')
parser.add_argument('--model-name', type=str, default='checkpoint',
                    help='model name')
parser.add_argument('--max-length', type=int, default=100,
                    help='max sentence length')
parser.add_argument('--gpus', type=str, default='',
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ')
parser.add_argument('--checkpoint', type=int, default=0,
                    help='max num of epochs')
parser.add_argument('--batch-size', type=int, default=1,
                    help='the batch size.')


def accuracy(yhat, y):
    if len(y) == 0 or len(y) != len(yhat):
        return 0
    count = 0
    for yp, yt in zip(yhat, y):
        if yp == yt:
            count += 1
    return count * 1.0 / len(y)

if __name__ == '__main__':
    # parse args
    args = parser.parse_args()
    ctx = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]

    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix='checkpoint/'+args.model_name, epoch=args.checkpoint)
    if args.evaluation:
        test_corpus = TestCorpus(args.test, args.vocab, args.max_length, args.config)
        test_corpus_iter = CorpusIter(test_corpus.x_test, test_corpus.x_test_len, test_corpus.y_test,
                                      args.batch_size, args.max_length)
        mod = mx.mod.Module(symbol=sym, data_names=test_corpus_iter.data_names,
                            label_names=test_corpus_iter.label_names, context=ctx)

        mod.bind(for_training=False, data_shapes=test_corpus_iter.provide_data,
                 label_shapes=test_corpus_iter.provide_label)

    else:
        test_corpus = TestCorpus(args.test, args.vocab, args.max_length)
        test_corpus_iter = CorpusIter(test_corpus.x_test, test_corpus.x_test_len, None,
                                      args.batch_size, args.max_length, False)
        mod = mx.mod.Module(symbol=sym, data_names=test_corpus_iter.data_names,
                            label_names=test_corpus_iter.label_names, context=ctx)
        mod.bind(for_training=False, data_shapes=test_corpus_iter.provide_data)
    mod.set_params(arg_params, aux_params)

    prob = mod.predict(test_corpus_iter)
    labels_matrix = prob.argmax(axis=1)
    labels_index = [int(label.asscalar()) for label in labels_matrix]
    label_name = util.read_txt(args.config)
    labels = [label_name[label] for label in labels_index]

    result = ['predict label <> sentence']
    for label, sentence in zip(labels, test_corpus.contents):
        result.append(label + ' <> ' + sentence)
    output = args.output

    if args.evaluation:  # calculate precision rate
        result[0] = 'predict label <> true label <> sentence'

    if output == '':
        for line in result:
            print(line)
    else:
        util.save_to_txt(output, result)

    if args.evaluation:  # calculate precision rate
        # logger.info('accuracy: %f' % accuracy(yhat=labels, y=true_labels))
        logger.info(mod.score(test_corpus_iter, eval_metric=mx.metric.Accuracy()))
    logger.info('finished...')
