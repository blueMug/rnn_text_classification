import os
import mxnet as mx
import argparse
import logging
from data_io import Corpus
from data_io import CorpusIter


parser = argparse.ArgumentParser(description="RNN for text classification",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train', type=str, default='',
                    help='train set')
parser.add_argument('--validate', type=str, default='',
                    help='validate set')
parser.add_argument('--config', type=str, default='',
                    help='config file, denote labels')
parser.add_argument('--vocab', type=str, default='./data/vocab.pkl',
                    help='vocab file path for generation')
parser.add_argument('--model-name', type=str, default='checkpoint',
                    help='model name')
parser.add_argument('--num-embed', type=int, default=300,
                    help='embedding layer size')
parser.add_argument('--hidden-num', type=int, default=512,
                    help='embedding layer size')
parser.add_argument('--max-length', type=int, default=100,
                    help='max sentence length')
parser.add_argument('--gpus', type=str, default='',
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ')
parser.add_argument('--kv-store', type=str, default='local',
                    help='key-value store type')
parser.add_argument('--num-epochs', type=int, default=150,
                    help='max num of epochs')
parser.add_argument('--batch-size', type=int, default=50,
                    help='the batch size.')
parser.add_argument('--optimizer', type=str, default='rmsprop',
                    help='the optimizer type')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='initial learning rate')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout rate')
parser.add_argument('--disp-batches', type=int, default=50,
                    help='show progress for every n batches')
parser.add_argument('--save-period', type=int, default=10,
                    help='save checkpoint for every n epochs')
parser.add_argument('--log-name', type=str, default='./rnn_text_classification.log',
                    help='log file path')
# parse args
args = parser.parse_args()

fmt = '%(asctime)s:filename %(filename)s: lineno %(lineno)d:%(levelname)s:%(message)s'
logging.basicConfig(format=fmt, filemode='a+', filename=args.log_name, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def save_model(checkname):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    return mx.callback.do_checkpoint("checkpoint/"+checkname, args.save_period)


def sym_gen(num_embed, vocab_size, sequence_length, num_label, hidden_num, dropout):
    input_x = mx.sym.Variable('sequence')
    input_x_len = mx.sym.Variable('sequence_len')
    input_y = mx.sym.Variable('label')

    # embedding layer
    embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')

    rnn_input = mx.sym.reshape(data=embed_layer, shape=(-1, sequence_length, num_embed))
    cells = mx.rnn.SequentialRNNCell()
    cells.add(mx.rnn.BidirectionalCell(mx.rnn.LSTMCell(num_hidden=hidden_num // 2, prefix='lstm_1l_'),
                                       mx.rnn.LSTMCell(num_hidden=hidden_num // 2, prefix='lstm_1r_')))
    cells.add(mx.rnn.LSTMCell(num_hidden=hidden_num, prefix='lstm_2_'))
    # (batch_size, seq_len, hidden_num)
    outputs, states = cells.unroll(length=sequence_length, inputs=rnn_input, merge_outputs=True)

    # weighted average hidden output
    att_weigth = mx.sym.Variable('att_weight')
    att_bias = mx.sym.Variable('att_bias')
    # mx.sym.dot(mx.sym.reshape(data=outputs, shape=(-1, num_filter)), att_weigth)
    # (batch_size * seq_len, hidden_num)
    att_act_output = mx.sym.Activation(data=mx.sym.reshape(data=outputs, shape=(-1, hidden_num)), act_type='tanh')
    # (batch_size * seq_len, 1)
    att_fc_output = mx.sym.FullyConnected(data=att_act_output, num_hidden=1, weight=att_weigth, bias=att_bias)
    # (batch_size, seq_len, 1)
    att_mask = mx.sym.reshape(data=att_fc_output, shape=(-1, sequence_length, 1))
    # (seq_len, batch_size, 1)
    att_mask = mx.sym.swapaxes(data=att_mask, dim1=0, dim2=1)
    attention_scores = mx.sym.SequenceMask(data=att_mask,
                                           use_sequence_length=True,
                                           sequence_length=input_x_len,
                                           value=-99999999.)
    # (batch_size, seq_len, 1)
    attention_scores = mx.sym.swapaxes(data=attention_scores, dim1=0, dim2=1)
    # (batch_size, seq_len)
    attention_scores = mx.sym.reshape(data=attention_scores, shape=(0, 0))
    # (batch_size, seq_len)
    attention_probs = mx.sym.softmax(data=attention_scores, name='att_softmax')
    # (batch_size, seq_len, 1)
    attention_probs_expanded = mx.sym.expand_dims(data=attention_probs, axis=2)
    encode_output = mx.sym.batch_dot(lhs=outputs, rhs=attention_probs_expanded, transpose_a=True, name='att_batch_dot')
    encode_output = mx.sym.reshape(data=encode_output, shape=(0, 0))

    # dropout layer
    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=encode_output, p=dropout)
    else:
        h_drop = encode_output

    # fully connected
    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')

    fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)

    # softmax output
    sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')

    return sm, ('sequence', 'sequence_len', ), ('label',)


def train(symbol, train_iter, valid_iter, data_names, label_names, checkname, optimizer, learning_rate, ctx):

    module = mx.mod.Module(symbol, data_names=data_names, label_names=label_names, context=ctx)

    module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    module.init_params(mx.init.Xavier(rnd_type="gaussian", factor_type="avg", magnitude=2.5))
    lr_sch = mx.lr_scheduler.FactorScheduler(step=25000, factor=0.999)
    module.init_optimizer(
        optimizer=optimizer, optimizer_params={'learning_rate': learning_rate, 'lr_scheduler': lr_sch})

    # monitor each parameters
    # def norm_stat(d):
    #     return mx.nd.norm(d) / np.sqrt(d.size)
    # mon = mx.mon.Monitor(25000, norm_stat)

    module.fit(train_data=train_iter,
               eval_data=valid_iter,
               eval_metric='acc',
               kvstore=args.kv_store,
               # monitor=mon,
               num_epoch=args.num_epochs,
               batch_end_callback=mx.callback.Speedometer(args.batch_size, args.disp_batches),
               epoch_end_callback=save_model(checkname))


if __name__ == '__main__':

    ctx = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]

    corpus = Corpus(args.train, args.validate, args.config, args.vocab, args.max_length)

    train_iter = CorpusIter(corpus.x_train, corpus.x_train_len, corpus.y_train, args.batch_size, args.max_length)
    dev_iter = CorpusIter(corpus.x_dev, corpus.x_dev_len, corpus.y_dev, args.batch_size, args.max_length)

    # network symbol
    symbol, data_names, label_names = sym_gen(args.num_embed,
                                              corpus.vocab_size, args.max_length,
                                              num_label=corpus.n_class, hidden_num=args.hidden_num,
                                              dropout=args.dropout)
    # train rnn model
    train(symbol, train_iter, dev_iter, data_names, label_names, args.model_name, args.optimizer, args.lr, ctx)
