from prepare_data import *
import time
from model_helper import *


def linear(input_, output_size, scope=None):

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class CNNClassfier(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.num_classes = config["n_class"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.filter_sizes = config["filter_sizes"]
        self.num_filters = config["num_filters"]
        self.l2_reg_lambda = config["l2_reg_lambda"]
        self.learning_rate = config["learning_rate"]

        self.x = tf.placeholder(tf.int32, [None, self.max_len], name="input_x")
        self.label = tf.placeholder(tf.int32, [None], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    def build_graph(self):
        print("building graph")
        l2_loss = tf.constant(0.0)
        with tf.variable_scope("discriminator"):
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.x) 
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            pooled_outputs = list()

            for filter_size, filter_num in zip(self.filter_sizes, self.num_filters):
                with tf.name_scope("cov2d-maxpool%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, filter_num]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")

                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.max_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")  
                    pooled_outputs.append(pooled)
            total_filters_num = sum(self.num_filters)

            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_filters_num])  

            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)


            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.keep_prob)

            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([total_filters_num, self.num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.prediction = tf.cast(tf.argmax(self.ypred_for_auc, 1), dtype=tf.int32)+1

            with tf.name_scope("loss"):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.label)
                self.loss = losses + self.l2_reg_lambda * l2_loss
            with tf.name_scope("accuracy"):
                self.accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(self.prediction, self.label), tf.float32))

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        print("graph built successfully!")


if __name__ == '__main__':

    x_train, y_train = load_data("train.csv", sample_ratio=1, one_hot=False)
    x_test, y_test = load_data("test.csv", one_hot=False)

    x_train, x_test, vocab_size = \
        data_preprocessing_v2(x_train, x_test, max_len=2000)
    print("train size: ", len(x_train))
    print("vocab size: ", vocab_size)

    x_test, x_dev, y_test, y_dev, dev_size, test_size = \
        split_dataset(x_test, y_test, 0.1)
    print("Validation Size: ", dev_size)

    config = {
        "max_len": 2000,
        "vocab_size": vocab_size,
        "embedding_size": 64,
        "learning_rate": 1e-2,
        "l2_reg_lambda": 1e-3,
        "batch_size": 4,
        "n_class": 4,
        "filter_sizes": [1, 2, 3, 4, 5, 10, 20, 50, 100, 120],
        "num_filters": [128, 256, 256, 256, 256, 128, 128, 128, 128, 256],
        "train_epoch": 50,
    }

    classifier = CNNClassfier(config)
    classifier.build_graph()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    sess.run(tf.global_variables_initializer())
    dev_batch = (x_dev, y_dev)
    start = time.time()
    cnn_file = open("cnn_file.txt","w")
    for e in range(config["train_epoch"]):

        t0 = time.time()
        print("Epoch %d start !" % (e + 1))
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, config["batch_size"]):
            return_dict = run_train_step(classifier, sess, (x_batch, y_batch))

        t1 = time.time()

        print("Train Epoch time:  %.3f s" % (t1 - t0))
        dev_acc = run_eval_step(classifier, sess, dev_batch)
        print("validation accuracy: %.3f " % dev_acc)
        cnn_file.write("Epoch:"+str(e+1)+" validation accuracy: " + str(dev_acc)+"\n")

    cnn_file.close()
    print("Training finished, time consumed : ", time.time() - start, " s")
    print("Start evaluating:  \n")
    cnt = 0
    test_acc = 0
    for x_batch, y_batch in fill_feed_dict(x_test, y_test, config["batch_size"]):
        acc = run_eval_step(classifier, sess, (x_batch, y_batch))
        test_acc += acc
        cnt += 1

    print("Test accuracy : %f %%" % (test_acc / cnt * 100))
