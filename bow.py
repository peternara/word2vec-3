import tensorflow as tf
from utils import data, reverse_dictionary, vocabulary_size, build_batch, plot_with_labels

BATCH_SIZE  = 256
WINDOW_SIZE = 3
NUM_STEPS   = 1
EMBEDDING_SIZE = 128

data_batch, labl_batch = build_batch(data, BATCH_SIZE, WINDOW_SIZE)

with tf.device('/gpu:0'):
    words_from_list = [tf.placeholder(tf.int32, shape=[BATCH_SIZE]) for _ in range(WINDOW_SIZE)]
    words_to   = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    i2h        = tf.Variable(tf.random_normal([vocabulary_size, EMBEDDING_SIZE], dtype=tf.float32), name='i2h')
    hb         = tf.Variable(0.0, dtype=tf.float32, name='hb')
    h2o        = tf.Variable(tf.random_normal([EMBEDDING_SIZE, vocabulary_size], dtype=tf.float32), name='h2o')
    ob         = tf.Variable(0.0, dtype=tf.float32, name='ob')

    # tf.nn.embedding_lookup(i2h, w) == tf.matmul(one_hot(w), i2h)
    words_pred = tf.matmul(sum([tf.nn.embedding_lookup(i2h, w) + hb for w in words_from_list]) / WINDOW_SIZE, h2o) + ob

    words_expt = tf.one_hot(words_to, vocabulary_size, on_value=1.0, off_value=0.0, dtype=tf.float32, name='one-hot-word-expt')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=words_expt, logits=words_pred))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.allow_soft_placement = True
config.log_device_placement = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    print('Training')
    for step in range(NUM_STEPS):
        print('Step ', step)
        average_loss = 0
        rng = range(data_batch.shape[0])
        for ind in rng:
            words_i = [data_batch[ind, :, wi] for wi in range(WINDOW_SIZE)]
            words_o = labl_batch[ind, :, 0]
            ph_feed = {i: d for i, d in zip(words_from_list, words_i)}
            ph_feed[words_to] = words_o
            _, lv = sess.run([optimizer, loss], feed_dict=ph_feed)
            if ind % 1000 == 0:
                print('\r  {}/{} - {}'.format(ind, len(rng), lv), end = '\r')
            average_loss += lv
        average_loss /= len(rng)
        print('Average loss at step ', step, ': ', average_loss)
    final_embeddings = tf.transpose(h2o).eval()

# pylint: disable=g-import-not-at-top
import pickle
with open("word-embedding.mat", "wb") as fi:
    pickle.dump(final_embeddings, fi)

# pylint: disable=g-import-not-at-top
from sklearn.manifold import TSNE

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
plot_only = min(500, vocabulary_size)
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)