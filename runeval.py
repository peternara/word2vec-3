import tensorflow as tf
import pickle
from utils import dictionary, reverse_dictionary

with open("word-embedding.mat", "rb") as f:
    emb1 = pickle.load(f)

with open("word-embedding-ref.mat", "rb") as f:
    emb2 = pickle.load(f)

import eval_tf
sess = tf.Session()
eval_tf.Word2VecEval(sess, emb1, dictionary, reverse_dictionary).eval()
eval_tf.Word2VecEval(sess, emb2, dictionary, reverse_dictionary).eval()

# import eval
# qs = eval.read_questions()
# eval.eval(emb1, qs)
# eval.eval(emb2, qs)