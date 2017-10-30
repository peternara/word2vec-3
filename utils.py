import collections
import zipfile
import numpy as np
import matplotlib.pyplot as plt

vocabulary_size = 50000

# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = f.read(f.namelist()[0]).decode('utf-8').split()
  return data

vocabulary = read_data("text8.zip")
print('Data size', len(vocabulary))

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

# >>> data[0:10]
# [5236, 3082, 12, 6, 195, 2, 3136, 46, 59, 156]
#
# >>> count[0:10]
# [['UNK', 418391], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764), 
#  ('in', 372201), ('a', 325873), ('to', 316376), ('zero', 264975), ('nine',250430)]
#
# dictionary
# { 'UNK' -> 0, 'the' -> 1, 'of' -> 2, 'and' -> 3, 'one' -> 4 }
#
# reverse_dictionary
# { 0 -> 'UNK', 1 -> 'the', 2 -> 'of', 3 -> 'and', 4 -> 'one' }
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.

def build_batch(meth, data, batch_size, window_size):
  '''
  meth = 'cbow' or 'skip'

  window_size: the actual size of windows is window_size+1, if meth is 'skip' and the given window_size is even.

  returns data_batch, labl_bath
  
  when meth = 'cbow'
    data_batch: [iter, batch_size, window_size]
    labl_batch: [iter, batch_size, 1]

  when meth = 'skip'
    data_batch: [iter, batch_size, 1]
    labl_batch: [iter, batch_size, window_size]
  '''

  if meth not in ['cbow', 'skip']:
    raise ValueError('invalid value for meth')

  data_batch = []
  labl_batch = []

  if meth == 'cbow':
    rng = range(len(data) - window_size)
  elif meth == 'skip':
    window_size = window_size // 2  # window size to one side
    rng = range(window_size, len(data) - window_size - 1)

  for ind in rng:
    if meth == 'cbow':
      words = data[ind:ind+window_size]
      labls = data[ind+window_size]
    elif meth == 'skip':
      words = data[ind]
      labls = data[ind-window_size:ind] + data[ind+1:ind+window_size+1]
    data_batch.append(np.array(words))
    labl_batch.append(labls)

  length_of_data = len(data_batch) // batch_size * batch_size
  data_batch = data_batch[0:length_of_data]
  labl_batch = labl_batch[0:length_of_data]

  if meth == 'cbow':
    data_batch = np.array(data_batch).reshape([-1, batch_size, window_size])
    labl_batch = np.array(labl_batch).reshape([-1, batch_size, 1])
  elif meth == 'skip':
    data_batch = np.array(data_batch).reshape([-1, batch_size, 1])
    labl_batch = np.array(labl_batch).reshape([-1, batch_size, window_size*2]) 

  return data_batch, labl_batch

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)
