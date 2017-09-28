from __future__ import print_function
from keras.models import load_model
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data

def get_data():
    try:
        path = get_file('babi-tasks-v1-2.tar.gz',
                        origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
    except:
        print('Error downloading dataset, please download it manually:\n'
              '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
              '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
        raise
    tar = tarfile.open(path)

    challenges = {
        # QA1 with 10,000 samples
        'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
        # QA2 with 10,000 samples
        'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
    }
    challenge_type = 'single_supporting_fact_10k'
    challenge = challenges[challenge_type]

    print('Extracting stories for the challenge:', challenge_type)
    test_stories = get_stories(tar.extractfile(challenge.format('test')))
    return test_stories


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))

def dataTransform(test_stories):
    vocab = set()
    for story, q, answer in test_stories:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    story_maxlen = max(map(len, (x for x, _, _ in test_stories)))
    query_maxlen = max(map(len, (x for _, x, _ in test_stories)))

    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Story max length:', story_maxlen, 'words')
    print('Query max length:', query_maxlen, 'words')
    print('Number of test stories:', len(test_stories))
    print('-')
    print('Here\'s what a "story" tuple looks like (input, query, answer):')
    print(test_stories[0])
    print('-')
    print('Vectorizing the word sequences...')

    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                                word_idx,
                                                                story_maxlen,
                                                                query_maxlen)
    return inputs_test, queries_test





if __name__ =='__main__':


    model = load_model('my_model.h5')

    testStories = get_data()
    inputs_train, queries_train = dataTransform(testStories)

    # evaluate loaded model on test data
    # Define X_test & Y_test data first
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # score = model.evaluate(X_test, Y_test, verbose=0)

    yFit = model.predict([inputs_train, queries_train], batch_size=32, verbose=0)
    print(yFit)

