def create_bag_of_words(corpus, ngram_range = (0, 1), stop_words = None, stem = False,
                        min_df = 0.05, max_df = 0.95, use_idf = False):
    """
    Turn a corpus of text into a bag-of-words.

    Parameters
    -----------
    corpus: ls
        test of documents in corpus
    ngram_range: tuple
        range of N-gram. Default (0,1)
    stop_words: ls
        list of commonly occuring words that have little semantic
        value
    stem: bool
        use a stemmer to stem words
    min_df: float
       exclude words that have a frequency less than the threshold
    max_df: float
        exclude words that have a frequency greater than the threshold
    use_idf: bool
        Re-weigh words according to the Term Frequency-Inverse Document Frequency
        (emphasize words unique to a document, suppress words common throughout the corpus)

    Returns
    -------
    bag_of_words: scipy sparse matrix
        scipy sparse matrix of text
    features:
        list of words
    """

    import pandas as pd
    import numpy as np
    import itertools
    import warnings
    import matplotlib.pyplot as plt

    from six.moves import zip, range
    from sklearn.model_selection import train_test_split
    import sklearn.feature_extraction.text as f_e
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.decomposition import LatentDirichletAllocation, PCA
    from sklearn.linear_model import LinearRegression, LogisticRegression, LassoLars
    from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, roc_curve, r2_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn import preprocessing
    from sklearn import model_selection as sk_ms

    from collections import Counter, OrderedDict

    import nltk
    from nltk import SnowballStemmer
    #parameters for vectorizer
    ANALYZER = "word" #unit of features are single words rather then phrases of words
    STRIP_ACCENTS = 'unicode'

    if stem:
        stemmer = nltk.SnowballStemmer("english")
        tokenize = lambda x: [stemmer.stem(i) for i in x.split()]
    else:
        tokenize = None
    vectorizer = CountVectorizer(analyzer=ANALYZER,
                                 tokenizer=tokenize,
                                 ngram_range=ngram_range,
                                 stop_words = stop_words,
                                 strip_accents=STRIP_ACCENTS,
                                 min_df = min_df,
                                 max_df = max_df)

    bag_of_words = vectorizer.fit_transform( corpus ) #transform our corpus is a bag of words
    features = vectorizer.get_feature_names()

    if use_idf:
        NORM = None #turn on normalization flag
        SMOOTH_IDF = True #prvents division by zero errors
        SUBLINEAR_IDF = True #replace TF with 1 + log(TF)
        transformer = TfidfTransformer(norm = NORM,smooth_idf = SMOOTH_IDF,sublinear_tf = True)
        #get the bag-of-words from the vectorizer and
        #then use TFIDF to limit the tokens found throughout the text
        tfidf = transformer.fit_transform(bag_of_words)

        return tfidf, features
    else:
        return bag_of_words, features


def get_word_counts(bag_of_words, feature_names):

    import pandas as pd
    import numpy as np
    import itertools
    import warnings
    import matplotlib.pyplot as plt

    from six.moves import zip, range
    from sklearn.model_selection import train_test_split
    import sklearn.feature_extraction.text as f_e
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.decomposition import LatentDirichletAllocation, PCA
    from sklearn.linear_model import LinearRegression, LogisticRegression, LassoLars
    from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, roc_curve, r2_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn import preprocessing
    from sklearn import model_selection as sk_ms

    from collections import Counter, OrderedDict

    import nltk
    from nltk import SnowballStemmer

    """
    Get the ordered word counts from a bag_of_words

    Parameters
    ----------
    bag_of_words: obj
        scipy sparse matrix from CounterVectorizer
    feature_names: ls
        list of words

    Returns
    -------
    word_counts: dict
        Dictionary of word counts
    """

    # convert bag of words to array
    np_bag_of_words = bag_of_words.toarray()

    # calculate word count.
    word_count = np.sum(np_bag_of_words,axis=0)

    # convert to flattened array.
    np_word_count = np.asarray(word_count).ravel()

    # create dict of words mapped to count of occurrences of each word.
    dict_word_counts = dict( zip(feature_names, np_word_count) )

    # Create ordered dictionary
    orddict_word_counts = OrderedDict( sorted(dict_word_counts.items(), key=lambda x: x[1], reverse=True), )

    return orddict_word_counts


def create_topics(tfidf, features, n_topics=3, n_top_words=5, seed=1):

    import pandas as pd
    import numpy as np
    import itertools
    import warnings
    import matplotlib.pyplot as plt

    from six.moves import zip, range
    from sklearn.model_selection import train_test_split
    import sklearn.feature_extraction.text as f_e
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.decomposition import LatentDirichletAllocation, PCA
    from sklearn.linear_model import LinearRegression, LogisticRegression, LassoLars
    from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, roc_curve, r2_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn import preprocessing
    from sklearn import model_selection as sk_ms

    from collections import Counter, OrderedDict

    import nltk
    from nltk import SnowballStemmer
    """
    Given a matrix of features of text data generate topics

    Parameters
    -----------
    tfidf: scipy sparse matrix
        sparse matrix of text features
    n_topics: int
        number of topics (default 10)
    n_top_words: int
        number of top words to display in each topic (default 10)

    Returns
    -------
    ls_keywords: ls
        list of keywords for each topics
    doctopic: array
        numpy array with percentages of topic that fit each category
    n_topics: int
        number of assumed topics
    n_top_words: int
        Number of top words in a given topic.
    """

    i=0
    lda = LatentDirichletAllocation(n_components= n_topics,
                                    learning_method='online', random_state=seed) #create an object that will create 5 topics
    i+=1
    doctopic = lda.fit_transform( tfidf )
    i+=1

    ls_keywords = []
    for i,topic in enumerate(lda.components_):
        word_idx = np.argsort(topic)[::-1][:n_top_words]
        keywords = ', '.join( features[i] for i in word_idx)
        ls_keywords.append(keywords)
        print(i, keywords)
        i+=1

    return ls_keywords, doctopic
