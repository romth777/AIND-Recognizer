import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        # Check if the warning
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # initial parameters
        num_feature = self.X.shape[1]
        best_score = float('inf')
        best_model = None

        # search for min to max
        for num_components in range(self.min_n_components,self.max_n_components):
            try:
                # set up HMM
                model = GaussianHMM(n_components=num_components, n_iter=1000).fit(self.X, self.lengths)

                # calculate BIC score
                logL = model.score(self.X, self.lengths)
                p = num_components * (num_components - 1) + 2 * num_components * num_feature
                score = -2 * logL + p * np.log(len(self.X))
            except:
                score = float("inf")

            # update best
            if score < best_score:
                best_score = score
                best_model = model

        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        # Check if the warning
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # initial parameters
        best_hmm_model = None
        best_score = float("-inf")

        # search for min to max
        for nb_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                # set up HMM
                hmm_model = GaussianHMM(n_components=nb_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                # calculate log loss
                logL = hmm_model.score(self.X, self.lengths)
            except:
                best_score = float("-inf")
                best_hmm_model = None

            # calculate log loss of hole words except from this word
            logL_other = []
            for word in self.words:
                if word != self.this_word:
                    X, lengths = self.hwords[word]
                    try:
                        logL_other.append(hmm_model.score(X, lengths))
                    except:
                        logL_other.append(float("-inf"))

            # calculate DIC with this score and mean of the others
            score = logL - np.mean(logL_other)

            # update best score
            if score > best_score:
                best_score = score
                best_hmm_model = hmm_model

        return best_hmm_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        # Check if the warning
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # initial parameters
        n_splits = min(3, len(self.lengths))
        max_mean_logL = -1000000
        best_hidden_state = 0

        # search for min to max
        for nb_hidden_state in range(self.min_n_components, self.max_n_components+1):
            sum_log = 0
            n_splits_done = 0
            mean_logL = -1000000

            # if not be splitted
            if n_splits == 1 :
                try :
                    # set up model
                    hmm_model = GaussianHMM(n_components=nb_hidden_state, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                    # calculate the score
                    mean_logL = hmm_model.score(self.X, self.lengths)
                except:
                    continue

            # if be splitted
            else :
                # set up K-Fold
                split_method = KFold(n_splits=n_splits)
                # for each splitted data with train and test
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    # recombine the sequence
                    X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                    try:
                        # generate the model
                        model, logL = self.selected_model(nb_hidden_state, X_train, lengths_train)
                        sum_log += logL
                        n_splits_done += 1
                    except:
                        continue

                # if split done
                if n_splits_done != 0 :
                    mean_logL = sum_log / float(n_splits_done)
                else:
                    mean_logL = -1000000

            # update best
            if(mean_logL> max_mean_logL):
                max_mean_logL = mean_logL
                best_hidden_state = nb_hidden_state

        # if there are no best model then choice n_constant
        if best_hidden_state == 0:
            return self.base_model(self.n_constant)
        # otherwise choice the best one
        else:
            return self.base_model(best_hidden_state)
