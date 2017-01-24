import argparse, sys
sys.path.append("../utils")
import datetime
from collections import Counter
from math import log, exp, pi

import numpy as np
from gensim.models import Word2Vec
from scipy.special import iv

from utils.gamma_dist import GammaDist
from utils.logarithms import log_add_list
from utils.uni_slice_sampler import UnivariateSliceSampler


def read_corpus(path_to_source, path_to_target, source_embeddings):
    '''Read a parallel corpus in text format and output the corpus in numberised format. Also map the source words
    to embeddings obtained from a word2Vec model.

    :param path_to_source: Path to the source file of the corpus
    :param path_to_target: Path to the target file of the corpus
    :param source_embeddings: A gensim Word2Vec model
    :return: The numberised corpus, a map from target indeces to words, a map from source indeces to (normalised) embeddings
    and the (normalised) mean of source vectors
    '''
    source_map = dict()
    source2vec = dict()
    source_mean = 0
    target_map = {"NULL": 0}
    corpus = list()
    s_count = 1
    t_count = 1

    with open(path_to_source) as source, open(path_to_target) as target:
        for s_line in source:
            s_line = s_line.split()
            t_line = target.readline().split()
            s_sent = list()
            t_sent = [0]

            for s_word in s_line:
                if s_word not in source_map:
                    source_map[s_word] = s_count
                    vector = VMFIBM1.normalise_vector(source_embeddings[s_word])
                    source2vec[s_count] = vector
                    source_mean += vector
                    s_sent.append(s_count)
                    s_count += 1
                else:
                    s_sent.append(source_map[s_word])

            for t_word in t_line:
                if t_word not in target_map:
                    target_map[t_word] = t_count
                    t_sent.append(t_count)
                    t_count += 1
                else:
                    t_sent.append(target_map[t_word])

            corpus.append((s_sent, t_sent))

    print(corpus)

    return corpus, source2vec, target_map, (source_mean / len(source2vec))


class VMFIBM1(object):
    '''An alignment model based in IBM model 1 that uses vMF distributions to model source embeddings'''

    slice_iterations = 5

    @staticmethod
    def normalise_vector(x):
        '''Normalise a vector to unit length

        :param x: The vector
        :return: The normalised vector
        '''
        return x / np.linalg.norm(x)

    def __init__(self, dim, source2embeddings, target2words):

        self.dim = dim
        self.dim_half = dim / 2
        self.bessel = lambda x: iv(self.dim_half - 1, x)
        self.bessel_plus_1 = lambda x: iv(self.dim_half, x)
        self.log_two_pi = log(2 * pi) * self.dim_half
        self.source_embeddings = source2embeddings
        self.target_vocab = target2words
        self.target_params = list()
        self.mu_0 = np.zeros(self.dim)
        self.kappa_0 = 0
        self.target_log_normaliser = list()
        self.expected_target_means = dict()
        self.expected_target_counts = Counter()
        self.slice_sampler = UnivariateSliceSampler(GammaDist(1, 1))

    def initialise_params(self):
        # TODO work this over to get a better starting point
        self.mu_0 = np.zeros(self.dim)
        self.kappa_0 = 1
        log_norm = self.log_normaliser(1)
        for target in sorted(self.target_vocab.values()):
            # TODO initialise with source mean later on
            self.target_params.append((np.zeros(self.dim), 1))
            self.target_log_normaliser.append(log_norm)

    def align(self, corpus, out_file, format="moses"):
        '''Aligns a corpus given the current paramter estimates.

        :param corpus: The corpus to be aligned
        :param out_file: Path to the file to which the output gets written
        '''

        with open(out_file, 'w') as out:
            for s_sent, t_sent in corpus:
                # print('source = {}, target = {}'.format(s_sent, t_sent))

                links = list()
                for s_idx, s_word in enumerate(s_sent):
                    best_idx = -1
                    best_score = float('-inf')
                    for t_idx, t_word in enumerate(t_sent):
                        score = self.log_density(s_word, t_word)
                        # print('s = {},t = {}, score = {}'.format(s_idx, t_idx, score))
                        if score > best_score:
                            best_score = score
                            best_idx = t_idx

                    if best_idx > 0:
                        links.append(str(s_idx) + '-' + str(best_idx))

                out.write(" ".join(links) + "\n")

    def train(self, corpus, iterations):
        for iter in range(iterations):
            print("Starting iteration {} at {}".format(iter + 1, datetime.datetime.now()))
            for source_sent, target_sent in corpus:
                self.compute_expectations(source_sent, target_sent)

            self.update_params()

    def compute_expectations(self, source_sent, target_sent):
        '''Compute the expectations for one sentence pair under the current variational parameters

        :param source_sent: The numberised source sentence
        :param target_sent: The numberised target sentence
        '''
        for source in source_sent:
            embedding = self.source_embeddings[source]
            log_scores = list()
            for target in target_sent:
                log_scores.append(self.log_density(source, target))

            total = log_add_list(log_scores)
            idx = 0
            for target in target_sent:
                posterior = exp(log_scores[idx] - total)
                try:
                    self.expected_target_means[target] += embedding * posterior
                except KeyError:
                    self.expected_target_means[target] = embedding * posterior
                self.expected_target_counts[target] += posterior

    def update_params(self):
        '''Update the variational parameters and auxiliary quantities.'''

        print('Starting to udpate params at {}'.format(datetime.datetime.now()))
        sum_of_means = self.update_target_params()
        self.update_global_params(sum_of_means)

    def update_target_params(self):
        '''Update the variational parameters and auxiliary quantities of the vMFs distributions associated with
         the target types.

         :return: The sum of the newly estimated expected mean parameters
         '''

        sum_of_means = 0
        prior_ss = self.kappa_0 * self.mu_0

        for idx, params in enumerate(self.target_params):
            mu_e, kappa_e = params
            ss = self.expected_target_means[idx]
            posterior_ss = kappa_e * ss + prior_ss
            normal_ss = self.normalise_vector(posterior_ss)

            # print("mu_e = {}, kappa_e = {}, count = {}, ss = {}".format(mu_e,kappa_e,self.expected_target_counts[idx],ss))
            new_kappa, new_log_norm, avg_bessel = self.sample_concentration(mu_e, kappa_e,
                                                                            self.expected_target_counts[idx],
                                                                            ss)
            new_mean = avg_bessel * normal_ss
            sum_of_means += new_mean
            self.target_params[idx] = (new_mean, new_kappa)
            self.target_log_normaliser[idx] = new_log_norm

            self.expected_target_counts[idx] = 0
            self.expected_target_means[idx] = 0

        return sum_of_means

    def update_global_params(self, sum_of_means):
        '''Update the global parameters through empirical Bayes (MLE on expected params)

        :param sum_of_means: The sum of the expected means of the draws from the global vMF
        '''

        # TODO check how to sample kappa_0
        self.mu_0 = self.normalise_vector(sum_of_means)
        r = np.linalg.norm(sum_of_means) / len(self.target_params)
        self.kappa_0 = (r * self.dim - r ** 3) / (1 - r ** 2)

    def sample_concentration(self, mu, kappa, num_observations, ss):
        self.slice_sampler.set_likelihood(lambda x: self.vmf_likelihood(mu, x, num_observations, ss))
        samples, average = self.slice_sampler.sample(kappa, self.slice_iterations)

        log_normaliser_samples = [self.log_normaliser(sample) for sample in samples]
        bessel_ratio_samples = [self.bessel_ratio(sample) for sample in samples]

        log_normaliser_avg = sum(log_normaliser_samples) / len(log_normaliser_samples)
        bessel_ratio_avg = sum(bessel_ratio_samples) / len(bessel_ratio_samples)

        return average, log_normaliser_avg, bessel_ratio_avg

    def bessel_ratio(self, kappa):
        return self.bessel_plus_1(kappa) / self.bessel(kappa)

    def log_density(self, source, target):
        mu, kappa = self.target_params[target]
        embedding = self.source_embeddings[source]
        # TODO precompute log-normalisers through sampling
        return self.target_log_normaliser[target] + np.dot(embedding, mu) * kappa

    def log_normaliser(self, kappa):
        '''Compute the log-normaliser of the vMF

        :param kappa: The concentration of the vMF
        :return: The value of the log-normaliser
        '''
        return log(kappa) * (self.dim_half - 1) - log(self.bessel(kappa)) - self.log_two_pi

    def vmf_likelihood(self, mu, kappa, num_observations, ss):
        '''Compute the log-likelihood of a vMF.

        :param mu: The vMF mean
        :param kappa: The vMF concentration
        :param num_observations: The number of observations drawn from the vMF
        :param ss: The sufficient statistics extracted from the data
        :return: The likelihood of the current params
        '''

        log_normaliser = self.log_normaliser(kappa) * num_observations
        exponent = kappa * np.dot(mu, ss)
        return log_normaliser + exponent

    def write_param(self, path_to_file):
        with open(path_to_file + ".means", "w") as means, open(path_to_file + ".concentration", "w") as conc:
            means.write(str(len(self.target_params)) + " " + str(self.dim) + "\n")
            # no embedding needed for NULL word
            for word, idx in self.target_vocab.items():
                params = self.target_params[idx]
                means.write(word + " " + ' '.join(map(str, params[0])) + "\n")
                conc.write(word + ' ' + str(params[1]) + "\n")


def main():
    command_line_parser = argparse.ArgumentParser("This is word alignment tool that uses word vectors on the source"
                                                  "side.")
    command_line_parser.add_argument('--source', '-s', type=str, required=True,
                                     help="Path to the source side of the corpus. "
                                          "The source side should be given as text, "
                                          "with preprocessing applied as necessary.")
    command_line_parser.add_argument('--target', '-t', type=str, required=True,
                                     help="Path to the target side of the corpus. "
                                          "The target side should be given as text, "
                                          "with preprocessing applied as necessary.")
    command_line_parser.add_argument('--embeddings', '-e', type=str, required=True,
                                     help="Path to the file which stores the source "
                                          "embeddings.")
    command_line_parser.add_argument('--binary', '-b', type=bool, default=False,
                                     help="Indicates whether the embeddings file is in word2vec "
                                          "binary or text format.")
    command_line_parser.add_argument('--iter', '-i', type=int, default=10, help="Set the number of iterations used for "
                                                                                "training.")
    command_line_parser.add_argument('--out-file', '-o', type=str, default="alignments.txt", help="Path to the file to"
                                                                                                  "which the output alignments shall be printed")

    args = vars(command_line_parser.parse_args())

    print("Loading embeddings at {}".format(datetime.datetime.now()))
    embeddings = Word2Vec.load_word2vec_format(args["embeddings"], binary=args["binary"])

    print("Constructing corpus at {}".format(datetime.datetime.now()))
    corpus, source_map, target_map, source_mean = read_corpus(args["source"], args["target"], embeddings)
    dim = embeddings.vector_size
    iter = args["iter"]
    out_file = args["out_file"]

    aligner = VMFIBM1(dim, source_map, target_map)
    aligner.initialise_params()
    aligner.train(corpus, iter)

    aligner.align(corpus, out_file)
    aligner.write_param("source")


if __name__ == "__main__":
    main()
