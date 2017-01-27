import argparse, sys, os

# necessary for processing on cluster
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import datetime
from collections import Counter
from math import log, exp, pi

import numpy as np
from gensim.models import Word2Vec
from scipy.special import iv, psi

from utils.gamma_dist import GammaDist
from utils.logarithms import log_add_list, log_add
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
        self.sample = False

    def sample_concentration_params(self, sample=True):
        self.sample = sample

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
                        links.append(str(s_idx) + '-' + str(best_idx - 1))

                out.write(" ".join(links) + "\n")

    def train(self, corpus, iterations):
        for iter in range(iterations):
            print("Starting iteration {} at {}".format(iter + 1, datetime.datetime.now()))
            for source_sent, target_sent in corpus:
                self.compute_expectations(source_sent, target_sent)

            print('Starting to udpate params at {}'.format(datetime.datetime.now()))
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
            if self.sample:
                new_kappa, new_log_norm, new_bessel_ratio = self.sample_concentration(mu_e, kappa_e,
                                                                                      self.expected_target_counts[idx],
                                                                                      ss)
            else:
                new_kappa, new_log_norm, new_bessel_ratio = self.update_concentration(self.expected_target_counts[idx],
                                                                                      ss)

            new_mean = new_bessel_ratio * normal_ss
            sum_of_means += new_mean
            self.target_params[idx] = (new_mean, new_kappa)
            self.target_log_normaliser[idx] = new_log_norm

            self.expected_target_counts[idx] = 0
            self.expected_target_means[idx] = 0

            idx += 1

        return sum_of_means

    def update_global_params(self, sum_of_means):
        '''Update the global parameters through empirical Bayes (MLE on expected params)

        :param sum_of_means: The sum of the expected means of the draws from the global vMF
        '''

        # TODO check how to sample kappa_0
        self.mu_0 = self.normalise_vector(sum_of_means)
        r = np.linalg.norm(sum_of_means) / len(self.target_params)
        self.kappa_0 = (r * self.dim - r ** 3) / (1 - r ** 2)
        print(self.kappa_0)

    def update_concentration(self, num_observations, ss):
        '''Update the concentration parameter of a vMF using empirical Bayes.

        :param num_observations: The (expected) number of times this distribution has been observed
        :param ss: The (expected) sufficient statistics for this distribution
        :return: The triple (updated concentration, updated log-normaliser, updated bessel ratio)
        '''
        r = np.linalg.norm(ss) / num_observations
        kappa = (r * self.dim - r ** 3) / (1 - r ** 2)
        return kappa, self.log_normaliser(kappa), self.bessel_ratio(kappa)

    def sample_concentration(self, mu, kappa, num_observations, ss):
        '''Sample a concentration value and functions of it using slice sampling

        :param mu: The mean of the vMF
        :param kappa: The current concentration parameter
        :param num_observations: The (expected) number of observations
        :param ss: The (expected) sufficient statistics for this vMF
        :return: The triple (updated concentration, updated log-normaliser, updated bessel ratio)
        '''
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
        '''Compute the model density for a source embedding given a target word

        :param source: The source index
        :param target: The target index
        :return: The density of the embedding given the target word
        '''
        mu, kappa = self.target_params[target]
        embedding = self.source_embeddings[source]
        # TODO precompute log-normalisers through sampling
        return self.target_log_normaliser[target] + np.dot(embedding, mu) * kappa

    def log_normaliser(self, kappa):
        '''Compute the log-normaliser of the vMF

        :param kappa: The concentration of the vMF
        :return: The value of the log-normaliser
        '''
        try:
            return log(kappa) * (self.dim_half - 1) - log(self.bessel(kappa)) - self.log_two_pi
        except:
            print('Kappa = {}, log(kappa) = {}, bessel = {}, log-bessel = {}'.format(kappa, log(kappa),
                                                                                     self.bessel(kappa),
                                                                                     log(self.bessel(kappa))))

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


class VMFIBM1Mult(VMFIBM1):

    def __init__(self, dim, source_embeddings, target2words, dirichlet_param):
        super().__init__(dim, source_embeddings, target2words)
        self.translation_categoricals = list()
        self.expected_translation_counts = list()
        self.dirichlet_param = dirichlet_param
        self.dirichlet_total = 0

    def initialise_params(self):
        # TODO work this over to get a better starting point
        self.mu_0 = np.zeros(self.dim)
        self.kappa_0 = 1
        log_norm = self.log_normaliser(1)
        self.dirichlet_total = self.dirichlet_param*len(self.source_embeddings)
        for _ in sorted(self.target_vocab.values()):
            # TODO initialise with source mean later on
            self.target_params.append((np.zeros(self.dim), 1))
            self.target_log_normaliser.append(log_norm)
            self.translation_categoricals.append(dict())
            self.expected_translation_counts.append(dict())

    def log_density(self, source, target):
        cat_score = self.translation_categoricals[target][source] if source in self.translation_categoricals[
            target] else 0
        vmv_score = super().log_density(source, target)
        return cat_score + vmv_score

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
                cat_score = self.translation_categoricals[target][source] if source in self.translation_categoricals[
                    target] else 0
                log_scores.append(cat_score)

            total = log_add_list(log_scores)
            idx = 0
            for target in target_sent:
                posterior = log_scores[idx] - total
                # if math.isnan(posterior):
                #     print("Expectation")
                #     print("e_idx = {}, s_idx = {}, count = {}, total = {}".format(idx, source, log_scores[idx],
                #                                                                   self.expected_target_counts[idx]))
                try:
                    self.expected_target_means[target] += embedding * exp(posterior)
                except KeyError:
                    self.expected_target_means[target] = embedding * exp(posterior)
                self.expected_target_counts[target] = log_add(self.expected_target_counts[target],posterior)

                # categorical expectations
                try:
                    self.expected_translation_counts[target][source] = log_add(self.expected_translation_counts[target][source],posterior)
                except KeyError:
                    self.expected_translation_counts[target][source] = posterior

                idx += 1

    def update_params(self):
        self.update_target_params()

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
                                                                            exp(self.expected_target_counts[idx]), ss)
            new_mean = avg_bessel * normal_ss
            sum_of_means += new_mean
            self.target_params[idx] = (new_mean, new_kappa)
            self.target_log_normaliser[idx] = new_log_norm

            # update categorical params
            new_params = dict()
            for s_idx, count in self.expected_translation_counts[idx].items():
                new_params[s_idx] = psi(count + self.dirichlet_param) - psi(self.expected_target_counts[idx] + self.dirichlet_total)
                # if math.isnan(new_param) or math.isinf(new_param) or new_param > 0:
                #     print("e_idx = {}, s_idx = {}, count = {}, total = {}".format(idx, s_idx, count, self.expected_target_counts[idx]))
                # else:
                #     new_params[s_idx] = new_param
            self.translation_categoricals[idx] = new_params

            self.expected_translation_counts[idx] = dict()
            self.expected_target_counts[idx] = 0
            self.expected_target_means[idx] = 0

        return sum_of_means


def main():
    command_line_parser = argparse.ArgumentParser("This is word alignment tool that uses word vectors on the source"
                                                  "side.")
    command_line_parser.add_argument('--model', '-m', type=str, default="vmf",
                                     help="Specify the model to be used during"
                                          "alignment.")
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
    command_line_parser.add_argument('--binary', '-b', action="store_true",
                                     help="Indicates whether the embeddings file is in word2vec "
                                          "binary or text format.")
    command_line_parser.add_argument('--iter', '-i', type=int, default=10, help="Set the number of iterations used for "
                                                                                "training.")
    command_line_parser.add_argument('--out-file', '-o', type=str, default="alignments.txt", help="Path to the file to"
                                                                                                  "which the output alignments shall be printed")
    command_line_parser.add_argument('--sample-concentration', '-c', action='store_true', help="Compute the expected concentration parameters under a Gamma "
                                                                                               "prior. Warning: this will take a lot of time, especially if the "
                                                                                               "target vocabulary is large.")
    command_line_parser.add_argument("--dirichlet-param", '-d', type=float, default=0.001, help="Set the parameter of the Dirichlet prior for the "
                                                                                                "combined categorical and vMF model.")

    args = vars(command_line_parser.parse_args())
    model = args["model"]
    iter = args["iter"]
    out_file = args["out_file"]
    sample = args["sample_concentration"]
    dir = args["dirichlet_param"]

    print("Loading embeddings at {}".format(datetime.datetime.now()))
    embeddings = Word2Vec.load_word2vec_format(args["embeddings"], binary=args["binary"])

    print("Constructing corpus at {}".format(datetime.datetime.now()))
    corpus, source_map, target_map, source_mean = read_corpus(args["source"], args["target"], embeddings)
    dim = embeddings.vector_size

    aligner = VMFIBM1(dim, source_map, target_map) if model == "vmf" else VMFIBM1Mult(dim, source_map, target_map,
                                                                                      dir)

    print("Starting to align at {}".format(datetime.datetime.now()))
    aligner.align(corpus, out_file)
    aligner.write_param("target")


if __name__ == "__main__":
    main()
