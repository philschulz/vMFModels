import argparse, sys, os, warnings

# warnings.filterwarnings("error")
# necessary for processing on cluster
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import datetime
from collections import Counter

import numpy as np
from gensim.models import Word2Vec
from scipy.special import iv, psi
from scipy.misc import logsumexp

from utils.gamma_dist import GammaDist
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
    # TODO adjust dim
    source2vec = np.zeros(50)
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
                    source2vec = np.vstack((source2vec, vector))
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

            # TODO think about how to make these into arrays
            corpus.append((s_sent, t_sent))

    return corpus, source2vec, target_map, (source_mean / (source2vec.shape[0] - 1))


class VMFIBM1(object):
    '''An alignment model based in IBM model 1 that uses vMF distributions to model source embeddings'''

    slice_iterations = 5

    @staticmethod
    def normalise_vector(x):
        '''Normalise a vector to unit length

        :param x: The vector
        :return: The normalised vector
        '''
        return x / np.linalg.norm(x, axis=1).reshape(x.shape[0], 1) if len(x.shape) > 1 else x / np.linalg.norm(x)

    def __init__(self, dim, source2embeddings, target2words, concentration_cap = None, concentration_fix = None):

        self.dim = dim
        self.dim_half = dim / 2
        self.bessel = lambda x: iv(self.dim_half - 1, x)
        self.bessel_plus_1 = lambda x: iv(self.dim_half, x)
        self.log_two_pi = np.log(2 * np.pi) * self.dim_half
        self.source_embeddings = source2embeddings
        self.target_vocab = target2words
        # row-matrices
        self.target_variational_means = np.zeros((len(self.target_vocab), self.dim))
        self.expected_target_means = np.zeros((len(self.target_vocab), self.dim))
        self.expected_target_means = np.zeros((len(self.target_vocab), self.dim))
        # vectors
        self.target_concentrations = np.ones(len(self.target_vocab)) * concentration_fix if concentration_fix else np.ones(len(self.target_vocab))
        self.target_log_normaliser = self.log_normaliser(self.target_concentrations)
        self.expected_target_counts = np.zeros(len(self.target_vocab))

        self.mu_0 = self.normalise_vector(np.ones(self.dim))
        self.kappa_0 = 1
        self.slice_sampler = UnivariateSliceSampler(GammaDist(10, 1))
        self.sample = False
        self.init = False
        self.concentration_cap = concentration_cap
        self.fix_concentration = concentration_fix != None

    def sample_concentration_params(self, sample=True):
        self.sample = sample

    def initialise_params(self):
        # TODO not needed anymore -> better init can be achieved in first corpus iteration, though
        pass

    def align(self, corpus, out_file, format="moses"):
        '''Aligns a corpus given the current paramter estimates.

        :param corpus: The corpus to be aligned
        :param out_file: Path to the file to which the output gets written
        '''

        with open(out_file, 'w') as out:
            for s_sent, t_sent in corpus:
                scores = self.log_density(s_sent, t_sent)
                links = np.argmax(scores, axis=1) - 1

                output = list()
                for idx, link in enumerate(links):
                    if link > -1:
                        output.append(str(idx) + "-" + str(link))

                out.write(" ".join(output) + "\n")

    def train(self, corpus, iterations):
        for iter in range(iterations):
            print("Starting iteration {} at {}".format(iter + 1, datetime.datetime.now()))
            for source_sent, target_sent in corpus:
                self.compute_expectations(source_sent, target_sent)

            print('Starting to udpate params at {}'.format(datetime.datetime.now()))
            self.update_params()
            self.init = True

    def compute_expectations(self, source_sent, target_sent):
        '''Compute the expectations for one sentence pair under the current variational parameters

        :param source_sent: The numberised source sentence
        :param target_sent: The numberised target sentence
        '''
        source_vecs = self.source_embeddings[source_sent]
        scores = self.log_density(source_sent, target_sent) if self.init else np.zeros((len(source_sent), len(target_sent)))
        # sum rows
        totals = logsumexp(scores, axis=1).reshape(len(source_sent), 1)
        posteriors = np.exp(scores - totals)

        self.expected_target_counts[target_sent] += np.sum(posteriors, axis=0)
        self.expected_target_means[target_sent] += np.dot(posteriors.T, source_vecs)

    def update_params(self):
        '''Update the variational parameters and auxiliary quantities.'''
        self.update_target_params()
        self.update_global_params()

    def update_target_params(self):
        '''Update the variational parameters and auxiliary quantities of the vMFs distributions associated with
         the target types.
         '''
        prior_ss = self.kappa_0 * self.mu_0

        posterior_ss = self.target_concentrations.reshape(
            (self.target_concentrations.size, 1)) * self.expected_target_means + prior_ss
        self.target_means = self.normalise_vector(posterior_ss)

        if self.sample:
            kappa, log_normaliser, bessel_ratio = self.sample_concentration(self.target_variational_means,
                                                                            self.target_concentrations,
                                                                            self.expected_target_counts,
                                                                            self.expected_target_means)
            self.target_concentrations = kappa
            self.target_log_normaliser = log_normaliser
            self.target_variational_means = self.target_means * bessel_ratio.reshape((self.target_concentrations.size, 1))
        else:
            new_concentration, bessel_ratio = self.update_concentration(self.expected_target_counts,
                                                                  self.expected_target_means)
            if not self.fix_concentration:
                self.target_concentrations = new_concentration
                self.target_log_normaliser = self.log_normaliser(self.target_concentrations)
            self.target_variational_means = self.target_means * bessel_ratio.reshape(self.target_concentrations.size, 1)
        # reset expectations
        self.expected_target_counts = np.zeros(len(self.target_vocab))
        self.expected_target_means = np.zeros((len(self.target_vocab), self.dim))

    def update_global_params(self):
        '''Update the global parameters through empirical Bayes (MLE on expected params)'''

        # sum rows
        sum_of_means = np.sum(self.target_variational_means, axis=0)
        self.mu_0 = self.normalise_vector(sum_of_means)
        self.kappa_0, _ = self.update_concentration(len(self.target_variational_means), sum_of_means.reshape((1, self.dim)))

    def update_concentration(self, num_observations, ss):
        '''Update the concentration parameter of a vMF using empirical Bayes.

        :param num_observations: The (expected) number of times this distribution has been observed
        :param ss: The (expected) sufficient statistics for this distribution
        :return: The updated concentration parameter
        '''
        # TODO this is just a smoothing hack to make sure r != 1
        r = np.linalg.norm(ss, axis=1) / (num_observations + 1)
        # make sure that kappa is never 0 for numerical stability

        # TODO Fix this! kappa is somethimes negative which should be impossible
        kappa = ((r * self.dim) / (1 - np.power(r, 2))) + 1e-10
        # if np.any(kappa <= 0):
        #     print(self.expected_target_means.shape)
        #     sorted_kappa = np.sort(kappa)
        #     suspicious_observations = num_observations[kappa < 0]
        #     suspicious_norms = np.linalg.norm(ss, axis=1)[kappa < 0]
        #     suspicious_means = ss[kappa < 0]
        #     print(len(self.target_vocab))
        #     print(kappa.shape)
        #     print("Fuck")
        #     print(np.linalg.norm(ss[kappa < 0]))
        #     print(r[kappa < 0])
        #     print(num_observations[kappa < 0])
        #     meh = r[kappa < 0]
        #     idx = np.where(kappa < 0)
        #     print((meh*self.dim - np.power(meh,3))/(1-np.power(meh, 2)))
        #     print(idx[0][0])
        #     for key, value in self.target_vocab.items():
        #         if value == idx[0][0]:
        #             print(key)
        #             print(self.target_concentrations[value])
        #             break
        if self.concentration_cap:
            kappa[kappa > self.concentration_cap] = self.concentration_cap
        # TODO only needed for debugging -> remove
        elif np.any(kappa < 0):
            print(kappa[kappa < 0])
        return kappa, r

    def sample_concentration(self, mu, kappa, num_observations, ss):
        '''Sample a concentration value and functions of it using slice sampling

        :param mu: The mean of the vMF
        :param kappa: The current concentration parameter
        :param num_observations: The (expected) number of observations
        :param ss: The (expected) sufficient statistics for this vMF
        :return: The triple (updated concentration, updated log-normaliser, updated bessel ratio)
        '''
        self.slice_sampler.set_likelihood(lambda x: self.vmf_likelihood(mu, x, num_observations, ss))
        samples, average = self.slice_sampler.sample(kappa[0], self.slice_iterations)

        log_normaliser_samples = [self.log_normaliser(sample) for sample in samples]
        bessel_ratio_samples = [self.bessel_ratio(sample) for sample in samples]

        log_normaliser_avg = sum(log_normaliser_samples) / len(log_normaliser_samples)
        bessel_ratio_avg = sum(bessel_ratio_samples) / len(bessel_ratio_samples)

        return np.repeat(average, len(self.target_vocab)), np.repeat(log_normaliser_avg,
                                                                     len(self.target_vocab)), np.repeat(
            bessel_ratio_avg, len(self.target_vocab))

    def bessel_ratio(self, kappa):
        result = self.bessel_plus_1(kappa) / self.bessel(kappa)
        result[np.isnan(result)] = 1
        return result

    def log_density(self, source_sent, target_sent):
        '''Compute the model density for a source embedding given a target word

        :param source: The source index
        :param target: The target index
        :return: The density of the embedding given the target word
        '''
        source_vecs = self.source_embeddings[source_sent]
        target_means = self.target_variational_means[target_sent]
        target_concentrations = self.target_concentrations[target_sent]
        log_normalisers = self.target_log_normaliser[target_sent]

        return log_normalisers + np.dot(source_vecs, target_means.T) * target_concentrations

    def log_normaliser(self, kappa):
        '''Compute the log-normaliser of the vMF

        :param kappa: The concentration of the vMF
        :return: The value of the log-normaliser
        '''
        try:
            result = np.log(kappa) * (self.dim_half - 1) - np.log(self.bessel(kappa)) - self.log_two_pi
            if any(np.isinf(result)):
                print('switching to asymptotic regime for the infinity values !')
                idx = np.isinf(result)
                result[idx] = self.dim_half * np.log(2 * np.pi) + kappa[idx] - 0.5 * np.log(2 * np.pi * kappa[idx])

            return result
        except:
            print(kappa)
            print('Kappa = {}, log(kappa) = {}, bessel = {}, log-bessel = {}'.format(kappa, np.log(kappa),
                                                                                     self.bessel(kappa),
                                                                                     np.log(self.bessel(kappa))))
            sys.exit(0)

    def vmf_likelihood(self, mu, kappa, num_observations, ss):
        '''Compute the log-likelihood of a vMF.

        :param mu: The vMF mean
        :param kappa: The vMF concentration
        :param num_observations: The number of observations drawn from the vMF
        :param ss: The sufficient statistics extracted from the data
        :return: The likelihood of the current params
        '''

        log_normaliser = self.log_normaliser(kappa) * num_observations
        exponent = kappa * np.dot(mu, ss.T)
        return log_normaliser + exponent

    def write_param(self, path_to_file):
        with open(path_to_file + ".means", "w") as means, open(path_to_file + ".concentration", "w") as conc:
            means.write(str(len(self.target_means)) + " " + str(self.dim) + "\n")
            # no embedding needed for NULL word
            for word, idx in self.target_vocab.items():
                mean = self.target_variational_means[idx]
                kappa = self.target_concentrations[idx]
                means.write(word + " " + ' '.join(map(str, mean)) + "\n")
                conc.write(word + ' ' + str(kappa) + "\n")


class VMFIBM1Mult(VMFIBM1):
    def __init__(self, dim, source_embeddings, target2words, dirichlet_param):
        super().__init__(dim, source_embeddings, target2words)
        self.translation_categoricals = list()
        self.expected_translation_counts = list()
        self.dirichlet_param = dirichlet_param
        self.dirichlet_total = (source_embeddings.shape[0] - 1) * self.dirichlet_param

    def initialise_params(self):
        # TODO work this over to get a better starting point
        super().initialise_params()
        for _ in sorted(self.target_vocab.values()):
            # TODO initialise with source mean later on
            self.translation_categoricals.append(dict())
            self.expected_translation_counts.append(Counter())

    def log_density(self, source_sent, target_sent):
        cat_score = np.zeros((len(source_sent), len(target_sent)))
        for tdx, target in enumerate(target_sent):
            pmf = self.translation_categoricals[target]
            for sdx, source in enumerate(source_sent):
                cat_score[sdx, tdx] = pmf[source]
        vmf_score = super().log_density(source_sent, target_sent)
        return cat_score + vmf_score

    def compute_expectations(self, source_sent, target_sent):
        '''Compute the expectations for one sentence pair under the current variational parameters

        :param source_sent: The numberised source sentence
        :param target_sent: The numberised target sentence
        '''
        source_vecs = self.source_embeddings[source_sent]
        scores = self.log_density(source_sent, target_sent) if self.init else np.zeros(
            (len(source_sent), len(target_sent)))
        # sum rows
        totals = logsumexp(scores, axis=1).reshape(len(source_sent), 1)
        posteriors = np.exp(scores - totals)

        self.expected_target_counts[target_sent] += np.sum(posteriors, axis=0)
        self.expected_target_means[target_sent] += np.dot(posteriors.T, source_vecs)

        for t_idx in range(len(target_sent)):
            t_counts = self.expected_translation_counts[target_sent[t_idx]]
            for s_idx in range(len(source_sent)):
                t_counts[source_sent[s_idx]] += posteriors[s_idx, t_idx]

    def update_target_params(self):
        '''Update the variational parameters and auxiliary quantities of the vMFs distributions associated with
         the target types.

         :return: The sum of the newly estimated expected mean parameters
         '''
        prior_ss = self.kappa_0 * self.mu_0

        posterior_ss = self.target_concentrations.reshape(
            (self.target_concentrations.size, 1)) * self.expected_target_means + prior_ss
        self.target_means = self.normalise_vector(posterior_ss)

        self.target_concentrations, bessel_ratio = self.update_concentration(self.expected_target_counts,
                                                                             self.expected_target_means)
        self.target_variational_means = self.target_means * bessel_ratio.reshape(self.target_concentrations.size, 1)
        self.target_log_normaliser = self.log_normaliser(self.target_concentrations)
        for target, count in enumerate(self.expected_target_counts):
            total = psi(count + self.dirichlet_total)
            new_dict = dict()
            for source, t_count in self.expected_translation_counts[target].items():
                new_dict[source] = psi(t_count + self.dirichlet_param) - total
            self.translation_categoricals[target] = new_dict
            self.expected_translation_counts[target] = Counter()

        # reset expectations
        self.expected_target_counts = np.zeros(len(self.target_vocab))
        self.expected_target_means = np.zeros((len(self.target_vocab), self.dim))


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
    command_line_parser.add_argument('--out-file', '-o', type=str, default="alignments", help="Path to the file to"
                                                                                              "which the output alignments shall be printed")
    command_line_parser.add_argument('--sample-concentration', '-c', action='store_true',
                                     help="Compute the expected concentration parameters under a Gamma "
                                          "prior. Warning: this will take a lot of time, especially if the "
                                          "target vocabulary is large.")
    command_line_parser.add_argument("--dirichlet-param", '-d', type=float, default=0.001,
                                     help="Set the parameter of the Dirichlet prior for the "
                                          "combined categorical and vMF model.")
    command_line_parser.add_argument("--fix-target-concentration", type=float,
                                     help="Set fixed concentration parameter for all target words.")
    command_line_parser.add_argument("--concentration-cap", type=float, help="Caps the possible concentration values at a maximum.")

    args = vars(command_line_parser.parse_args())
    model = args["model"]
    iter = args["iter"]
    out_file = args["out_file"]
    sample = args["sample_concentration"]
    dir = args["dirichlet_param"]
    fix = args["fix_target_concentration"]
    cap = args["concentration_cap"]

    print("Loading embeddings at {}".format(datetime.datetime.now()))
    embeddings = Word2Vec.load_word2vec_format(args["embeddings"], binary=args["binary"])

    print("Constructing corpus at {}".format(datetime.datetime.now()))
    corpus, source_map, target_map, source_mean = read_corpus(args["source"], args["target"], embeddings)
    dim = embeddings.vector_size

    aligner = VMFIBM1(dim, source_map, target_map, cap, fix) if model == "vmf" else VMFIBM1Mult(dim, source_map, target_map,
                                                                                      dir)
    aligner.sample_concentration_params(sample)
    aligner.initialise_params()
    aligner.train(corpus, iter)

    print("Starting to align at {}".format(datetime.datetime.now()))
    aligner.align(corpus, out_file)
    aligner.write_param("target")


if __name__ == "__main__":
    main()
