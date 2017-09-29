import argparse, sys, os, warnings

# warnings.filterwarnings("error")
# necessary for processing on cluster
import datetime, logging
from collections import Counter
import random as r

import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from scipy.special import iv as bessel, psi
from scipy.misc import logsumexp
from typing import Tuple, List, Dict, Optional

from utils import functions as f
from vmf.vmf_distribution import VMF


def read_corpus(path_to_source: str, path_to_target: str, source_embeddings: Word2Vec) -> Tuple[
    List[Tuple[List[int]]], Dict[str, int], float, int]:
    """
    Read a parallel corpus in text format and output the corpus in numberised format. Also map the source words
    to embeddings obtained from a word2Vec model.

    :param path_to_source: Path to the source file of the corpus
    :param path_to_target: Path to the target file of the corpus
    :param source_embeddings: A gensim Word2Vec model
    :return: The numberised corpus, a map from target indeces to words, a map from source indeces to (normalised) embeddings
    and the (normalised) mean direction of source vectors
    """
    source_map = dict()
    # TODO adjust dim
    source2vec = np.zeros(50)
    source_mean_direction = 0
    target_map = {"NULL": 0}
    corpus = list()
    s_count = 1
    t_count = 1

    with open(path_to_source) as source, open(path_to_target) as target:
        source_tokens = 0

        for s_line in source:
            s_line = s_line.split()
            t_line = target.readline().split()
            s_sent = list()
            # prepend with NULL word
            t_sent = [0]
            source_tokens += len(s_line)

            for s_word in s_line:
                if s_word not in source_map:
                    source_map[s_word] = s_count
                    vector = VMFIBM1.normalise(source_embeddings[s_word])
                    source2vec = np.vstack((source2vec, vector))
                    source_mean_direction += vector
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

    return corpus, source2vec, target_map, (source_mean_direction / (source2vec.shape[0] - 1)), source_tokens


class BatchIterator(object):
    def __init__(self, corpus: List[Tuple[List[int], List[int]]], batch_size: int, shuffle: bool = True):
        self.corpus = corpus
        if shuffle:
            r.shuffle(corpus)
        self.corpus_size = len(corpus)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stop = False
        self.index = 0

    def next(self):
        """
        Return the next batch.

        :return: A batch or None if the corpus has been exhausted.
        """
        start = self.index
        end = self.index + self.batch_size
        self.index = end

        if end > self.corpus_size + self.batch_size:
            return None
        elif end > self.corpus_size:
            missing = end - self.corpus_size
            batch = self.corpus[start:]
            # fill up batch with initial sentences
            batch += self.corpus[:end]
            return batch
        else:
            return self.corpus[start:end]

    def reset(self):
        """
        Reset this iterator.
        """
        self.index = 0
        if self.shuffle:
            r.shuffle(self.corpus)


class VMFIBM1(object):
    """
    An alignment model based in IBM model 1 that uses vMF distributions to model source embeddings.

    :param dim: The dimensionality of the word embeddings.
    :param source_embeddings: Embedding matrix for the source vocabulary.
    :param source_tokens: Number of source tokens.
    :param target_vocab: Target vocabulary.
    :param random_initial_directions: Have all initial target vectors point in random directions.
    :param concentration_cap: Maximal concentration value.
    :param concentration_fix: Value for fixed concentration parameters.
    :param logger: A logger for events during processing.
    """
    slice_iterations = 5

    @staticmethod
    def normalise(x: np.array) -> np.array:
        """
        Normalise vectors to unit length.

        :param x: The vector.
        :return: The normalised vector.
        """
        return x / np.linalg.norm(x, axis=1).reshape(x.shape[0], 1) if len(x.shape) > 1 else x / np.linalg.norm(x)

    def __init__(self, dim: int, source_embeddings: np.array, source_tokens: int, target_vocab: Dict[int, str],
                 random_initial_directions: bool = False, concentration_cap: Optional[int] = None,
                 concentration_fix: Optional[int] = None, logger: Optional[logging.Logger] = None):

        self.dim = dim
        self.vmf = VMF(dim)
        self.source_embeddings = source_embeddings
        self.source_tokens = source_tokens
        self.target_vocab = target_vocab
        self.bessel_ratio = lambda x: bessel(dim/2, x) / bessel(dim/1, x)

        # vMF mean parameters
        self.vMF_natural_params = np.random.normal(size=(len(self.target_vocab), dim)) if random_initial_directions \
            else np.zeros((len(self.target_vocab), dim))
        self.target_directions = self.normalise(self.vMF_natural_params)

        # vMF concentration parameters and noise
        self.target_norm_means = np.ones((len(self.target_vocab),1)) # * dim
        self.target_norm_std_embed = np.ones((len(self.target_vocab),1))
        self.noise_samples = None

        self.mu_0 = self.normalise(np.ones(dim))
        self.kappa_0 = 1
        self.init = random_initial_directions
        self.concentration_cap = concentration_cap
        self.fix_concentration = concentration_fix is not None
        if concentration_fix:
            self.target_concentration = concentration_fix

    def _random_init_directions(self) -> np.array:
        """
        Randomly initialise the directions of the target embeddings.
        """
        return self.normalise(np.random.normal(loc=0, scale=1, size=(len(self.target_vocab), self.dim)))

    def align(self, corpus: List[Tuple[List[int], List[int]]], out_file: str, format: str = "moses") -> None:
        """
        Aligns a corpus given the current paramter estimates.

        :param corpus: The corpus to be aligned.
        :param out_file: Path to the file to which the output gets written.
        """

        with open(out_file, 'w') as out:
            for s_sent, t_sent in corpus:
                # TODO change target concentration
                scores = self.vmf.log_density(s_sent, t_sent, self.target_directions, self.target_concentration)
                links = np.argmax(scores, axis=1) - 1

                output = list()
                for idx, link in enumerate(links):
                    if link > -1:
                        output.append(str(idx) + "-" + str(link))

                out.write(" ".join(output) + "\n")

    def train(self, corpus: List[Tuple[List[int], List[int]]], iterations: int, batch_size: int) -> None:
        iterator = BatchIterator(corpus, batch_size)

        for i in range(iterations):
            print("Starting iteration {} at {}".format(i + 1, datetime.datetime.now()))

            batch = iterator.next()
            while batch is not None:

                self.noise_samples = np.random.normal(loc=0, scale=1, size=self.target_norm_std_embed.shape)
                log_likelihood = 0
                data_points = 0
                target_ss = np.zeros((len(self.target_vocab), self.dim))
                mean_grad = np.zeros((len(self.target_vocab), 1))
                std_grad = np.zeros((len(self.target_vocab), 1))

                for source_sent, target_sent in batch:
                    data_points += len(source_sent)
                    log_marginal, expected_ss, var_mean_grad, var_std_grad = self.compute_expectations(
                        source_sent, target_sent)
                    # print("grad = {}".format(var_std_grad))
                    log_likelihood += log_marginal
                    target_ss[target_sent] += expected_ss
                    mean_grad[target_sent] += var_mean_grad
                    std_grad[target_sent] += var_std_grad

                print("Log-likelihood: {}".format(log_likelihood))

                target_ss *= self.source_tokens / data_points
                mean_grad /= data_points
                std_grad /= data_points

                print('Starting to udpdate params at {}'.format(datetime.datetime.now()))
                self.update_params(target_ss, mean_grad, std_grad)
                batch = iterator.next()

            iterator.reset()

    def compute_expectations(self, source_sent: List[int], target_sent: List[int]) -> None:
        """
        Compute the expectations for one sentence pair under the current variational parameters.

        :param source_sent: The numberised source sentence.
        :param target_sent: The numberised target sentence.
        """
        source_vecs = self.source_embeddings[source_sent]
        target_directions = self.target_directions[target_sent]

        # print(target_directions)

        if self.fix_concentration:
            kappa = self.target_concentration
        else:
            target_var_means = self.target_norm_means[target_sent]
            target_var_stds = np.exp(self.target_norm_std_embed[target_sent])

            kl_mean_grad, kl_std_grad = f.diagonal_gaussian_kl_grad(target_var_means, target_var_stds)

            epsilon = self.noise_samples[target_sent]
            z = target_var_means + target_var_stds * epsilon
            kappa, _ = f.soft_plus(z)

        scores, kappa_grad = self.vmf.log_density(source_vecs, target_directions, kappa)

        # compute posterior
        totals = logsumexp(scores, axis=0).reshape(len(source_sent), 1)
        posteriors = np.exp(scores.T - totals).T

        # compute sum of marginal log-likelihoods
        log_marginal = np.sum(totals)

        # compute expected sufficient statistics
        observed_vecs = np.dot(posteriors, source_vecs)

        if self.fix_concentration:
            return log_marginal, observed_vecs, np.zeros((len(target_sent),1)), np.zeros((len(target_sent),1))

        # add gradient of log-jacobian
        kappa_grad += f.sigmoid(-kappa)
        var_mean_grad = kappa_grad - len(source_sent) * kl_mean_grad[:, np.newaxis]
        # multiply with gradients of std normal and embedding transformation
        var_std_grad = kappa_grad * epsilon * target_var_stds - kl_std_grad[:, np.newaxis]


        return log_marginal, observed_vecs, var_mean_grad, var_std_grad

    def update_params(self, direction_ss: np.array, variational_mean_gradient: np.array,
                      variational_std_gradient: np.array) -> None:
        """
        Update the variational parameters.

        :param direction_ss: Sufficient statistics of the component vMFs.
        :param variational_mean_gradient: Gradient of the mean of the variational Gaussian.
        :param variational_std_gradient: Gradient of the embedding of the variational standard deviation.
        """
        # TODO make learning rate adjustable
        self.update_target_params(direction_ss, variational_mean_gradient, variational_std_gradient, 0.001)
        self.update_global_params()

    def _update_target_directions(self, direction_ss: np.array, learning_rate: int) -> None:
        prior_ss = self.kappa_0 * self.mu_0

        if self.fix_concentration:
            target_concentrations = self.target_concentration
        else:
            target_concentrations = self.target_norm_means + np.exp(self.target_norm_std_embed) * self.noise_samples
            target_concentrations = target_concentrations.reshape((target_concentrations.size, 1))

        # scale expected ss to corpus size
        direction_ss *= self.source_tokens
        # update natural parameter estimates
        natural_params = target_concentrations * direction_ss + prior_ss
        self.vMF_natural_params += learning_rate * (natural_params - self.vMF_natural_params)

        # compute expected mean directions
        target_means = self.normalise(self.vMF_natural_params)
        self.target_directions = target_means * self.bessel_ratio(target_concentrations)

    def _update_target_concentrations(self, variational_mean_grad:np.array, variational_std_grad: np.array, learning_rate) -> None:
        self.target_norm_means += learning_rate * variational_mean_grad
        self.target_norm_std_embed += learning_rate * variational_std_grad

    def update_target_params(self, direction_ss: np.array, variational_mean_grad: np.array,
                             variational_std_grad: np.array, learning_rate: float) -> None:
        """
        Update the variational parameters and auxiliary quantities of the vMFs distributions associated with
        the target types.
        """
        self._update_target_directions(direction_ss, learning_rate)
        if not self.fix_concentration:
            self._update_target_concentrations(variational_mean_grad, variational_std_grad, learning_rate)

    def update_global_params(self) -> None:
        """Update the global parameters through empirical Bayes (MLE on expected params)"""

        sum_of_means = np.sum(self.target_directions, axis=0)
        self.mu_0 = self.normalise(sum_of_means)
        self.kappa_0, _ = self.update_concentration(len(self.target_directions),
                                                    sum_of_means.reshape((1, self.dim)))

    def update_concentration(self, num_observations: int, ss: np.array) -> Tuple[float, float]:
        """
        Update the concentration parameter of a vMF using empirical Bayes.

        :param num_observations: The (expected) number of times this distribution has been observed
        :param ss: The (expected) sufficient statistics for this distribution
        :return: The updated concentration parameter
        """
        # TODO this is just a smoothing hack to make sure r != 1
        r = np.linalg.norm(ss, axis=1) / (num_observations + 1)
        # make sure that kappa is never 0 for numerical stability

        # TODO Fix this! kappa is somethimes negative which should be impossible
        kappa = ((r * self.dim) / (1 - np.power(r, 2))) + 1e-10

        if self.concentration_cap:
            kappa[kappa > self.concentration_cap] = self.concentration_cap
        # TODO only needed for debugging -> remove
        elif np.any(kappa < 0):
            print(kappa[kappa < 0])
        return kappa, r

    def write_param(self, path_to_file: str) -> None:
        """
        Write the parameters of this model to file.

        :param path_to_file: Path to the output file.
        """
        with open(path_to_file + ".means", "w") as means, open(path_to_file + ".concentration", "w") as conc:
            means.write(str(len(self.target_means)) + " " + str(self.dim) + "\n")
            # no embedding needed for NULL word
            for word, idx in self.target_vocab.items():
                mean_direction = self.target_directions[idx]
                kappa = self.target_concentrations[idx]
                means.write(word + " " + ' '.join(map(str, mean_direction)) + "\n")
                conc.write(word + ' ' + str(kappa) + "\n")


class VMFIBM1Mult(VMFIBM1):
    def __init__(self, dim, source_embeddings, target_vocab, dirichlet_param):
        super().__init__(dim, source_embeddings, target_vocab)
        self.translation_categoricals = list()
        self.expected_translation_counts = list()
        self.dirichlet_param = dirichlet_param
        self.dirichlet_total = (source_embeddings.shape[0] - 1) * self.dirichlet_param

    def initialise_params(self):
        # TODO work this over to get a better starting point
        super().initialise_params()
        for _ in sorted(self.target_vocab.values()):
            # TODO initialise with source mean direction later on
            self.translation_categoricals.append(dict())
            self.expected_translation_counts.append(Counter())

    def log_density(self, source_sent: List[int], target_sent: List[int]) -> np.array:
        cat_score = np.zeros((len(source_sent), len(target_sent)))
        for tdx, target in enumerate(target_sent):
            pmf = self.translation_categoricals[target]
            for sdx, source in enumerate(source_sent):
                cat_score[sdx, tdx] = pmf[source]
        vmf_score = super().log_density(source_sent, target_sent)
        return cat_score + vmf_score

    def compute_expectations(self, source_sent: List[int], target_sent: List[int]) -> None:
        """
        Compute the expectations for one sentence pair under the current variational parameters

        :param source_sent: The numberised source sentence
        :param target_sent: The numberised target sentence
        """
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

    def update_target_params(self) -> None:
        """
        Update the variational parameters and auxiliary quantities of the vMFs distributions associated with
        the target types.

        :return: The sum of the newly estimated expected mean direction parameters
        """
        prior_ss = self.kappa_0 * self.mu_0

        posterior_ss = self.target_concentrations.reshape(
            (self.target_concentrations.size, 1)) * self.expected_target_means + prior_ss
        self.target_means = self.normalise(posterior_ss)

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
    command_line_parser.add_argument('--batch-size', type=int, required=True,
                                     help="The batch size for training.")
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
    command_line_parser.add_argument("--concentration-cap", type=float,
                                     help="Caps the possible concentration values at a maximum.")

    args = command_line_parser.parse_args()
    model = args.model
    iterations = args.iter
    out_file = args.out_file
    batch_size = args.batch_size
    dir = args.dirichlet_param
    fix = args.fix_target_concentration
    cap = args.concentration_cap

    print("Loading embeddings at {}".format(datetime.datetime.now()))
    embeddings = KeyedVectors.load_word2vec_format(args.embeddings, binary=args.binary)

    print("Constructing corpus at {}".format(datetime.datetime.now()))
    corpus, source_map, target_map, source_mean_direction, source_tokens = \
        read_corpus(args.source, args.target, embeddings)
    dim = embeddings.vector_size

    aligner = VMFIBM1(dim, source_map, source_tokens, target_map, True, cap, fix) \
        if model == "vmf" else VMFIBM1Mult(dim, source_map, target_map, dir)
    aligner.train(corpus, iterations, batch_size)

    print("Starting to align at {}".format(datetime.datetime.now()))
    aligner.align(corpus, out_file)
    aligner.write_param("target")


if __name__ == "__main__":
    main()
