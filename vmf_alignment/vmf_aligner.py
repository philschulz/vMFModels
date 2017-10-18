import argparse, sys, os
# warnings.filterwarnings("error")
# necessary for processing on cluster
import logging
import random as r
from collections import Counter
from typing import Tuple, List, Dict, Optional

from gensim.models import Word2Vec, KeyedVectors
from scipy.misc import logsumexp
from scipy.special import iv as bessel, psi

from vmf_utils import functions as f
from vmf_utils.gamma_distribution import GammaDist
from vmf_utils.vmf_distribution import VMF
from vmf_utils.optimisers import *


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
    source2vec = np.zeros(source_embeddings.vector_size)
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
        self.corpus = corpus.copy()
        if shuffle:
            r.shuffle(corpus)
        self.corpus_size = len(corpus)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stop = False
        self.index = 0
        self.batch_num = 0

    def next(self) -> List[Tuple[List[int], List[int]]]:
        """
        Return the next batch.

        :return: A batch or None if the corpus has been exhausted.
        """
        start = self.index
        end = self.index + self.batch_size
        self.index = end
        self.batch_num += 1

        if end >= self.corpus_size + self.batch_size:
            return None, self.batch_num
        elif end >= self.corpus_size:
            missing = end - self.corpus_size
            batch = self.corpus[start:]
            # fill up batch with initial sentences
            batch += self.corpus[:end]
            return batch, self.batch_num
        else:
            return self.corpus[start:end], self.batch_num

    def reset(self) -> None:
        """
        Reset this iterator.
        """
        self.index = 0
        self.batch_num = 0
        if self.shuffle:
            r.shuffle(self.corpus)


class VMFIBM1(object):
    """
    An alignment model based in IBM model 1 that uses vMF distributions to model source embeddings.

    :param dim: The dimensionality of the word embeddings.
    :param source_embeddings: Embedding matrix for the source vocabulary.
    :param source_tokens: Number of source tokens.
    :param target_vocab: Target vocabulary.
    :param optimiser: An optimiser to update the parameters.
    :param mc_decoder: Number of samples to take for MC decoding. The mean transformed concentration is used if this
    value is 0.
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
        return x / np.linalg.norm(x, axis=x.ndim - 1, keepdims=True)

    def __init__(self, dim: int, source_embeddings: np.array, source_tokens: int, target_vocab: Dict[int, str],
                 optimiser: Optimiser,
                 mc_decode: Optional[int] = 0,
                 concentration_fix: Optional[int] = None,
                 logger: Optional[logging.Logger] = None):

        self.dim = dim
        self.vmf = VMF(dim)
        self.source_embeddings = source_embeddings
        self.source_tokens = source_tokens
        self.target_vocab = target_vocab
        self.bessel_ratio = lambda x: bessel(dim / 2, x) / bessel(dim / 2 - 1, x)
        self.corpus_scale = 1 / self.source_tokens
        self.mc_decode = mc_decode

        # gamma prior
        self.gamma_prior = GammaDist(2, 100)

        # vMF mean parameters
        self.vMF_natural_params = np.random.normal(size=(len(self.target_vocab), dim))
        self.target_directions = self.normalise(self.vMF_natural_params)

        # vMF concentration parameters and noise
        self.target_norm_means = np.random.uniform(low=0, high=1, size=(len(self.target_vocab), 1))
        self.target_norm_std_embed = np.random.uniform(low=0,high=1,size=(len(self.target_vocab), 1))
        self.noise_samples = None
        self.target_concentration = np.ones(
            self.target_norm_std_embed.shape) * concentration_fix if concentration_fix else None
        self.fix_concentration = concentration_fix is not None

        # prior on mean directions
        self.mu_0 = self.normalise(
            np.random.normal(size=(1, self.source_embeddings.shape[1])))  # np.mean(self.source_embeddings, axis=0)
        self.kappa_0 = np.array([dim], dtype=np.float64)
        self.fix_concentration = concentration_fix is not None

        # initialise optimiser
        self.child_optimiser = optimiser(
            list([self.vMF_natural_params, self.target_norm_means, self.target_norm_std_embed]))
        self.root_optimiser = optimiser(list([self.mu_0]))

        # logger
        self.logger = logger

    def align(self, corpus: List[Tuple[List[int], List[int]]], out_file: str, format: str = "moses") -> None:
        """
        Aligns a corpus given the current paramter estimates.

        :param corpus: The corpus to be aligned.
        :param out_file: Path to the file to which the output gets written.
        """
        # Set noises to 1 to get mean of Gaussian variational approximation
        self.noise_samples = np.ones(shape=self.target_norm_std_embed.shape)

        with open(out_file, 'w') as out:
            for s_sent, t_sent in corpus:
                source_embeddings = self.source_embeddings[s_sent]
                target_directions = self.target_directions[t_sent]

                scores = np.zeros((len(t_sent), len(s_sent)))
                if self.mc_decode > 0:
                    for _ in range(self.mc_decode):
                        target_concentrations = self._sample_concentration(t_sent)
                        sample_scores, *_ = self.vmf.log_density(source_embeddings, target_directions,
                                                                 target_concentrations)
                        scores += sample_scores
                else:
                    target_concentrations, *_ = self._compute_concentration(t_sent)
                scores, *_ = self.vmf.log_density(source_embeddings, target_directions, target_concentrations)
                links = np.argmax(scores, axis=1) - 1

                output = list()
                for idx, link in enumerate(links):
                    if link > -1:
                        output.append(str(idx) + "-" + str(link))

                out.write(" ".join(output) + "\n")

    def _sample_concentration(self, target_sentence: List[int]) -> np.array:
        target_var_means = self.target_norm_means[target_sentence]
        target_var_stds = np.exp(self.target_norm_std_embed[target_sentence])
        epsilon = np.random.normal(size=target_var_stds.shape)
        z = target_var_means + target_var_stds * epsilon
        kappa, soft_plus_grad = f.soft_plus(z)
        return kappa

    def train(self, corpus: List[Tuple[List[int], List[int]]], iterations: int, batch_size: int) -> None:
        iterator = BatchIterator(corpus, batch_size)

        for i in range(iterations):
            self.logger.info("Starting epoch {}".format(i + 1))

            batch, batch_num = iterator.next()
            while batch is not None:

                mean_direction = np.mean(self.target_directions, axis=0)
                mean_mean = np.mean(self.target_norm_means)
                mean_std = np.mean(self.target_norm_std_embed)

                self.logger.debug("Mean direction for batch = {}".format(mean_direction))
                self.logger.debug('Variational mean for kappa = {}'.format(mean_mean))
                self.logger.debug('Variational std embedding for kappa = {}'.format(mean_std))

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
                    log_likelihood += log_marginal
                    target_ss[target_sent] += expected_ss
                    mean_grad[target_sent] += var_mean_grad
                    std_grad[target_sent] += var_std_grad

                # integrate density from top-level distribution
                log_likelihood *= self.source_tokens / data_points
                prior_density, *_ = self.vmf.log_density(np.sum(self.target_directions, axis=0, keepdims=True),
                                                         self.mu_0, self.kappa_0)
                prior_density = np.sum(prior_density)
                log_likelihood += prior_density
                self.logger.info("Log-likelihood at batch {}: {}".format(batch_num, log_likelihood))

                target_ss *= self.source_tokens / data_points
                mean_grad *= self.source_tokens / data_points
                std_grad *= self.source_tokens / data_points

                self.logger.info('Updating variational parameters')
                self.update_params(target_ss, mean_grad, std_grad)
                batch, batch_num = iterator.next()

            iterator.reset()

    def _compute_concentration(self, target_sentence: Optional[List[int]] = None) -> Tuple[np.array, np.array]:
        """
        Compute the concentration parameters for a target sentence or the entire corpus if not target sentence is
        provided.

        :param target_sentence: The target sentence.
        :return: The concentration parameters and softplus gradients.
        """
        if self.fix_concentration:
            kappa = self.target_concentration[target_sentence] if target_sentence else self.target_concentration
            soft_plus_grad = None
        else:
            target_var_means = self.target_norm_means[
                target_sentence] if target_sentence else self.target_norm_std_embed
            target_var_stds = np.exp(
                self.target_norm_std_embed[target_sentence] if target_sentence else self.target_norm_std_embed)
            epsilon = self.noise_samples[target_sentence] if target_sentence else self.noise_samples
            z = target_var_means + target_var_stds * epsilon
            kappa, soft_plus_grad = f.soft_plus(z)

        return kappa, target_var_stds, z, soft_plus_grad

    def compute_expectations(self, source_sent: List[int], target_sent: List[int]) -> None:
        """
        Compute the expectations for one sentence pair under the current variational parameters.

        :param source_sent: The numberised source sentence.
        :param target_sent: The numberised target sentence.
        """
        source_vecs = self.source_embeddings[source_sent]
        target_directions = self.target_directions[target_sent]
        epsilon = self.noise_samples[target_sent]
        kappa, target_var_stds, z, soft_plus_grad = self._compute_concentration(target_sent)
        data_points = len(source_sent)
        scale = data_points * self.corpus_scale

        scores, _, kappa_grad = self.vmf.log_density(source_vecs, target_directions, kappa)
        prior_scores, prior_grad = self.gamma_prior.log_density(kappa)
        # add log-Jacobian term
        scores += scale * (prior_scores + np.log(f.sigmoid(z)))

        # compute posterior
        totals = logsumexp(scores, axis=0).reshape(len(source_sent), 1)
        posteriors = np.exp(scores.T - totals).T

        # compute sum of marginal log-likelihoods
        log_marginal = np.sum(totals)

        # compute expected sufficient statistics
        observed_vecs = np.dot(posteriors, source_vecs)

        if self.fix_concentration:
            return log_marginal, observed_vecs, np.zeros((len(target_sent), 1)), np.zeros((len(target_sent), 1))

        # likelihood gradient
        kappa_grad *= soft_plus_grad
        # transformed prior gradient
        prior_grad = scale * (prior_grad * soft_plus_grad + f.sigmoid(-z))
        # add gradient of log-jacobian
        var_mean_grad = kappa_grad + prior_grad
        # multiply with gradients of std normal and embedding transformation and add entropy
        var_std_grad = var_mean_grad * epsilon * target_var_stds + scale

        return log_marginal, observed_vecs, var_mean_grad, var_std_grad

    def update_params(self, direction_ss: np.array, variational_mean_gradient: np.array,
                      variational_std_gradient: np.array) -> None:
        """
        Update the variational parameters.

        :param direction_ss: Sufficient statistics of the component vMFs.
        :param variational_mean_gradient: Gradient of the mean of the variational Gaussian.
        :param variational_std_gradient: Gradient of the embedding of the variational standard deviation.
        """
        self.update_target_params(direction_ss, variational_mean_gradient, variational_std_gradient)
        self.update_global_params()

    def update_target_params(self, direction_ss: np.array, variational_mean_grad: np.array,
                             variational_std_grad: np.array) -> None:
        """
        Update the variational parameters and auxiliary quantities of the vMFs distributions associated with
        the target types.
        """
        prior_ss = self.kappa_0 * self.mu_0
        target_concentrations, *_ = self._compute_concentration()
        natural_params = target_concentrations * direction_ss + prior_ss
        natural_gradient = natural_params - self.vMF_natural_params

        natural_param_upate, var_mean_update, var_std_embed_update = \
            self.child_optimiser(natural_gradient, variational_mean_grad, variational_std_grad)

        self.vMF_natural_params += natural_param_upate
        if not self.fix_concentration:
            self.target_norm_means += var_mean_update
            self.target_norm_std_embed += var_std_embed_update

        # compute expected mean directions
        target_means = self.normalise(self.vMF_natural_params)
        self.target_directions = target_means * self.bessel_ratio(target_concentrations)

    def update_global_params(self) -> None:
        """Update the global parameters through empirical Bayes (MLE on expected params)"""

        ss = np.sum(self.target_directions, axis=0)
        density, mu_grad, kappa_grad = self.vmf.log_density(ss, self.mu_0, self.kappa_0)
        mu_update = self.root_optimiser(mu_grad)[0]

        self.mu_0 += mu_update

    def write_param(self, path_to_file: str) -> None:
        """
        Write the parameters of this model to file.

        :param path_to_file: Path to the output file.
        """
        with open(path_to_file + ".means", "w") as means, open(path_to_file + ".concentration", "w") as conc:
            means.write(str(self.target_directions.shape[0]) + " " + str(self.dim) + "\n")
            # no embedding needed for NULL word
            kappas, _ = f.soft_plus(self.target_norm_means + np.exp(self.target_norm_std_embed))
            kappas = kappas.reshape((kappas.size,))

            for word, idx in self.target_vocab.items():
                # TODO think about what to do here -> use mean parameter or expected mean
                mean_direction = self.target_directions[idx]
                kappa = kappas[idx]
                means.write(word + " " + ' '.join(map(str, mean_direction)) + "\n")
                conc.write(word + ' ' + str(kappa) + "\n")


class VMFIBM1Mult(VMFIBM1):
    """
    A model that pairs the hierarchical vMF model with a Bayesian version of IBM1 that places a Dirichlet prior on the
    translation parameters.
    """

    def __init__(self, dim, source_embeddings, target_vocab, dirichlet_param):
        super().__init__(dim, source_embeddings, target_vocab)
        self.translation_categoricals = list()
        self.expected_translation_counts = list()
        self.dirichlet_param = dirichlet_param
        self.dirichlet_total = (source_embeddings.shape[0] - 1) * self.dirichlet_param

    def compute_categorical_scores(self, source_sent: List[int], target_sent: List[int]) -> np.array:
        cat_score = np.zeros((len(target_sent), len(source_sent)))
        for tdx, target in enumerate(target_sent):
            pmf = self.translation_categoricals[target]
            for sdx, source in enumerate(source_sent):
                cat_score[tdx, sdx] = pmf[source]
        return cat_score

    def compute_expectations(self, source_sent: List[int], target_sent: List[int]) -> None:
        """
        Compute the expectations for one sentence pair under the current variational parameters.

        :param source_sent: The numberised source sentence.
        :param target_sent: The numberised target sentence.
        """
        source_vecs = self.source_embeddings[source_sent]
        target_directions = self.target_directions[target_sent]
        epsilon = self.noise_samples[target_sent]
        kappa, target_var_stds, z, soft_plus_grad = self._compute_concentration(target_sent)
        data_points = len(source_sent)
        scale = data_points * self.corpus_scale

        scores, _, kappa_grad = self.vmf.log_density(source_vecs, target_directions, kappa)
        prior_scores, prior_grad = self.gamma_prior.log_density(kappa)
        # add log-Jacobian term
        scores += scale * (prior_scores + np.log(f.sigmoid(z)))
        scores += self.compute_categorical_scores(source_sent, target_sent)

        # compute posterior
        totals = logsumexp(scores, axis=0)[:,np.newaxis]
        posteriors = np.exp(scores.T - totals).T

        # compute sum of marginal log-likelihoods
        log_marginal = np.sum(totals)

        # compute expected sufficient statistics
        observed_vecs = np.dot(posteriors, source_vecs)

        if self.fix_concentration:
            return log_marginal, observed_vecs, np.zeros((len(target_sent), 1)), np.zeros((len(target_sent), 1))

        # likelihood gradient
        kappa_grad *= soft_plus_grad
        # transformed prior gradient
        prior_grad = scale * (prior_grad * soft_plus_grad + f.sigmoid(-z))
        # add gradient of log-jacobian
        var_mean_grad = kappa_grad + prior_grad
        # multiply with gradients of std normal and embedding transformation and add entropy
        var_std_grad = var_mean_grad * epsilon * target_var_stds + scale

        return log_marginal, posteriors, observed_vecs, var_mean_grad, var_std_grad

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
    command_line_parser.add_argument('--model', '-m', type=str, default="vmf", choices=["vmf", "cat-vmf"],
                                     help="Specify the model to be used during"
                                          "alignment. Choices: %(choices)s. Default: %(default)s.")
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
    command_line_parser.add_argument('--iter', '-i', type=int, default=3, help="Set the number of iterations used for "
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
    command_line_parser.add_argument("--learning-rate", "--r", type=float, default=0.0001,
                                     help="Set the (initial) learning rate used during optmisation. "
                                          "Default: %(default)s.")
    command_line_parser.add_argument("--clip-gradient", type=float, default=10,
                                     help="The likelihood term yields rather big gradient estimates which need to be "
                                          "clipped to avoid numerical problems. This option specifies the cut-off "
                                          "point. Default: %(default)s.")
    command_line_parser.add_argument("--optimiser", type=str, choices=["sgd", "adagrad", "adadelta"], default="sgd",
                                     help="Choose an optimiser from %(choices)s. Default: %(default)s.")
    command_line_parser.add_argument("--mc-decoding", type=int,
                                     help="Number of samples for MC decoding. Decoding will usually just take the "
                                          "average unconstrained concentration (z). MC decoding instead samples values "
                                          "of z.")

    args = command_line_parser.parse_args()
    model = args.model
    iterations = args.iter
    out_file = args.out_file
    batch_size = args.batch_size
    dir = args.dirichlet_param
    fix = args.fix_target_concentration
    learning_rate = args.learning_rate
    opt = args.optimiser
    clipping = args.clip_gradient

    embedding_output_prefix = "target"

    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s]: %(message)s')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create optimiser
    optimiser = get_optimiser(optimiser=opt, learning_rate=learning_rate, gradient_clipping=clipping)

    # load embeddings
    logger.info("Loading embeddings from {}".format(args.embeddings))
    embeddings = KeyedVectors.load_word2vec_format(args.embeddings, binary=args.binary)

    logger.info("Constructing corpus")
    corpus, source_map, target_map, source_mean_direction, source_tokens = \
        read_corpus(args.source, args.target, embeddings)
    dim = embeddings.vector_size

    aligner = VMFIBM1(dim, source_map, source_tokens, target_map, optimiser, True, fix, logger=logger) \
        if model == "vmf" else VMFIBM1Mult(dim, source_map, target_map, dir)
    aligner.train(corpus, iterations, batch_size)

    logger.info("Starting to align")
    aligner.align(corpus, out_file)
    logger.info("Alignment finished")

    logger.info(
        "Writing learned parameters to files with prefix {}".format(os.path.join(os.getcwd(), embedding_output_prefix)))
    aligner.write_param(embedding_output_prefix)


if __name__ == "__main__":
    main()
