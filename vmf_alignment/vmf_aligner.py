import argparse
# warnings.filterwarnings("error")
# necessary for processing on cluster
import logging
import os
import random as r
import sys
from collections import Counter
from typing import Tuple, Dict

import mxnet as mx
from gensim.models import Word2Vec, KeyedVectors
from scipy.misc import logsumexp
from scipy.special import psi

from vmf_utils import functions as f
from vmf_utils.gamma_distribution import GammaDist
from vmf_utils.optimisers import *
from vmf_utils.vmf_distribution import *


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


def norm(data, axis, order=2, keepdims=True):
    exp = mx.sym.pow(base=data, exp=order)
    sum = mx.sym.sum(data=exp, axis=axis, keepdims=keepdims)
    norm = mx.sym.pow(base=sum, exp=1/order)

    return norm


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
    corpus, source_map, target_map, source_mean_direction, source_vocab_size = \
        read_corpus(args.source, args.target, embeddings)
    dim = embeddings.vector_size
    target_vocab_size = len(target_map)

    source_corpus = mx.nd.array(corpus[0])
    target_corpus = mx.nd.array(corpus[1])

    data_iter = mx.io.NDArrayIter(data={"target": target_corpus}, label={"source": source_corpus},
                                  batch_size=batch_size,
                                  shuffle=True, label_name="source", data_name="target")

    source = mx.sym.Variable("source")
    target = mx.sym.Variable("target")
    mu_0 = mx.sym.tile(data=mx.sym.Variable("mu0_weight", shape=(1,dim)), reps=(batch_size, 1))
    kappa_0 = mx.sym.tile(data=mx.sym.Variable("kappa0_weight", shape=(1, 1)), reps=(batch_size, 1))

    target_embed_weight = mx.sym.Variable("target_embed_weight")
    kappa_std_embed_weight = mx.sym.Variable("kappa_std_embed_weight")
    kappa_mean_weight = mx.sym.Variable("kappa_mean_weight")

    source_embed = mx.sym.Embedding(data=source, input_dim=source_vocab_size, output_dim=dim)
    target_embed = mx.sym.Embedding(data=target, input_dim=target_vocab_size, output_dim=dim,
                                    weight=target_embed_weight)
    kappa_std_embed = mx.sym.Embedding(data=target, input_dim=target_vocab_size, output_dim=1,
                                       weight=kappa_std_embed_weight)
    kappa_mean = mx.sym.Embedding(data=target_embed, input_dim=target_vocab_size, output_dim=1,
                                  weight=kappa_mean_weight)

    kappa_std = mx.sym.exp(data=kappa_std_embed)
    z = kappa_mean + kappa_std * mx.sym.random_normal(loc=0, scale=1, shape=(0, 1))
    kappa = mx.sym.Activation(data=z, act_type="softplus")
    kappa_expanded = mx.sym.expand_dims(data=kappa, axis=1)

    target_embed_norm = norm(data=target_embed, axis=2, keepdims=True)
    target_direction = target_embed / target_embed_norm
    target_direction = mx.sym.broadcast_mul(lhs=mx.sym.Custom(dim=dim/2, data=target_embed_norm, op_type="bessel") / mx.sym.Custom(dim=dim/2-1, data=target_embed_norm), rhs=target_direction)

    energy = mx.sym.batch_dot(lhs=source_embed, rhs=target_direction, transpose_b=True)
    normaliser = vmf_normaliser(dim=dim/2-1, kappa=kappa)
    likelihood = mx.sym.broadcast_add(lhs=normaliser, rhs=mx.sym.broadcast_mul(lhs=energy, rhs=kappa_expanded))
    max = mx.sym.max(likelihood, axis=2)
    total_logLikelihood = max + mx.sym.log(mx.sym.sum(data=mx.sym.exp(likelihood - max), axis=2, keepdims=False))
    total_logLikelihood = mx.sym.reshape(data=total_logLikelihood, shape=(-1,))


    prior_ss = mx.sym.expand_dims(mx.sym.broadcast_mul(lhs=kappa_0, rhs=mu_0), axis=1)
    vmf_prior = mx.sym.broadcast_add(lhs=vmf_normaliser(dim=dim, kappa=kappa_0),
                                     rhs=mx.sym.sum(mx.sym.broadcast_mul(lhs=prior_ss, rhs=target_direction), axis=2, keepdims=False))

    gamma_prior = mx.sym.Custom(shape=1, rate=1, label=kappa, op_type="gammaDist")
    model = likelihood + mx.sym.broadcast_add(lhs=vmf_prior, rhs=gamma_prior) + mx.sym.log(
        mx.sym.Activation(data=kappa, act_type="sigmoid", name="model_jacobian"))
    loss = mx.sym.MakeLoss(data=model)

    mod = mx.mod.Module(loss)

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
