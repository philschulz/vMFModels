import argparse, gensim, logging, codecs
from os.path import join

sg = 1
iter = 5
negative_samples = 10
vector_file = "vectors"
vocab_file = "vocab"


class TextIterator(object):
    def __init__(self, file_name):
        self.file = file_name

    def __iter__(self):
        for line in codecs.open(self.file, encoding="utf-8"):
            yield line.split()


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    command_line_parser = argparse.ArgumentParser("Generates word vectors from text using skipgram.\n")

    command_line_parser.add_argument("--threads", type=int, default=1,
                                     help="Specify the number of threads used while generating the vectors. "
                                          "Default: %(default).")
    command_line_parser.add_argument("--dims", type=int, default=50,
                                     help="Specify the dimensionality of the wordvectors. Default: %s(default).")
    command_line_parser.add_argument("--output-dir", type=str,
                                     help="Specify a path to an output directory. If no such path is given, the output "
                                          "will be stored in the working directory")
    command_line_parser.add_argument("--min-count", type=int, default=3,
                                     help="Minimal count of tokens for a type to be embedded. Default: %s(default).")
    command_line_parser.add_argument("file", help="The input text file.")

    args = command_line_parser.parse_args()

    threads = args.threads
    dims = args.dims
    train_file = args.file
    output_dir = args.output_dir
    min_count = args.min_count

    global vector_file
    global vocab_file
    vector_file += "-{}dim.txt".format(dims)

    if output_dir is not None:
        vector_file = join(output_dir, vector_file)
        vocab_file = join(output_dir, vocab_file)

    training_stream = TextIterator(train_file)

    model = gensim.models.word2vec.Word2Vec(size=dims, min_count=min_count, sg=sg, negative=negative_samples,
                                            workers=threads, iter=iter)
    model.build_vocab(training_stream)
    model.train(training_stream)

    model.save_word2vec_format(vector_file, vocab_file, binary=False)


if __name__ == "__main__":
    main()
