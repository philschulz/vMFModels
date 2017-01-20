from gensim.models import Word2Vec

embeddings = Word2Vec.load_word2vec_format("/home/philip/Desktop/vectors/de/vectors-50dim.txt", binary=False)

print(embeddings.vector_size)