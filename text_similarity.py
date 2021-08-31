from sentence_transformers import util
import numpy as np


def text_similarity(text1,
                    text2,
                    embedding_model):

    # encode sentences to get their embeddings
    embedding1 = embedding_model.encode(text1, convert_to_tensor=True)
    embedding2 = embedding_model.encode(text2, convert_to_tensor=True)
    # compute similarity scores of two embeddings
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_scores.item()


# Calculate Wasserstein Distance values for all perturbations
"""    from scipy.stats import wasserstein_distance
    wasser_list = []
    for e in e_test:
        for d in d_test:
            print("English: {}, Danish: {} (Wasserstein_distance = {})".format(e_test.index(e),
                                                                               d_test.index(d),
                                                                               wasserstein_distance(e, d)))
            wasser_list.append((e_test.index(e),
                                d_test.index(d),
                                wasserstein_distance(e, d)))
    wasser_list = sorted(wasser_list, key=lambda x: x[2])[:5]
    for w in wasser_list[:5]:
        print("---")
        print("English @ {}: {}".format(w[0], e_text_test[w[0]][4][:100]))
        print("Danish @ {}: {}".format(w[1], d_text_test[w[1]][4][:100]))
        print("---")"""
