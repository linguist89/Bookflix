from sentence_transformers import SentenceTransformer
import text_similarity


def align_text(e_text_test, d_text_test):
    # Calculate Similarity
    model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
    for e_i, e in enumerate(e_text_test):
        try:
            for d_i, d in enumerate(d_text_test):
                for i in range(10):
                    english_section = e_text_test[e_i][i]
                    danish_section = d_text_test[d_i][i]
                    if len(danish_section) > 10 and len(english_section) > 10:
                        sim_score = text_similarity.text_similarity(english_section, danish_section, embedding_model=model)
                        if sim_score > 0.9:
                            """print("English num: {} (text length {})".format(e_i, len(english_section)))
                            print("Danish num: {} (text length {})".format(d_i, len(danish_section)))
                            print("Internal num: {}".format(i))
                            print(sim_score)"""
                            return e_i, d_i
        except:
            print("Error at {} {}".format(e_i, d_i))

