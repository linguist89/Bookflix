import time
from collections import Counter

import pysbd
import read_epub
import numpy as np
import pandas as pd
from translate import Translator
from sentence_transformers import SentenceTransformer

from numba import jit


@jit(nopython=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta


def cosine_similarity(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta


def extract_epub_contents(d_filename,
                          e_filename):
    danish = read_epub.extract_contents(d_filename)
    english = read_epub.extract_contents(e_filename)
    return danish, english


def check_match(da_vector, en_vector_list, cutoff=0.9):
    sim_score = (0, [])
    for i, en_vector in enumerate(en_vector_list):
        temp_sim_score = cosine_similarity_numba(da_vector, en_vector)
        if temp_sim_score > sim_score[0]:
            sim_score = (temp_sim_score, "{}: {}".format(i, e_split[i]))
    if sim_score[0] > cutoff:
        return sim_score
    else:
        return (0.0, "No Match")


if __name__ == '__main__':
    """Create an anchor point by manually checking the first matching paragraphs.
    based on this, try match the rest via computational methods...."""
    multilingual_model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
    d_text, e_text = extract_epub_contents("Den_lille_prins.epub", "The_little_prince.epub")
    newlines = "\n"
    temp_split = [len(t.split()) for t in d_text.split(newlines) if len(t.split()) > 0]
    temp_length = sum(temp_split)/len(temp_split)
    da_count = 0
    da_start = 0
    sentence_split = True
    en_seg = pysbd.Segmenter(language="en", clean=False)
    da_seg = pysbd.Segmenter(language="da", clean=False)
    d_split = [t for t in d_text.split(newlines) if len(t) > temp_length]
    # Split into sentences
    if sentence_split == True:
        d_split = [da_seg.segment(text) for text in d_split]
        d_split = [i for j in d_split for i in j]
    for i, section in enumerate(d_split):
        if len(section.split()) > temp_length:
            print(i)
            da_start = i
            da_count += 1
            if da_count == 1:
                break
    print("----")
    temp_split = [len(t.split()) for t in e_text.split(newlines) if len(t.split()) > 0]
    temp_length = sum(temp_split) / len(temp_split)
    en_count = 0
    en_start = 0
    e_split = [t for t in e_text.split(newlines) if len(t) > temp_length]
    if sentence_split == True:
        e_split = [en_seg.segment(text) for text in e_split]
        e_split = [i for j in e_split for i in j]
    for i, section in enumerate(e_split):
        if len(section.split()) > temp_length:
            print(i)
            en_start = i
            en_count += 1
            if en_count == 1:
                break
    print("=====")

    test_list = []
    if len(d_split) > len(e_split):
        end = len(d_split)
        padding = len(d_split) - len(e_split)
        for i in range(padding):
            e_split.append("None")
    else:
        end = len(e_split)
        padding = len(e_split)-len(d_split)
        for i in range(padding):
            d_split.append("None")
    for i in range(end):
        temp_da = multilingual_model.encode(d_split[i])
        temp_en = multilingual_model.encode(e_split[i])
        test_list.append((temp_da, temp_en))

    start = time.time()
    da_total = [np.array(e[0]) for e in test_list]
    en_total = [np.array(e[1]) for e in test_list]
    df_da = pd.DataFrame([da_total, d_split]).transpose()
    cutoff = 0.9
    type = "1newlines"
    df_da['Best Sim Score'] = df_da.apply(lambda x: check_match(x[0], en_total, cutoff), axis=1)
    end = time.time()
    print("Time for numba {}".format(end - start))
    df_da.to_csv("sentences_df_da_cutoff_{}%_{}.csv".format(int(cutoff*100), type), index=None)
    english_sentence_references = []
    for i, row in df_da.iterrows():
        print("{} {}".format(i, row[1]))
        print("-----")
        print(row['Best Sim Score'][0])
        print("-----")
        english_sentence = row['Best Sim Score'][1]
        if english_sentence != "No Match":
            english_sentence_number = int(english_sentence.split(":")[0])
            english_sentence_references.append(english_sentence_number)
        else:
            english_sentence_references.append(-1)
        print(row['Best Sim Score'][1])
        print("-----")
        print("=====")

    # Check the counts of No Match
    no_match_list = []
    print("Total length of df: {}".format(len(df_da)))
    for e in Counter(df_da['Best Sim Score'].to_list()).items():
        no_match_list.append(e)
    print(sorted(no_match_list, key=lambda x: x[1], reverse=True)[:5])

    temp_references = [(e, english_sentence_references.index(e)) for e in english_sentence_references if e != -1]
    print(sorted(temp_references, key=lambda x: x[0]))

    def back_forward_check(temp_list):
        new_list = []
        for i, e in enumerate(temp_list):
            if 0 < i < len(temp_list)-1:
                if temp_list[i-1][1] < e[1] < temp_list[i+1][1] and temp_list[i-1][0] < e[0] < temp_list[i+1][0]:
                    new_list.append(e)
            elif i == 0 and e[1] < temp_list[i+1][1] and e[0] < temp_list[i+1][0]:
                new_list.append(e)
        print("Temp_list: {}".format(len(temp_list)))
        print("New_list: {}".format(len(new_list)))
        return new_list
    new_list = back_forward_check(temp_references)
    for e in new_list[:5]:
        print("New list - element {} - English: {}".format(new_list.index(e), e_split[e[0]]))
        print("---")
        print("New list - element {} - Danish: {}".format(new_list.index(e), d_split[e[1]]))
        print("===")


    exit()
    start = time.time()
    sim_scores_numba = []
    num = 30
    for count in range(num):
        for i, first_t in enumerate(test_list):
            for j, second_t in enumerate(test_list):
                sim_scores_numba.append(cosine_similarity_numba(test_list[i][0], test_list[j][1]))
    end = time.time()
    print("Numba num {}".format(num))
    print("Time for numba {}".format(end - start))
    print("Length for numba {}".format(len(sim_scores_numba)))
    print("First 10 for numba {}".format(sim_scores_numba[:10]))

    start = time.time()
    sim_scores_regular = []
    num = 5
    for count in range(num):
        for i, first_t in enumerate(test_list):
            for j, second_t in enumerate(test_list):
                sim_scores_regular.append(cosine_similarity(test_list[i][0], test_list[j][1]))
        #print("---")
    end = time.time()
    print("Regular num {}".format(num))
    print("Time for regular {}".format(end - start))
    print("Length for regular {}".format(len(sim_scores_regular)))
    print("First 10 for regular {}".format(sim_scores_regular[:10]))

    exit()
    da_df = pd.DataFrame(d_split[da_start:])
    print("Length of Danish {}".format(len(da_df)))
    print(da_df.head())
    print("---")
    en_df = pd.DataFrame(e_split[en_start:])
    print("Length of English {}".format(len(en_df)))
    print(en_df.head())


