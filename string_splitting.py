import pandas as pd
import pysbd
import read_epub
from translate import Translator
from scipy.stats import wasserstein_distance
translator = Translator(to_lang="en", from_lang="da")


en_seg = pysbd.Segmenter(language="en", clean=False)
da_seg = pysbd.Segmenter(language="da", clean=False)


def create_translations():
    danish = read_epub.extract_contents("Den_lille_prins.epub")
    english = read_epub.extract_contents("The_little_prince.epub")
    d_split = danish.split("\n\n\n")
    e_split = english.split("\n\n")
    tran_list = [translator.translate(d) for d in d_split]
    with open("d_split.txt", "w", encoding="utf-8") as f:
        f.writelines(d_split)
    with open("e_split.txt", "w", encoding="utf-8") as f:
        f.writelines(e_split)
    with open("d_translated.txt", "w", encoding="utf-8") as f:
        f.writelines(tran_list)
# Press the green button in the gutter to run the script.


def get_wasserstein_distances(d_text,
                              e_text,
                              ideal_distance=100):
    print("Len of d_text: {}".format(len(d_text)))
    print("Len of e_text: {}".format(len(e_text)))
    d_num = 0
    e_num = 0
    for i in range(2):
        for j in range(2):
            d_lengths = [len(t.split()) for t in d_text if len(t.split()) > i]
            e_lengths = [len(t.split()) for t in e_text if len(t.split()) > j]
            distance = wasserstein_distance(d_lengths, e_lengths)
            if distance < ideal_distance:
                ideal_distance = distance
                d_num = i
                e_num = j
    print("Ideal Distance: {}".format(ideal_distance))
    print("d_num = {}".format(d_num))
    print("e_num = {}".format(e_num))
    d_filtered = [t for t in d_text if len(t.split()) > d_num]
    e_filtered = [t for t in e_text if len(t.split()) > e_num]
    print("Filtered d_length".format(len(d_filtered)))
    print("Filtered e_length".format(len(e_filtered)))
    for count in range(11):
        print(d_filtered[count])
        print(e_filtered[count])
        print("----")


if __name__ == '__main__':
    with open("d_split.txt") as f:
        d_text = f.read()
    with open("e_split.txt") as f:
        e_text = f.read()
    #print(len(d_text))
    full_d_text = []
    for split_d_text in d_text.split("\n\n"):
        temp_d_text = da_seg.segment(split_d_text)
        full_d_text.append(temp_d_text)
    full_d_text = [i for j in full_d_text for i in j]
    full_e_text = []
    for split_e_text in e_text.split("\n\n"):
        temp_e_text = en_seg.segment(split_e_text)
        full_e_text.append(temp_e_text)
    full_e_text = [i for j in full_e_text for i in j]
    get_wasserstein_distances(full_d_text, full_e_text)


