import pandas as pd
import read_epub
from translate import Translator
from scipy.stats import wasserstein_distance
translator = Translator(to_lang="en", from_lang="da")


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


if __name__ == '__main__':
    with open("d_split.txt") as f:
        d_text = f.readlines()
    with open("e_split.txt") as f:
        e_text = f.readlines()
    for i in range(100):
        d_lengths = [len(t.split()) for t in d_text if len(t.split()) > i]
        e_lengths = [len(t.split()) for t in e_text if len(t.split()) > i]
        print(wasserstein_distance(d_lengths, e_lengths))
