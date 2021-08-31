import sys
import ebooklib
import pandas as pd
from ebooklib import epub
from bs4 import BeautifulSoup as bs
from sentence_transformers import SentenceTransformer

import alignment
import text_similarity


def extract_contents(book_filename):
    # Load epub book into paragraphs
    book = epub.read_epub(book_filename)
    all_paragraphs = []
    for token in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = bs(token.get_content())
        input_soup = [soup.text.strip()]
        all_paragraphs.append(input_soup)
        # print(input_soup)
    text = ' '.join([' '.join(text) for text in all_paragraphs])
    return text


def determine_paragraphs(epub_text):
    newline_list = []
    newline_continue = True
    newline_count = 1
    while newline_continue:
        temp_newline = "\n" * newline_count
        newline_num = epub_text.count(temp_newline)
        if newline_num != 0:
            newline_list.append((newline_count, newline_num))
            newline_count += 1
        else:
            newline_continue = False
    return newline_list


def count_paragraph_lengths(newline_list,
                            epub_text):
    paragraph_count_dict = {}
    for element in newline_list:
        split_text = [p for p in epub_text.split("\n" * element[0])]
        split_text_lengths = [len(p) for p in split_text]
        paragraph_count_dict.update({element[0]: (
            element[1], split_text_lengths, int(sum(split_text_lengths) / len(split_text_lengths)), split_text)})
    return paragraph_count_dict


if __name__ == "__main__":
    # English Book
    e_epub_contents = extract_contents(sys.argv[1])
    e_newlines = determine_paragraphs(e_epub_contents)
    e_paragraph_lengths = count_paragraph_lengths(e_newlines, e_epub_contents)
    e_test = []
    e_text_test = []
    for p in e_paragraph_lengths.items():
        e_test.append(p[1][1])
        e_text_test.append(p[1][3])

    # Danish Book
    d_epub_contents = extract_contents(sys.argv[2])
    d_newlines = determine_paragraphs(d_epub_contents)
    d_paragraph_lengths = count_paragraph_lengths(d_newlines, d_epub_contents)
    d_test = []
    d_text_test = []
    for p in d_paragraph_lengths.items():
        d_test.append(p[1][1])
        d_text_test.append(p[1][3])

    # Find alignment location
    alignment_location = alignment.align_text(e_text_test, d_text_test)
    model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
    temp_score = 0
    temp_e_section = ""
    temp_d_section = ""
    for i in range(5):
        e_section = e_text_test[alignment_location[0]][i]
        d_section = d_text_test[alignment_location[1]][i]
        sim_score = text_similarity.text_similarity(e_section,
                                                    d_section,
                                                    model)
        if sim_score > temp_score:
            temp_score = sim_score
            temp_e_section = e_section, (alignment_location[0], i)
            temp_d_section = d_section, (alignment_location[1], i)

    """
    ###########################
    1. Check scores of perfectly aligned sentences e.g. UN documents or Danish Parliament
    2. Check how those scores change when noise is added.
    3. Find correlations between the 1-to-1 translations and noise.
    ###########################
    """
    first_e = temp_e_section[1][0]
    second_e = temp_e_section[1][1]
    first_d = temp_d_section[1][0]
    second_d = temp_d_section[1][1]

    for i in range(3):
        print(e_text_test[first_e][second_e+i+i])
        print("---")
        print(d_text_test[first_d][second_d+i])
        print("===")

    """from scipy.stats import linregress
    for pos in test:
        a = pos
        b = [pos.index(num) for num in pos]
        print(linregress(a, b)[0])"""
