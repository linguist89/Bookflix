from bs4 import BeautifulSoup
import spacy
from sentence_transformers import SentenceTransformer

import text_similarity


def get_html_text(html_document):
    with open(html_document, 'r', encoding="utf-8") as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')
        return soup.text


def get_language_sentences(english_filename,
                           danish_filename):
    # Load spacy pipeline
    nlp = spacy.load("xx_sent_ud_sm")

    # Load the English and Danish texts
    english_text = get_html_text(english_filename)
    danish_text = get_html_text(danish_filename)

    # Calculate the average similarity scores
    en_sentences = []
    da_sentences = []
    eng_split = nlp(english_text.replace("\n", ""))
    for sent in eng_split.sents:
        en_sentences.append(sent)
    da_split = nlp(danish_text.replace("\n", ""))
    for sent in da_split.sents:
        da_sentences.append(sent)
    return en_sentences, da_sentences


def get_navigation_lists(current_position,
                         search_width=1):
    # Get forward and backward navigation list
    forward_positions = [current_position]
    backward_positions = [current_position]
    temp_forward_position = current_position
    temp_backward_position = current_position
    for i in range(search_width):
        temp_forward_position += 1
        forward_positions.append(temp_forward_position)
        if temp_backward_position > 0:
            temp_backward_position -= 1
            backward_positions.append(temp_backward_position)
    return forward_positions, backward_positions


def forward_backward_search(forward_positions,
                            backward_positions,
                            english_sentences,
                            danish_sentences):
    # Begin with the forward search
    temp_en_len = 0
    temp_da_len = 0
    results_list_en = []
    results_list_da = []
    joined_position_finder = 0
    for en_fp in forward_positions:
        if len(english_sentences) > en_fp and len(danish_sentences) > en_fp:
            temp_en_len += len(english_sentences[en_fp])
            temp_da_len += len(danish_sentences[en_fp])
            for da_fp in forward_positions:
                if len(danish_sentences) > da_fp:
                    results_list_en.append(len(english_sentences[en_fp]))
                    results_list_da.append(len(danish_sentences[da_fp]))
                    joined_position_finder += 1
                else:
                    break
        else:
            break
    results_list_en.append(temp_en_len)
    results_list_da.append(temp_da_len)
    return results_list_en, results_list_da, joined_position_finder


def find_closest_elements(length_list_en,
                          length_list_da):
    closest_list = []
    for en in length_list_en:
        for da in length_list_da:
            closest_list.append((length_list_en.index(en),
                                 length_list_da.index(da),
                                 abs(en - da)))
    return closest_list


def align_texts_on_lengths(english_sentences,
                           danish_sentences):
    aligned_list = []
    fp, bp = get_navigation_lists(0)
    while len(english_sentences) > 0 and \
            len(danish_sentences) > 0:
        temp_aligned = []
        en_lengths, da_lengths, joined_position_finder = forward_backward_search(fp,
                                                                                 bp,
                                                                                 english_sentences,
                                                                                 danish_sentences)
        most_likely = sorted(list(set(find_closest_elements(en_lengths, da_lengths))), key=lambda x: x[2])[0]
        if most_likely[0] == joined_position_finder:
            temp_aligned.append(' '.join([e.text for e in english_sentences[:joined_position_finder - 2]]))
            for i in range(joined_position_finder - 1):
                english_sentences.pop(0)
        else:
            temp_aligned.append(english_sentences[most_likely[0]].text)
            english_sentences.pop(0)
        if most_likely[1] == joined_position_finder:
            temp_aligned.append(' '.join([e.text for e in danish_sentences[:joined_position_finder - 2]]))
            for i in range(joined_position_finder - 1):
                danish_sentences.pop(0)
        else:
            temp_aligned.append(danish_sentences[most_likely[1]].text)
            danish_sentences.pop(0)
        aligned_list.append(temp_aligned)
    return aligned_list


def get_danish_coordinates(english_sentence,
                           danish_sentence_search_width,
                           current_position):
    temp_length_results = []
    for i, d_sentence in enumerate(danish_sentence_search_width):
        temp_length_results.append(([current_position + i], abs(len(english_sentence.text) - len(d_sentence.text))))
        temp_length_results.append(([current_position, current_position + i], abs(len(english_sentence.text) - len(
            ''.join([t.text for t in danish_sentence_search_width[0:i]])))))
    return temp_length_results


def align_texts_on_lengths_references(english_sentences,
                                      danish_sentences,
                                      search_width=3):
    # Reference_list should be the same length as the English sentences
    reference_list = []
    danish_coordinates = 0
    for i, english_sentence in enumerate(english_sentences):
        temp_reference_list = []
        if len(reference_list) == len(english_sentence):
            temp_reference_list.append(danish_coordinates)
        elif i + search_width < len(danish_sentences):
            danish_coordinates = sorted(get_danish_coordinates(english_sentence,
                                                               danish_sentences[i:i + search_width],
                                                               current_position=i), key=lambda x: x[1])[0]
            #print(danish_coordinates)
            temp_reference_list.append(danish_coordinates)
        else:
            temp_reference_list.append(None)
        reference_list.append(temp_reference_list)
    return reference_list


if __name__ == "__main__":
    """    
    1. Process for adding similarity scores has been added, but there are still problems.
    It seems to reorder the lists fine, but it needs to perform similarity scores because it's still out of sync.
    """
    english_sentences, danish_sentences = get_language_sentences("english_eu.html",
                                                                 "danish_eu.html")

    model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')

    coordinates = align_texts_on_lengths_references(english_sentences, danish_sentences)
    for i, c in enumerate(coordinates):
        print("----")
        print("Number: {}".format(i))
        if len(c) != 2:
            print("DANISH SENTENCE: {}".format(c[0][0][0]))
            print("English: {}\n Danish: {}".format(english_sentences[i], danish_sentences[c[0][0][0]]))
        else:
            print("DANISH SENTENCE: {}".format(c[0][0][0], c[0][0][1]))
            print("English: {}\n Danish: {}".format(english_sentences[i], danish_sentences[c[0][0][0], c[0][0][1]]))
        print("----")
