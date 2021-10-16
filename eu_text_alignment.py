from collections import Counter
import time
import spacy
import pandas as pd
import text_similarity
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# Try use multiple cores with pandas
import pandas as pd
import dask.dataframe as dd



def get_html_text(html_document):
    with open(html_document, 'r', encoding="utf-8") as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')
        return soup.text


def get_language_sentences(english_filename,
                           danish_filename):
    # Load spacy pipeline
    nlp = spacy.load("xx_sent_ud_sm")
    nlp_danish = spacy.load('da_core_news_lg')

    # Load the English and Danish texts
    english_text = get_html_text(english_filename)
    danish_text = get_html_text(danish_filename)

    # Calculate the average similarity scores
    en_sentences = []
    da_sentences = []
    eng_split = nlp(english_text.replace("\n", ""))
    for sent in eng_split.sents:
        en_sentences.append(sent)
    da_split = nlp_danish(danish_text.replace("\n", ""))
    for sent in da_split.sents:
        da_sentences.append(nlp(sent.text))
    return en_sentences, da_sentences


def remove_duplicates(lst):
    return list(set([i for i in lst]))


def match_english_sentences(english_sentences,
                            danish_sentences,
                            model,
                            search_space=3,
                            sim_threshold=0.7):
    english_sentence_results = []
    danish_sentence_results = []
    for i, english_sentence in enumerate(english_sentences):
        temp_abs = (100, 0, 0)
        for j in range(search_space):
            if len(danish_sentences) >= i + j:
                try:
                    current_abs = (abs(len(english_sentence) - len(danish_sentences[i + j])))
                    # Using spacy instead of the external sim script.
                    sim_score = text_similarity.text_similarity(english_sentence.text, danish_sentences[i + j].text,
                                                                model)
                    #sim_score = english_sentence.similarity(danish_sentences[i+j])
                    verb_sim_score = sim_score_pos(english_sentence, danish_sentences[i + j], pos_type="VERB")
                    noun_sim_score = sim_score_pos(english_sentence, danish_sentences[i + j], pos_type="NOUN")
                    if current_abs < temp_abs[0]:
                        temp_abs = (current_abs, i + j, sim_score, verb_sim_score, noun_sim_score)
                except:
                    print("There has been an error with len(danish_sentences) in def match_english_sentences")
            else:
                break
        dan_position = temp_abs[1]

        if temp_abs[3] > sim_threshold or temp_abs[4] > sim_threshold:
            print(i)
            print("SIM_SCORE:{}".format(temp_abs[2]))
            print("VERB_SIM_SCORE: {}".format(temp_abs[3]))
            print("NOUN_SIM_SCORE: {}".format(temp_abs[4]))
            print("ENGLISH: {} \n DANISH: {} \n DAN_POSITION: {}".format(english_sentence.text,
                                                                         danish_sentences[dan_position].text,
                                                                         dan_position))
            print("-----")
            english_sentence_results.append(
                ((english_sentence.text, i), (danish_sentences[dan_position].text, dan_position)))
        else:
            temp_en_sentence_search = [e[1][0] for e in english_sentence_results]
            if danish_sentences[dan_position].text not in temp_en_sentence_search:
                danish_sentence_results.append((danish_sentences[dan_position].text, dan_position))
    return english_sentence_results, remove_duplicates(danish_sentence_results)


def repeat_matching(english_sentence_results,
                    english_sentences,
                    danish_sentences,
                    model,
                    repeat=1,
                    search_space=5):
    for repeat_count in range(repeat):
        # print("ENG XXXXXX {}".format(english_sentence_results))
        # Remove English sentences already assigned corresponding Danish translations.
        if repeat_count == 0:
            english_positions = sorted([item[0][1] for item in english_sentence_results], reverse=True)
            print("FIRST EN POS: {}".format(english_positions))
        print("EN sentence len: {}".format(len(english_sentences)))
        for num in english_positions:
            del english_sentences[num]
        print("EN sentence len: {}".format(len(english_sentences)))

        # Remove Danish sentences already assigned corresponding English translations.
        if repeat_count == 0:
            danish_positions = sorted([item[1][1] for item in english_sentence_results], reverse=True)
            print("FIRST DA POS: {}".format(danish_positions))
        print("DA sentence len: {}".format(len(danish_sentences)))
        for num in danish_positions:
            del danish_sentences[num]
        print("DA sentence len: {}".format(len(danish_sentences)))

        # Repeat the alignment with new sentences
        internal_en_sentence_results, internal_da_sentence_results = match_english_sentences(english_sentences,
                                                                                             danish_sentences,
                                                                                             model,
                                                                                             search_space=search_space)

        english_positions = sorted([item[0][1] for item in internal_en_sentence_results], reverse=True)
        danish_positions = sorted([item[1][1] for item in internal_en_sentence_results], reverse=True)
        print("NEXT EN POS: {}".format(english_positions))
        print("NEXT DA POS: {}".format(danish_positions))

        if not internal_en_sentence_results:
            print("No matches - ending alignment.")
            break
        english_sentence_results += internal_en_sentence_results

    return english_sentence_results


def validate_results(series,
                     gold_standard):
    if series['English'] in gold_standard.keys():
        if series['Danish'] == gold_standard[series['English']]:
            return "Perfect Match"
        else:
            return "English Match"
    else:
        return "No Match"


def sim_score_pos(spacy_english_sentence,
                    spacy_danish_sentence,
                    pos_type="VERB"):
    average_verb_sim_score = 0
    internal_nlp_english = spacy.load('en_core_web_lg')
    internal_nlp_danish = spacy.load('da_core_news_lg')
    multilingual_model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
    internal_english_spacy = internal_nlp_english(spacy_english_sentence.text)
    internal_danish_spacy = internal_nlp_danish(spacy_danish_sentence.text)
    english_pos = [pos.text.lower() for pos in internal_english_spacy if pos.pos_ == pos_type]
    danish_pos = [pos.text.lower() for pos in internal_danish_spacy if pos.pos_ == pos_type]
    for ev in english_pos:
        ev_sim_score = 0
        for dv in danish_pos:
            temp_sim_score = text_similarity.text_similarity(ev, dv, multilingual_model)
            if temp_sim_score > ev_sim_score:
                ev_sim_score = temp_sim_score
        average_verb_sim_score += ev_sim_score
    if len(english_pos) != 0:
        return average_verb_sim_score/len(english_pos)
    else:
        return -1



    """for spacy_english_sentence in spacy_english_sentences:
        temp_english_verbs = [verb for verb in internal_nlp_english(spacy_english_sentence.text) if verb.pos_ == "VERB"]
        #temp_english_verbs = [verb for verb in internal_nlp_english(spacy_english_sentence.text) if verb.pos_ == "VERB"]
        english_verbs.append(temp_english_verbs)
    for spacy_danish_sentence in spacy_danish_sentences:
        temp_danish_verbs = [verb for verb in internal_nlp_danish(spacy_danish_sentence.text) if verb.pos_ == "VERB"]
        #temp_danish_verbs = [verb for verb in spacy_danish_sentence]
        danish_verbs.append(temp_danish_verbs)"""


def calc_sim_score(x, danish_column_name="Danish Spacy"):
    # Verbs
    en_verbs = [verb for verb in x['English Spacy'] if verb.pos_ == "VERB"]
    da_verbs = [verb for verb in x[danish_column_name] if verb.pos_ == "VERB"]
    """verbs_best_match = []
    for en_verb in en_verbs:
        temp_match_sim = 0
        temp_match_words = []
        for da_verb in da_verbs:
            new_match = en_verb.similarity(da_verb)
            if new_match > temp_match_sim:
                temp_match_sim = new_match
                temp_match_words = [en_verb, da_verb]
        verbs_best_match.append((temp_match_sim, temp_match_words))"""

    # Nouns
    en_nouns = [noun for noun in x['English Spacy'] if noun.pos_ == "NOUN"]
    da_nouns = [noun for noun in x[danish_column_name] if noun.pos_ == "NOUN"]
    """nouns_best_match = []
    for en_noun in en_nouns:
        temp_match_sim = 0
        temp_match_words = []
        for da_noun in da_nouns:
            new_match = en_noun.similarity(da_noun)
            if new_match > temp_match_sim:
                temp_match_sim = new_match
                temp_match_words = [en_noun, da_noun]
        nouns_best_match.append((temp_match_sim, temp_match_words))"""

    # Overall
    #da = nlp_multi(x['Danish'])
    #en = nlp_multi(x['English'])

    #results = x['English Spacy'].similarity(x['Danish Spacy']), verbs_best_match, nouns_best_match
    #results = [en_verbs, da_verbs, en_nouns, da_nouns]
    if not isinstance(x[danish_column_name], str):
        results = x['English Spacy'].similarity(x[danish_column_name])
    else:
        results = 0.0
    return results


if __name__ == "__main__":
    start = time.time()
    # da_en statistics (pandas parallel - confirmed)
    df = pd.read_csv("vikingeskibsmuseet_da_en_statistic.csv")
    print("Length of total df: {}".format(len(df)))
    df = pd.DataFrame(df[:100])
    print(df.head())
    exit()
    nlp_multi = spacy.load("xx_paraphrase_xlm_r_multilingual_v1")
    nlp_english = spacy.load("en_core_web_lg")
    nlp_danish = spacy.load('da_core_news_lg')
    nlp_english.add_pipe("sentence_bert", source=spacy.load("xx_paraphrase_xlm_r_multilingual_v1"))
    nlp_danish.add_pipe("sentence_bert", source=spacy.load("xx_paraphrase_xlm_r_multilingual_v1"))
    df['English Spacy'] = df['English'].apply(lambda x: nlp_english(x))
    df['Danish Spacy'] = df['Danish'].apply(lambda x: nlp_danish(x))
    df['Danish Spacy1'] = df['Danish Spacy'].shift(1)
    df['Danish Spacy2'] = df['Danish Spacy'].shift(2)
    df['Danish Spacy3'] = df['Danish Spacy'].shift(3)
    df['Danish Spacy4'] = df['Danish Spacy'].shift(4)
    df['Danish Spacy5'] = df['Danish Spacy'].shift(-1)
    df['Danish Spacy6'] = df['Danish Spacy'].shift(-2)
    df['Danish Spacy7'] = df['Danish Spacy'].shift(-3)
    df['Danish Spacy8'] = df['Danish Spacy'].shift(-4)
    df.fillna("", inplace=True)
    #Try use Bleualign with splitting the documents at \n\n...

    # Dask
    ddf = dd.from_pandas(df, npartitions=16)
    for i in range(9):
        if i == 0:
            ddf['Sim_Scores'] = ddf.apply(calc_sim_score, axis=1, danish_column_name="Danish Spacy", meta=float)
        elif i != 3 or i != 4 or i != 7 or i != 8:
            ddf['Sim_Scores{}'.format(i)] = ddf.apply(calc_sim_score, axis=1, danish_column_name="Danish Spacy{}".format(i), meta=float)
    df_csv_save = pd.DataFrame(ddf)
    df_csv_save.to_csv("sim_scores_4up_4down.csv")
    print(ddf.head())
    print(ddf.columns)
    for j, row in ddf.iterrows():
        print(row['Sim_Scores'])
        print(row['Sim_Scores1'])
        print(row['Sim_Scores2'])
        print(row['Sim_Scores3'])
        print(row['Sim_Scores4'])
        print(row['Sim_Scores5'])
        print(row['Sim_Scores6'])
        print(row['Sim_Scores7'])
        print(row['Sim_Scores8'])
        print("-----")
        if j == 5:
            break
    end = time.time()
    print("Number of partitions: {}".format(16))
    print((end-start)/60)

"""if __name__ == "__main__":

    start = time.time()
    # da_en statistics (pandas parallel - confirmed)
    df = pd.read_csv("vikingeskibsmuseet_da_en_statistic.csv")
    gold_standard = {}
    for i, row in df.iterrows():
        gold_standard.update({row['English']: row['Danish']})
    df = pd.DataFrame(df[:150])
    gold_standard_df = pd.DataFrame(df)
    end = time.time()
    print("Time taken for loading: {} minutes.".format((end-start)/60))

    start = time.time()
    nlp_multi = spacy.load("xx_paraphrase_xlm_r_multilingual_v1")
    nlp_danish = spacy.load('da_core_news_lg')
    # Base sentence segmentation code
    #english_sentences = [nlp(sent) for sent in df['English'].to_list()[:100] if not isinstance(sent, float)]
    #danish_sentences = [nlp(sent) for sent in df['Danish'].to_list()[:100] if not isinstance(sent, float)]


    eng_split = nlp_multi(' '.join(df['English'].to_list()))
    da_split = nlp_danish(' '.join(df['Danish'].to_list()))
    english_sentences = []
    danish_sentences = []
    for sent in eng_split.sents:
        english_sentences.append(sent)
    for sent in da_split.sents:
        danish_sentences.append(nlp_multi(sent.text))
    end = time.time()
    print("Time taken for intial Spacy Load: {} minutes.".format((end - start) / 60))

    start = time.time()
    print("eng_split: {} \n english_sentences: {}".format(len(eng_split), len(english_sentences)))
    print("da_split: {} \n danish_sentences: {}".format(len(da_split), len(danish_sentences)))
    model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
    #gold_standard_df['Similarity Score'] = gold_standard_df.apply(lambda x: text_similarity.text_similarity(x['English'],
    #                                                                                                        x['Danish'],
    #                                                                                                        embedding_model=model),
    #                                                              axis=1)
    ddf = dd.from_pandas(gold_standard_df, npartitions=16)

    ddf['Similarity Score'] = ddf.apply(calc_sim_score, axis=1, meta=float)
    gold_standard_df = pd.DataFrame(ddf)
    gold_standard_df.to_csv("vikingeskibsmuseet_gold_standard.csv", index=None)

    en_sentence_results, da_sentence_results = match_english_sentences(english_sentences,
                                                                       danish_sentences,
                                                                       model,
                                                                       search_space=3)
    end = time.time()
    print("Time taken for gold_standard: {} minutes.".format((end - start) / 60))

    start = time.time()
    # English Sentences
    print("===ENGLISH RESULTS ===")
    for e in en_sentence_results:
        print(e)
        print("----")
    repeat = 3
    search_space = 2
    repeating_results = repeat_matching(en_sentence_results,
                                        english_sentences,
                                        danish_sentences,
                                        model=model,
                                        # I can try adjusting this to see if the repeat value improves accuracy.
                                        repeat=repeat,
                                        # Same with search space..
                                        search_space=search_space)
    en_repeating_results = sorted(repeating_results, key=lambda x: x[0][1])
    end = time.time()
    print("Time taken for repeating section: {} minutes.".format((end - start) / 60))

    start = time.time()
    df_final_results_list = []
    for e in en_repeating_results:
        print("ENGLISH: {} \n DANISH: {}".format(e[0], e[1]))
        df_final_results_list.append([e[0][0], e[1][0]])
        print("---")
    df_final_results = pd.DataFrame(df_final_results_list, columns=["English", "Danish"])
    df_final_results['Match'] = df_final_results.apply(lambda x: validate_results(x,
                                                                                  gold_standard=gold_standard), axis=1)

    df_final_results['Sim_Score'] = df_final_results.apply(lambda x: text_similarity.text_similarity(x['English'],
                                                                                                     x['Danish'],
                                                                                                     embedding_model=model), axis=1)
    df_final_results.to_csv("vikingeskibsmuseet_final_matched_r{}_ss{}.csv".format(repeat, search_space), index=None)
    print(df_final_results)
    print(Counter(df_final_results['Match'].to_list()))
    end = time.time()
    print("Time taken for final matching: {} minutes.".format((end - start) / 60))"""
