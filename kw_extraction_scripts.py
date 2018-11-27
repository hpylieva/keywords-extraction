import re
from typing import List

import nltk
import string
import pymorphy2
import numpy as np
import pandas as pd
import math
from rake import *
from rutermextract import TermExtractor
from operator import itemgetter
from bs4 import BeautifulSoup

MAX_WORDS_IN_KEYPHRASE = 3
N_GENERATED_KEYWORDS = 7

cyrrilic = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
tokenizer = nltk.RegexpTokenizer(r'\w+')
morph_ru = pymorphy2.MorphAnalyzer()
term_extractor = TermExtractor()
rake = RakeKeywordExtractor()


def filter_phrases_by_len(phrase_list: List[str], min_len=2, max_len=3) -> List[str]:
    """
    :param phrase_list: list of clean phrases (no punctuation)
    :param min_len:
    :param max_len:
    :return:
    """
    filtered_ph_list = []
    for p in phrase_list:
        ph_len = len(p.split())
        if min_len <= ph_len <= max_len:
            filtered_ph_list.append(p)
    return filtered_ph_list


def ensure_nltk_data_is_loaded():
    try:
        nltk.data.find('averaged_perceptron_tagger_ru')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_ru')
        # log.info('Checking NLTK data')
    try:
        nltk.data.find('averaged_perceptron_tagger_ru')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


def clean(input_text):
    text_lower = input_text.lower()
    # printable = set(cyrrilic + string.printable)
    # #     print(printable)
    # # filter funny characters, if any.
    # text = re.sub('[^{}]'.format(printable), '', text_lower)
    return text_lower


def textrank(input_text, return_phrase_score=False):
    ensure_nltk_data_is_loaded()

    cleaned_text = clean(input_text)
    text = tokenizer.tokenize(cleaned_text)
    lemmatized_text = [morph_ru.parse(word)[0].normal_form for word in text]
    POS_tag = nltk.pos_tag(lemmatized_text, lang='rus')

    stopwords = []
    adjective_tags = ['A=m', 'A=f', 'A=pl', 'A=n', 'A-NUM']
    wanted_POS = ['S',
                  #               'V'
                  ] + adjective_tags

    for word in POS_tag:
        if word[1] not in wanted_POS:
            stopwords.append(word[0])

    punctuations = list(str(string.punctuation))
    stopwords = stopwords + punctuations
    stopword_file = open("long_stopwords_ru.txt", "r")
    # Source = https://www.ranks.nl/stopwords

    lots_of_stopwords = []

    for line in stopword_file.readlines():
        lots_of_stopwords.append(str(line.strip()))

    stopwords_plus = stopwords + lots_of_stopwords
    stopwords_plus = set(stopwords_plus)
    processed_text = []
    for word in lemmatized_text:
        if word not in stopwords_plus:
            processed_text.append(word)
    # print (processed_text)
    vocabulary = list(set(processed_text))
    vocab_len = len(vocabulary)
    weighted_edge = np.zeros((vocab_len, vocab_len), dtype=np.float32)

    score = np.zeros((vocab_len), dtype=np.float32)
    window_size = 3
    covered_coocurrences = []

    for i in range(0, vocab_len):
        score[i] = 1
        for j in range(0, vocab_len):
            if j == i:
                weighted_edge[i][j] = 0
            else:
                for window_start in range(0,
                                          (len(processed_text) - window_size)):

                    window_end = window_start + window_size

                    window = processed_text[window_start:window_end]

                    if (vocabulary[i] in window) and (vocabulary[j] in window):

                        index_of_i = window_start + window.index(vocabulary[i])
                        index_of_j = window_start + window.index(vocabulary[j])

                        # index_of_x is the absolute position of the xth term in the window
                        # (counting from 0)
                        # in the processed_text

                        if [index_of_i, index_of_j] not in covered_coocurrences:
                            weighted_edge[i][j] += 1 / math.fabs(
                                index_of_i - index_of_j)
                            covered_coocurrences.append(
                                [index_of_i, index_of_j])
    inout = np.zeros((vocab_len), dtype=np.float32)

    for i in range(0, vocab_len):
        for j in range(0, vocab_len):
            inout[i] += weighted_edge[i][j]

    max_iterations = 50
    d = 0.85
    threshold = 0.0001  # convergence threshold

    for iter in range(0, max_iterations):
        prev_score = np.copy(score)
        for i in range(0, vocab_len):
            summation = 0
            for j in range(0, vocab_len):
                if weighted_edge[i][j] != 0:
                    summation += (weighted_edge[i][j] / inout[j]) * score[j]

            score[i] = (1 - d) + d * summation
        if np.sum(np.fabs(prev_score - score)) <= threshold:  # convergence condition
            # print("Converging at iteration " + str(iter) + "....")
            break

    phrases = []
    phrase = " "
    for word in lemmatized_text:
        if word in stopwords_plus:
            if phrase != " ":
                phrases.append(str(phrase).strip().split())
            phrase = " "
        elif word not in stopwords_plus:
            phrase += str(word)
            phrase += " "
    unique_phrases = []

    for phrase in phrases:
        if phrase not in unique_phrases:
            unique_phrases.append(phrase)

    for word in vocabulary:
        # print word
        for phrase in unique_phrases:
            if (word in phrase) and ([word] in unique_phrases) and (
                    len(phrase) > 1):
                # if len(phrase)>1 then the current phrase is multi-worded.
                # if the word in vocabulary is present in unique_phrases as a single-word-phrase
                # and at the same time present as a word within a multi-worded phrase,
                # then I will remove the single-word-phrase from the list.
                unique_phrases.remove([word])
    phrase_scores = []
    keywords = []
    for phrase in unique_phrases:
        phrase_score = 0
        keyword = ''
        for word in phrase:
            keyword += str(word)
            keyword += " "
            phrase_score += score[vocabulary.index(word)]
        phrase_scores.append(phrase_score)
        keywords.append(keyword.strip())
    filtered_keywords = [kw for kw in keywords if
                         len(kw.split()) <= MAX_WORDS_IN_KEYPHRASE]
    # change order of key phrases to descending my score
    kw_score = list(zip(filtered_keywords, phrase_scores))
    kw_score.sort(key=itemgetter(1), reverse=True)
    # TODO: make it more beautiful
    if return_phrase_score:
        return [i[0] for i in kw_score], [i[1] for i in kw_score]
    else:
        return [i[0] for i in kw_score]


def extract_keywords(text, method):
    if method == 'textrank':
        terms = textrank(text)
    elif method == 'rake':

        terms = list(rake.extract(text, incl_scores=False))
    elif method == 'rtex':
        terms = [term.normalized for term in term_extractor(text)]
    # filter terms by length
    return filter_phrases_by_len(terms, 1, 1), filter_phrases_by_len(terms, 2, 2), filter_phrases_by_len(terms, 3, 3)
    # filtered_terms = [kw for kw in terms if
    #                   len(kw.split()) <= MAX_WORDS_IN_KEYPHRASE]
    # return filtered_terms


if __name__ == '__main__':
    text = [u"""
    Одной из популярных позиций нашего производства является угловой диван Респект. Если Вы современный и деловой человек, любите комфорт и уют, а каждый сантиметр жилплощади для Вас на вес золота – тогда угловой диван Респект отлично впишется в интерьер Вашей квартиры, заняв при этом минимум места. Угловой диван Респект не только практичен в использовании, удобен, мягок и современен, но и еще удивительно компактен. В его изготовлении используются только самые экологически чистые ткани и наполнители, а механизм раскладки еврокнижка - поможет Вам сэкономить массу усилий и времени для его преобразования в полноценную кровать.
     Модель со схемой раскладки еврокнижка, сделана на основе деревянного каркаса с использованием пружины типа Боннель в паре с пружинной змейкой.
     Особенностью этого дивана является небольшой габаритный размер короткой части модели 135см, что позволяет устанавливать его в наши малогабаритные квартиры. Стильные боковые перила в виде полудуги придают угловому дивану Респект чрезвычайной элегантности.
     Как и в большинстве моделей нашего производства в модели Респект есть возможность добавления на спальную и сидячую части натуральных наполнителей (кокос, латекс, спрут, шерсть и др.), что придаст модели более выраженный ортопедический эффект. В конструкцию углового дивана заложено два больших места для хранения постельного белья, одно под сидячей продольной частью, второе в угловой части. Угловой диван Респект комплектуется тремя большими подушками для удобного сидения в собранном виде.
     Все материалы, которые используются при производстве углового дивана Респект проверены и сертифицированы, на весь модельный ряд и в частности на эту модель распространяется гарантийное и послегарантийное обслуживание.
    """,
            u"""
    Уважаемые покупатели. Перед тем, как делать заказ, убедитесь в наличии вашего размера и нужного цвета ( все размеры и цвета в наличии указаны в тексте объявления). Уточнить можно в вайбер 0503952439, по телефону или в чате на странице нашей компании.
    Отправляем наложенным платежом при условии внесения аванса 150 грн на карту Приватбанка 5167 9855 0018 7539 Андреева Наталия.
    Женский длинный пуховик Элис
    Верхний материал полиэстер с водоотталкивающей пропиткой. Наполнитель - искусственный пух (тинсулейт)
    Силуэт слегка приталенный. Застёжка - молния и кнопки.  Вшитый капюшон со съёмным искусственным мехом. Два боковых кармана.
    Цвета  и размеры в наличии - черный ( 46, 48) оливковый ( 48)  красный (44, 46)
    Параметры по куртке:
    44 размер - объем груди 104 см, плечи 42 см, длина рукава 64 см, длина куртки - 100 см
    46 размер - объем груди 108 см, плечи 43 см, длина рукава 65 см, длина куртки - 100 см
    48 размер - объем груди 112 см, плечи 44 см, длина рукава 66 см, длина куртки - 100 см
    """,
            u"""
    Наклейки виниловые на окно С 30453 (60) /ЦЕНА ЗА УПАКОВКУ/ 12шт в упаковке
    Новогоднее украшение.
    Вид - наклейка на окно.
    Материал - винил.
    * Характеристики и комплектация товара могут изменяться производителем без уведомления
     Мы рады приветствовать Вас в нашем интернет-магазине!!! 
    Мы предлагаем вам Товары , которые вы можете приобрести по самым выгодным условиям и ценам. Наша продукция соответствует всем нормам и требованиям качества.
    Предлагаемый вам товар, рассчитан главным образом на самых маленьких потребителей – наших детей. Уровень и качество нашего товара тщательно проверяется, и вы приобретаете самые лучшие детские игрушки.
    Все лучшее – детям! Такой девиз соблюдаем мы, предлагая вам высококлассный товар. Интернет-магазин позволят вам выбрать именно то, что необходимо ребенку. Известно, что дети начинают развиваться с момента рождения. Уже в возрасте нескольких месяцев ребенку необходимы развивающие игры, которые помогают ускорить процесс становления маленького человека. Лучшим способом обучения ребенка является игра, а в процессе игры развивается память, смекалка и логика.
    Ассортимент детских игрушек, предлагает товар, подходящий для разных покупателей и возможностей. При выборе игрушек также важно учитывать способности и таланты вашего малыша. Покупка ребенку игрушки дает возможность развить в нем новые таланты и способности. Поэтому менее важны строгие различия между мальчиками и девочками. Для полноценного и правильного развития, ребенку необходимо присутствие разных игр и игрушек. Девочкам крайне необходимо помимо красивых кукол и мягких мишек, наличие машинок. А мальчику кроме солдатиков и танков помогут развиваться и раскрываться машинки и конструкторы.
    Мы предлагаем детские игрушки, которые удовлетворят даже самого взыскательного клиента. Наша продукция протестирована и проверенна, а потому является абсолютно безопасной для вас и ваших детей. Ассортимент предлагаемой продукции постоянно меняется и растет. Вы можете приобрести игрушки непосредственно от производителя по приемлемым ценам.
    Работая с нами, вы экономите время и деньги!
    * Цвет или оттенок изделия на фотографии может отличаться от реального. 
    * Характеристики и комплектация товара могут изменятся производителем без уведомления. 
    * Магазин не несет ответственности за изменения внесенные производителем.
    """
            ]

    methods = ['textrank', 'rake', 'rtex']
    df = pd.DataFrame()
    df['desc']=text
    ensure_nltk_data_is_loaded()
    for i in text:
        for method in methods:
            terms = extract_keywords(i, method)
            # print('\n'.join(terms))
