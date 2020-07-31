import csv
import re
import stanfordnlp
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
from wordcloud import WordCloud, STOPWORDS  # copy right!!!
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mtick
from adjustText import adjust_text

# the info store in each index
DOCUMENTS_INDEX_0 = 0
PUBLISHER_1 = 1
PUBLISHED_DATE_2 = 2
TOPIC_3 = 3
BYLINE_4 = 4
SECTION_5 = 5
SOURCE_6 = 6
LENGTH_7 = 7
DATELINE_8 = 8
SUMMARY_9 = 9
CONTEXT_10 = 10
LOAD_DATE_11 = 11
LANGUAGE_12 = 12
CORRECTION_DATE_13 = 13
CORRECTION_14 = 14
GRAPHIC_15 = 15
TYPE_16 = 16
DOCUMENT_TYPE_17 = 17
PUBLISH_TYPE_18 = 18
COPYRIGHT_19 = 19
FREQUENCY_DISTRIBUTION = 20

def separate_articles(input_text, publisher_name):
    input_text = re.sub("\n([^\n])", r' \1', input_text)  # Connect the separate line in a paragraph
    input_text = re.sub("\n+", r'\n', input_text)  # Connect each paragraph with one newline between

    input_text_list = input_text.split("\n")

    input_text_list_with_separate_line = []

    line = 0
    while line < len(input_text_list):
        #print(input_text_list[line])
        #publisher_name = "USA TODAY"
        if "documents" in input_text_list[line].lower() and publisher_name in input_text_list[line+1].lower() and line != 0:
            input_text_list_with_separate_line.append('---------------------------'
                    r'-------------------------------'
                    r'-------------------------------'
                    r'-------------------------------')
            input_text_list_with_separate_line.append(input_text_list[line])
            input_text_list_with_separate_line.append(input_text_list[line+1])
            line += 2
        else:
            input_text_list_with_separate_line.append(input_text_list[line])
            line += 1

    # add end of line and one new line so the organize article could read in
    input_text_list_with_separate_line.append('---------------------------'
                                              r'-------------------------------'
                                              r'-------------------------------'
                                              r'-------------------------------')
    input_text_list_with_separate_line.append("")

    input_text = "\n".join(input_text_list_with_separate_line)

    input_text = re.sub("\n\s+", r'\n', input_text)  # Align Text to the Left

    return input_text

def get_month_name(number):
    if number == 0:
        month_name = "January"
    elif number == 1:
        month_name = "February"
    elif number == 2:
        month_name = "March"
    elif number == 3:
        month_name = "April"
    elif number == 4:
        month_name = "May"
    elif number == 5:
        month_name = "June"
    elif number == 6:
        month_name = "July"
    elif number == 7:
        month_name = "August"
    elif number == 8:
        month_name = "September"
    elif number == 9:
        month_name = "October"
    elif number == 10:
        month_name = "November"
    else:
        month_name = "December"
    return month_name

def get_month_index(month_name):
    if month_name == "January":
        month_index = 0
    elif month_name == "February":
        month_index = 1
    elif month_name == "March":
        month_index = 2
    elif month_name == "April":
        month_index = 3
    elif month_name == "May":
        month_index = 4
    elif month_name == "June":
        month_index = 5
    elif month_name == "July":
        month_index = 6
    elif month_name == "August":
        month_index = 7
    elif month_name == "September":
        month_index = 8
    elif month_name == "October":
        month_index = 9
    elif month_name == "November":
        month_index = 10
    else:
        month_index = 11
    return month_index

def combine_two_dictionary(dic_input1, dic_input2):
    dic1 = dic_input1.copy()
    dic2 = dic_input2.copy()
    dic3 = {}
    if dic1 == {}:
        return dic2
    elif dic2 == {}:
        return dic1
    else:
        for key in dic1.keys():
            if key in list(dic2.keys()):
                dic3[key] = dic1[key] + dic2[key]
            else:
                dic3[key] = dic1[key]
        dic2.update(dic3)
        return dic2

def generate_word_cloud(text, output_file_name):
    if text == "":
        return
    nltk_stopwords = stopwords.words("english")
    #  print(nltk_stopwords)
    #  word_cloud_stopwords = set(STOPWORDS)
    #  print(word_cloud_stopwords)

    wc = WordCloud(background_color="white",
                   max_words=200,
                   stopwords=nltk_stopwords,
                   width=4000,  # 4000*2000 would have better quality
                   height=2000,
                   collocations=False)
    wc.generate(text)
    wc.to_file(os.path.join("/Users/jack/Desktop/FD_output", output_file_name))

def organize_data(input_text, publisher_name):
    input_text = separate_articles(input_text, publisher_name)
    input_text = input_text.split("\n---------------------------"
                                  "-------------------------------"
                                  "-------------------------------"
                                  "-------------------------------\n")

    article_amounts = len(input_text) - 1

    for i in range(0, len(input_text)):
        input_text[i] = input_text[i].split('\n')

    input_text[0][0] = re.sub(".+(\d+) of (\d+) DOCUMENTS", r'\1 of \2 DOCUMENTS',
                              input_text[0][0])  # Align the first sentence to the Left

    structuralized_data = []  # all the info in one article

    for article_index in range(0, article_amounts):
        structuralized_data.append([])

        for i in range(0, FREQUENCY_DISTRIBUTION+1):  # from index of DOCUMENTS_INDEX to COPYRIGHT
            structuralized_data[article_index].append("")

        structuralized_data[article_index][DOCUMENTS_INDEX_0] = input_text[article_index][DOCUMENTS_INDEX_0]
        structuralized_data[article_index][PUBLISHER_1] = input_text[article_index][PUBLISHER_1]
        structuralized_data[article_index][TOPIC_3] = input_text[article_index][TOPIC_3]
        structuralized_data[article_index][COPYRIGHT_19] = input_text[article_index][-1]

        # Get rid of Final Edition and spaces behind the date
        match = re.match('(.*)Monday', input_text[article_index][PUBLISHED_DATE_2])
        if match is not None:
            structuralized_data[article_index][PUBLISHED_DATE_2] = match.group(1) + "Monday"
        else:
            match = re.match('(.*)Tuesday', input_text[article_index][PUBLISHED_DATE_2])
            if match is not None:
                structuralized_data[article_index][PUBLISHED_DATE_2] = match.group(1) + "Tuesday"
            else:
                match = re.match('(.*)Wednesday', input_text[article_index][PUBLISHED_DATE_2])
                if match is not None:
                    structuralized_data[article_index][PUBLISHED_DATE_2] = match.group(1) + "Wednesday"
                else:
                    match = re.match('(.*)Thursday', input_text[article_index][PUBLISHED_DATE_2])
                    if match is not None:
                        structuralized_data[article_index][PUBLISHED_DATE_2] = match.group(1) + "Thursday"
                    else:
                        match = re.match('(.*)Friday', input_text[article_index][PUBLISHED_DATE_2])
                        if match is not None:
                            structuralized_data[article_index][PUBLISHED_DATE_2] = match.group(1) + "Friday"
                        else:
                            match = re.match('(.*)Saturday', input_text[article_index][PUBLISHED_DATE_2])
                            if match is not None:
                                structuralized_data[article_index][PUBLISHED_DATE_2] = match.group(1) + "Saturday"
                            else:
                                match = re.match('(.*)Sunday', input_text[article_index][PUBLISHED_DATE_2])
                                if match is not None:
                                    structuralized_data[article_index][PUBLISHED_DATE_2] = match.group(1) + "Sunday"
                                else:
                                    structuralized_data[article_index][PUBLISHED_DATE_2] = input_text[article_index][
                                        PUBLISHED_DATE_2]

        not_match = 0
        for i in range(BYLINE_4, SUMMARY_9 + 3):
            match = re.match('BYLINE:\s*(.*)', input_text[article_index][i])
            if match is not None:
                structuralized_data[article_index][BYLINE_4] = match.group(1)
            else:
                match = re.match('SECTION:\s*(.*)', input_text[article_index][i])
                if match is not None:
                    structuralized_data[article_index][SECTION_5] = match.group(1)
                else:
                    match = re.match('SOURCE:\s*(.*)', input_text[article_index][i])
                    if match is not None:
                        structuralized_data[article_index][SOURCE_6] = match.group(1)
                    else:
                        match = re.match('LENGTH:\s*(.*)', input_text[article_index][i])
                        if match is not None:
                            structuralized_data[article_index][LENGTH_7] = match.group(1)
                        else:
                            match = re.match('DATELINE:\s*(.*)', input_text[article_index][i])
                            if match is not None:
                                structuralized_data[article_index][DATELINE_8] = match.group(1)
                            else:
                                match = re.match('SUMMARY:\s*(.*)', input_text[article_index][i])
                                '''
                                if match is not None:
                                    structuralized_data[article_index][SUMMARY_9] = match.group(1)
                                else:
                                    start_of_article = i
                                    break
                                '''
                                if match is not None:
                                    not_match = 0
                                    structuralized_data[article_index][SUMMARY_9] = match.group(1)
                                else:
                                    not_match += 1
                                    if not_match == 1:
                                        start_of_article = i
                                    if not_match == 3:
                                        break

        not_match = 0  # count if not match twice, break
        end_of_article = None
        for i in range(-2, -(len(input_text[article_index]) - 1), -1):
            match = re.match('LOAD-DATE:\s*(.*)', input_text[article_index][i])
            if match is not None:
                not_match = 0
                structuralized_data[article_index][LOAD_DATE_11] = match.group(1)
            else:
                match = re.match('LANGUAGE:\s*(.*)', input_text[article_index][i])
                if match is not None:
                    not_match = 0
                    structuralized_data[article_index][LANGUAGE_12] = match.group(1)
                else:
                    match = re.match('CORRECTION-DATE:\s*(.*)', input_text[article_index][i])
                    if match is not None:
                        not_match = 0
                        structuralized_data[article_index][CORRECTION_DATE_13] = match.group(1)
                    else:
                        match = re.match('CORRECTION:\s*(.*)', input_text[article_index][i])
                        if match is not None:
                            not_match = 0
                            structuralized_data[article_index][CORRECTION_14] = match.group(1)
                        else:
                            match = re.match('GRAPHIC:\s*(.*)', input_text[article_index][i])
                            if match is not None:
                                not_match = 0
                                line_fragments = ""
                                if end_of_article is not None:
                                    for j in range(i+1, end_of_article+1):
                                        line_fragments = line_fragments + "\n" + input_text[article_index][j]
                                    end_of_article = None
                                structuralized_data[article_index][GRAPHIC_15] = match.group(1) + line_fragments
                            else:
                                match = re.match('TYPE:\s*(.*)', input_text[article_index][i])
                                if match is not None:
                                    not_match = 0
                                    structuralized_data[article_index][TYPE_16] = match.group(1)
                                else:
                                    match = re.match('DOCUMENT-TYPE:\s*(.*)', input_text[article_index][i])
                                    if match is not None:
                                        not_match = 0
                                        structuralized_data[article_index][DOCUMENT_TYPE_17] = match.group(1)
                                    else:
                                        match = re.match('PUBLICATION-TYPE:\s*(.*)', input_text[article_index][i])
                                        if match is not None:
                                            not_match = 0
                                            structuralized_data[article_index][PUBLISH_TYPE_18] = match.group(1)
                                        else:
                                            not_match += 1
                                            if not_match == 1:
                                                end_of_article = i
                                            if not_match == 15:
                                                break

        structuralized_data[article_index][CONTEXT_10] = "\n".join(
            input_text[article_index][start_of_article:len(input_text[article_index]) + end_of_article + 1])

    return structuralized_data

def add_in_frequency_distribution(organized_data, advance_stop_words):
    # Calculate Frequency Distribution
    article_amounts = len(organized_data)
    for article_index in range(0, article_amounts):  # 0 to article_amounts
        print("Title:", organized_data[article_index][DOCUMENTS_INDEX_0])
        if organized_data[article_index][CONTEXT_10] != "":
            stopWords = set(stopwords.words('english'))

            """
            #  old feature
            tokenizer = RegexpTokenizer(r'(\d+/\d+)|(\$*\d*,*\d*\.*\d+)|(\w+)')  # NLTK
            multiple_bag_of_words = tokenizer.tokenize(organized_data[article_index][CONTEXT_10].lower())
        
            for i in range(0, len(multiple_bag_of_words)):
                for j in range(0, len(multiple_bag_of_words[i])):
                    if multiple_bag_of_words[i][j] != '':
                        bag_of_words.append(multiple_bag_of_words[i][j])
            #  print(bag_of_words)        
            #  end of old feature
            """ # Old Feature

            ##############################################################
            # Named Entity Recognition                                   #
            ##############################################################
            words = nltk.word_tokenize(organized_data[article_index][CONTEXT_10])
            #print("Step 1: ", words)
            tagged = nltk.pos_tag(words)

            nameEnt = nltk.ne_chunk(tagged, binary=True)

            # Words not in short phrase will pass basic & advance stopwords filter,
            # tokenization and lemmatization
            # Short phrase will combine with other words after they passed stopwords filter
            # and lemmatized
            bag_of_no_short_phrase_words = []
            bag_of_short_phrase_words = []
            for node in nameEnt:
                temp_word = ""
                if type(node) is nltk.Tree:
                    for i in range(0, len(node)):
                        if temp_word == "":
                            temp_word = node[i][0]
                        else:
                            temp_word = temp_word + "_" + node[i][0]
                    bag_of_short_phrase_words.append(temp_word)
                else:
                    temp_word = node[0]
                    if "-" in temp_word:
                        ## replace "-" with "_" one line a time
                        for replace_time in range(0, 20):
                            temp_word = re.sub("([A-Za-z]+)-([A-Za-z]+)", r'\1_\2', temp_word)
                        ## clear words with only "-"
                        if "-" not in temp_word:
                            bag_of_short_phrase_words.append(temp_word)
                    else:
                        bag_of_no_short_phrase_words.append(temp_word.lower())

            ### End of Named Entity Recognition ###
            #print("Step 2: ", bag_of_short_phrase_words)
            #print("Step 3: ", bag_of_no_short_phrase_words)
            ##################################################
            # Basic Stopwords Filter                         #
            ##################################################
            bag_of_filtered_words = []
            for word in bag_of_no_short_phrase_words:
                if word not in stopWords:
                    if word not in string.punctuation:
                        bag_of_filtered_words.append(word)
            bag_of_filtered_words = " ".join(bag_of_filtered_words)
            #print("Step 4: ", bag_of_filtered_words)
            ### End of Basic Filter ###
            #################################################################
            # Lemmatization for non-phrases words                           #
            #################################################################
            stanford_nlp = stanfordnlp.Pipeline(processors='tokenize,lemma')

            lemmatized_words = []
            bag_of_filtered_words = stanford_nlp(bag_of_filtered_words)
            #print("Step 5:", bag_of_filtered_words)
            for sent in bag_of_filtered_words.sentences:
                for word in sent.words:
                    lemmatized_words.append(word.lemma)
            ### End of Lemmatization ###

            ##################################################
            # Advance Stopwords Filter                       #
            ##################################################
            lemmatized_filtered_words = []
            alphabet_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
                         "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "x"]
            for i in range(0, len(alphabet_list)):
                advance_stop_words.append(alphabet_list[i])

            advance_stop_words.append("``")
            advance_stop_words.append("'")
            advance_stop_words.append("/")
            advance_stop_words.append(".")
            advance_stop_words.append("-")
            advance_stop_words.append("\"\"")
            advance_stop_words.append("''")
            advance_stop_words.append("...")
            advance_stop_words.append("\'s")

            for i in range(0, 100):  # get rid of number 1-100
                advance_stop_words.append("{}".format(i))
            for word in lemmatized_words:
                if word not in advance_stop_words and word is not None:
                    lemmatized_filtered_words.append(word.lower())
            ### End of Advance Stopwords Filter ###

            ###################################################################
            # Combine Filtered Non-phrases Words and Phrases together         #
            ###################################################################
            for i in range(0, len(bag_of_short_phrase_words)):
                lemmatized_filtered_words.append(bag_of_short_phrase_words[i])
            ### End of Combine Filtered Non-phrases Words and Phrases together ###

            ##############################################
            # Calculate Frequency Distribution           #
            ##############################################
            frequency_distribution_dic = {}
            for word in lemmatized_filtered_words:
                if word in frequency_distribution_dic:
                    frequency_distribution_dic[word] += 1
                else:
                    frequency_distribution_dic[word] = 1
            ### Calculate Frequency Distribution ###

            organized_data[article_index].append({})
            organized_data[article_index][FREQUENCY_DISTRIBUTION] = frequency_distribution_dic

    return organized_data

def plot_monthly_FD(organized_data_with_freq, top_Xth, P_NP):
    article_amounts = len(organized_data_with_freq)
    #  Put Each Article in the Right Category, Store In word_frequency_months
    year_name = re.match('.*, (\d*)\s.*', organized_data_with_freq[0][PUBLISHED_DATE_2]).group(1)
    word_frequency_months = []
    for i in range(0, 12):
        word_frequency_months.append({})
    for article_index in range(0, article_amounts):
        match = re.match('(\w*)\s.*', organized_data_with_freq[article_index][PUBLISHED_DATE_2])  # find the same months
        if match is not None:
            month_index = get_month_index(match.group(1))

            article_freq_dic = (organized_data_with_freq[article_index][FREQUENCY_DISTRIBUTION])
            result = []  # print the article freqeuncy distribution
            for w in sorted(article_freq_dic, key=article_freq_dic.get, reverse=True):
                result.append([w, article_freq_dic[w]])

            word_frequency_months[month_index] = combine_two_dictionary(word_frequency_months[month_index],
                                                                        article_freq_dic)  # Update January Word Freqeuncy
            result1 = []  # print the monthly freqeuncy distribution after combining article freqeuncy distribution
            for w in sorted(word_frequency_months[month_index], key=word_frequency_months[month_index].get,
                            reverse=True):
                result1.append([w, word_frequency_months[month_index][w]])
            #print(result)
            #print(result1)
            #print("---------------------------------")
    if P_NP == "P":
        # Plot the word_frequency_months
        plot_x_axis_month = []
        plot_y_axis_month = []
        for month in range(0, len(word_frequency_months)):
            month_name = get_month_name(month)

            for w in sorted(word_frequency_months[month], key=word_frequency_months[month].get, reverse=True):
                if w != '$':
                    plot_x_axis_month.append(w)
                    plot_y_axis_month.append(word_frequency_months[month][w])

            bar_width = 0.2  # inch per bar
            spacing = 3  # spacing between subplots in units of barwidth
            fig_y = (top_Xth * spacing) * bar_width
            fig_x = fig_y * 13 / 10
            plt.figure(figsize=(fig_x, fig_y))
            plt.rcParams['axes.titlesize'] = 20
            plt.rcParams['axes.labelsize'] = 15
            plt.rcParams['xtick.labelsize'] = 16
            plt.rcParams['ytick.labelsize'] = 16

            plt.barh(plot_x_axis_month[0:top_Xth], plot_y_axis_month[0:top_Xth])
            plt.xlabel('Frequency Distribution')
            plt.ylabel('Words')
            plt.title("Top {0}th Words, {1} {2}".format(top_Xth, month_name, year_name))
            file_name_and_dir = "/Users/jack/Desktop/FD_output/Monthly_FD/" + month_name + ".png"
            plt.savefig(file_name_and_dir, dpi=500)
            plt.clf()
            plot_x_axis_month.clear()
            plot_y_axis_month.clear()
    return word_frequency_months

def plot_year_FD(organized_data_with_freq_local, top_Xth, P_NP):
    article_amounts = len(organized_data_with_freq_local)
    #  Put All Articles in the Same Category, Store In word_frequency_year_local
    print(organized_data_with_freq_local[0][PUBLISHED_DATE_2])
    year_name = re.match('.*, (\d*).*', organized_data_with_freq_local[0][PUBLISHED_DATE_2]).group(1)
    word_frequency_year_local = {}
    for article_index in range(0, article_amounts):
        if organized_data_with_freq_local[article_index][FREQUENCY_DISTRIBUTION] != "":
            article_freq_dic = (organized_data_with_freq_local[article_index][FREQUENCY_DISTRIBUTION])
            word_frequency_year_local = combine_two_dictionary(word_frequency_year_local, article_freq_dic)
            result = []  # print the monthly freqeuncy distribution after combining article freqeuncy distribution
            for w in sorted(word_frequency_year_local, key=word_frequency_year_local.get, reverse=True):
                result.append([w, word_frequency_year_local[w]])
    if P_NP == "P":
        # Plot the top_x_year_frequency
        plot_x_axis_year = []
        plot_y_axis_year = []
        for w in sorted(word_frequency_year_local, key=word_frequency_year_local.get, reverse=True):
            if w != '$':
                plot_x_axis_year.append(w)
                plot_y_axis_year.append(word_frequency_year_local[w])
        #print(plot_x_axis_year)

        bar_width = 0.2  # inch per bar
        spacing = 3  # spacing between subplots in units of barwidth
        fig_y = (top_Xth * spacing) * bar_width
        fig_x = fig_y * 13 / 10
        plt.figure(figsize=(fig_x, fig_y))
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['axes.labelsize'] = 15
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.barh(plot_x_axis_year[0:top_Xth], plot_y_axis_year[0:top_Xth])
        plt.xlabel('Frequency Distribution')
        plt.ylabel('Words')
        plt.title("Top {0}th Words, {1}".format(top_Xth, year_name))
        file_name_and_dir = "/Users/jack/Desktop/FD_output/" + year_name + ".png"
        plt.savefig(file_name_and_dir, dpi=400) # less than 500
    return word_frequency_year_local

def plot_word_cloud_month(word_frequency_months, amounts_of_words_on_graph):
    for month in range(0, len(word_frequency_months)):
        text_for_generate_cloud = ""
        top_x_words_list_for_cloud_month = []
        for w in sorted(word_frequency_months[month], key=word_frequency_months[month].get, reverse=True):
            top_x_words_list_for_cloud_month.append([w, word_frequency_months[month][w]])

        for word in range(0, amounts_of_words_on_graph):
            if word >= len(top_x_words_list_for_cloud_month):
                break
            for freq in range(0, top_x_words_list_for_cloud_month[word][1]):
                text_for_generate_cloud = text_for_generate_cloud + " " + top_x_words_list_for_cloud_month[word][0]

        print(month, text_for_generate_cloud)

        output_file_name = get_month_name(month)+".png"
        generate_word_cloud(text_for_generate_cloud, output_file_name)

def plot_word_cloud_year(word_frequency_year, amounts_of_words_on_graph, publisher_dir):
    text_for_generate_cloud = ""
    top_x_words_list_for_cloud_year = []
    for w in sorted(word_frequency_year, key=word_frequency_year.get, reverse=True):
        top_x_words_list_for_cloud_year.append([w, word_frequency_year[w]])

    for word in range(0, amounts_of_words_on_graph):
        if word >= len(top_x_words_list_for_cloud_year):
            break
        for freq in range(0, top_x_words_list_for_cloud_year[word][1]):
            text_for_generate_cloud = text_for_generate_cloud + " " + top_x_words_list_for_cloud_year[word][0]

    print(text_for_generate_cloud)
    #  print(month, text_for_generate_cloud)
    publisher_dir = publisher_dir.replace('.TXT', '.png')
    generate_word_cloud(text_for_generate_cloud, publisher_dir)

def write_csv_file(organized_data_with_freq, publisher_dir):
    publisher_dir = publisher_dir.replace('.TXT', '.csv')
    article_amounts = len(organized_data_with_freq)
    file = open(publisher_dir, "w")
    writer = csv.writer(file)

    writer.writerow(["PUBLISHED_DATE", "DOCUMENTS_INDEX", "PUBLISHER", "TOPIC", "BYLINE", "SECTION", "SOURCE", "LENGTH",
                     "DATELINE", "SUMMARY", "LOAD_DATE", "LANGUAGE", "CORRECTION_DATE", "CORRECTION", "GRAPHIC",
                     "TYPE", "DOCUMENT_TYPE", "PUBLISH_TYPE", "COPYRIGHT"])

    for article_index in range(0, article_amounts):
        #  WRITE IN CSV FILE
        writer.writerow([organized_data_with_freq[article_index][PUBLISHED_DATE_2],
                         organized_data_with_freq[article_index][DOCUMENTS_INDEX_0],
                         organized_data_with_freq[article_index][PUBLISHER_1],
                         organized_data_with_freq[article_index][TOPIC_3],
                         organized_data_with_freq[article_index][BYLINE_4],
                         organized_data_with_freq[article_index][SECTION_5],
                         organized_data_with_freq[article_index][SOURCE_6],
                         organized_data_with_freq[article_index][LENGTH_7],
                         organized_data_with_freq[article_index][DATELINE_8],
                         organized_data_with_freq[article_index][SUMMARY_9],
                         organized_data_with_freq[article_index][LOAD_DATE_11],
                         organized_data_with_freq[article_index][LANGUAGE_12],
                         organized_data_with_freq[article_index][CORRECTION_DATE_13],
                         organized_data_with_freq[article_index][CORRECTION_14],
                         organized_data_with_freq[article_index][GRAPHIC_15],
                         organized_data_with_freq[article_index][TYPE_16],
                         organized_data_with_freq[article_index][DOCUMENT_TYPE_17],
                         organized_data_with_freq[article_index][PUBLISH_TYPE_18],
                         organized_data_with_freq[article_index][COPYRIGHT_19]]) # write row

        top_x_words_list = []
        top_x_frequency_list = []
        top_x = 100  # choose top x to print
        i = 0
        top_x_words_list.append("")  # to empty the of 1 column
        top_x_frequency_list.append("")  # to empty the of 1 column

        if organized_data_with_freq[article_index][FREQUENCY_DISTRIBUTION] != "":
            for w in sorted(organized_data_with_freq[article_index][FREQUENCY_DISTRIBUTION], key=organized_data_with_freq[article_index][FREQUENCY_DISTRIBUTION].get, reverse=True):
                top_x_words_list.append(w)
                top_x_frequency_list.append(organized_data_with_freq[article_index][FREQUENCY_DISTRIBUTION][w])
                # top_x_frequency_list.append([])
                # top_x_frequency_list[i] = [w, frequency_distribution_dic[w]]  # access to key and value of frequency distribution
                if i == top_x:
                    break
                i += 1

        print(top_x_words_list)
        print(top_x_frequency_list)
        writer.writerow(top_x_words_list[0:top_x])
        writer.writerow(top_x_frequency_list[0:top_x])
        #  WRITE IN CSV FILE

def plot_scatter_FD(month1, month2):
    words_x_y = []
    x = []
    y = []

    for i in list(month1.keys()):
        if i in list(month2.keys()):
            words_x_y.append([i, month1[i], month2[i]])
    fig = plt.figure(1, (7,7))
    ax = fig.add_subplot(1,1,1)

    sum_month_1 = 0
    sum_month_2 = 0
    for i in range(0, len(words_x_y)):
        sum_month_1 += words_x_y[i][1]
    for i in range(0, len(words_x_y)):
        sum_month_2 += words_x_y[i][2]

    for i in range(0, len(words_x_y)):
        ax.scatter(math.log(words_x_y[i][1]/sum_month_1*13000)-1.5, math.log(words_x_y[i][2]/sum_month_2*13000)-1.5)
        #  plt.annotate(words_x_y[i][0], (math.log(words_x_y[i][1]/sum_month_1*13000)-1.5, math.log(words_x_y[i][2]/sum_month_2*13000)-1.5))

    texts = []
    for i in range(0, len(words_x_y)):
        annotate_text = words_x_y[i][0]
        x_coordinate = math.log(words_x_y[i][1] / sum_month_1 * 13000) - 1.7
        y_coordinate = math.log(words_x_y[i][2] / sum_month_2 * 13000) - 1.7
        texts.append(plt.text(x_coordinate, y_coordinate, annotate_text))

    print(texts)
    adjust_text(texts)
    plt.xlim(0, 5)
    plt.ylim(0, 5)

    plt.plot([0,10], [0,10], "--")

    plt.xlabel('January')
    plt.ylabel('February')
    plt.title("Frequency Distribution Comparison")
    file_name_and_dir = "/Users/jack/Desktop/FD_output/" + "J_F_Comp" + ".png"
    plt.savefig(file_name_and_dir, dpi=500)

    print("end")

def read_all_file_name(path):
    all_dir_in_data = os.listdir(path)

    all_txtfile_dir = []  # Read all file name and directory
    for publisher_collection_dir in range(1, len(all_dir_in_data)):  # skip DS.Store

        publisher_dir = path + "/" + all_dir_in_data[publisher_collection_dir]

        all_txtfile_dir.append([])
        all_txtfile_dir[publisher_collection_dir - 1].append(all_dir_in_data[publisher_collection_dir])

        for file in os.listdir(publisher_dir):
            if file.endswith(".TXT"):
                text_file_location = publisher_dir + "/" + file
                all_txtfile_dir[publisher_collection_dir - 1].append(file)
    return all_txtfile_dir

## Testing input_text make sure the documents were separated right
def TESTING_organized_data(input_text, organized_data, publisher_name):
    # check if read in all the articles
    articles_amount = len(re.findall("DOCUMENTS", input_text,))
    if len(organized_data) <= articles_amount-5:
        print(publisher_name, "No Pass, less than 50 articles")

    Pass_Check = 1
    for i in range(0, len(organized_data)):
        if "documents" not in str(organized_data[i][DOCUMENTS_INDEX_0]).lower():
            print(publisher_name, "No Pass, documents")
            print(organized_data[i])
            print(organized_data[i][DOCUMENTS_INDEX_0])
            Pass_Check = 0
        if "words" not in str(organized_data[i][LENGTH_7]).lower():
            print(publisher_name, "No Pass, words")
            print(organized_data[i])
            print(organized_data[i][LENGTH_7])
            Pass_Check = 0
        if "english" not in str(organized_data[i][LANGUAGE_12]).lower():
            print(publisher_name, "No Pass, english")
            print(organized_data[i])
            print(organized_data[i][LANGUAGE_12])
            Pass_Check = 0
        if "copyright" not in str(organized_data[i][COPYRIGHT_19]).lower():
            print(publisher_name, "No Pass, copyright")
            print(organized_data[i])
            print(organized_data[i][COPYRIGHT_19])
            Pass_Check = 0

        ###################################################################
        # Check All the Dates, empty allowed except load-date             #
        ###################################################################
        # check LOAD_DATE_11
        month_name_list = ["january", "february", "march", "april", "may", "june", "july", "august",
                           "september", "october", "november", "december"]
        for month_index in range(0, len(month_name_list)):
            if month_name_list[month_index] in organized_data[i][LOAD_DATE_11].lower():
                break
            if month_index == len(month_name_list)-1:
                print(publisher_name, "No Pass, load-date")
                print(organized_data[i])
                print(organized_data[i][LOAD_DATE_11])
                Pass_Check = 0

        # check PUBLISHED_DATE_2
        month_name_list = ["january", "february", "march", "april", "may", "june", "july", "august",
                           "september", "october", "november", "december"]
        for month_index in range(0, len(month_name_list)):
            if organized_data[i][PUBLISHED_DATE_2] == "": # could be empty
                break
            if month_name_list[month_index] in organized_data[i][PUBLISHED_DATE_2].lower():
                break
            if month_index == len(month_name_list)-1:
                print(publisher_name, "No Pass, published-date")
                print(organized_data[i])
                print(organized_data[i][PUBLISHED_DATE_2])
                Pass_Check = 0

        # check CORRECTION_DATE_13
        month_name_list = ["january", "february", "march", "april", "may", "june", "july", "august",
                           "september", "october", "november", "december"]
        for month_index in range(0, len(month_name_list)):
            if organized_data[i][CORRECTION_DATE_13] == "": # could be empty
                break
            if month_name_list[month_index] in organized_data[i][CORRECTION_DATE_13].lower():
                break
            if month_index == len(month_name_list) - 1:
                print(publisher_name, "No Pass, correction date")
                print(organized_data[i])
                print(organized_data[i][CORRECTION_DATE_13])
                Pass_Check = 0

        #if Pass_Check == 1:
        #    print(publisher_name, organized_data[i][DOCUMENTS_INDEX_0], "Pass")

import os

path = "/Users/jack/Desktop/Frequency Distribution Data/Freq_Input"
storing_path = "/Users/jack/Desktop/Frequency Distribution Data/Freq_Output"

# Read All .txt Path in "/Users/jack/Desktop/Data"
all_txtfile_dir = read_all_file_name(path)

for publisher_Name in range(0, len(all_txtfile_dir)):
    # all_txtfile_dir[publisher_Name][0] is publisher dir
    # all_txtfile_dir[publisher_Name][1:] is text file name
    # So start from 1
    for textFile_Name in range(1, len(all_txtfile_dir[publisher_Name])):
        file_name = path + "/" + all_txtfile_dir[publisher_Name][0] + "/" + all_txtfile_dir[publisher_Name][textFile_Name]

        f = open(file_name, 'r')
        input_text = f.read()

        organized_data = organize_data(input_text, all_txtfile_dir[publisher_Name][0].lower())  # send in the publisher name as well

        TESTING_organized_data(input_text, organized_data, all_txtfile_dir[publisher_Name][textFile_Name])

        ## Ploting
        advanced_stop_words_list = ["would", "say", "year", "not", "get", "like", "one",
                                    "day", "also", "could", "month", "even", "day", "yesterday",
                                    "two", "come", "want", "will", "see", "be", "since", "first"]
        print(all_txtfile_dir[publisher_Name][textFile_Name])
        organized_data_with_freq = add_in_frequency_distribution(organized_data, advanced_stop_words_list)



        write_csv_file(organized_data_with_freq,
                       storing_path + "/" + all_txtfile_dir[publisher_Name][0] + "/" + "csv output" + "/" + all_txtfile_dir[publisher_Name][textFile_Name])

        word_frequency_year = plot_year_FD(organized_data_with_freq, 100, "NP")
        plot_word_cloud_year(word_frequency_year, 100,
                        storing_path + "/" + all_txtfile_dir[publisher_Name][0] + "/" + "word cloud output" + "/" + all_txtfile_dir[publisher_Name][textFile_Name])


'''
f = open('USA_Today_2017.TXT', 'r')
input_text = f.read()

organized_data = organize_data(input_text, "Pittsburgh Post-Gazette".lower())

TESTING_organized_data(input_text, organized_data, "None")

advanced_stop_words_list = ["would", "say", "year", "not", "get", "like", "one",
                                    "day", "also", "could", "month", "even", "day", "yesterday",
                                    "two", "come", "want", "will", "see", "be", "since", "first"]

organized_data_with_freq = add_in_frequency_distribution(organized_data, advanced_stop_words_list)
'''