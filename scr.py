def text_cleaning(text):
    '''
    Converts all text to lower case, Removes special charecters, emojis and multiple spaces
    text - Sentence that needs to be cleaned
    '''
    text = ''.join([k for k in text if k not in string.punctuation])
    text = re.sub('[^A-Za-z0-9]+', ' ', str(text).lower()).strip()
    text = re.sub(' +', ' ', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text


def prepare_text(text, nlp=nlp):
    '''
    Returns the text after stop-word removal and lemmatization.
    text - Sentence to be processed
    nlp - Spacy NLP model
    '''
    doc = nlp(text)
    lemma_list = [token.lemma_ for token in doc if not token.is_stop]
    lemmatized_sentence = ' '.join(lemma_list)

    return lemmatized_sentence


