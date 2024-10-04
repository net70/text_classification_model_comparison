import pandas as pd
import numpy as np

import gensim
from gensim import corpora

from langdetect import detect, DetectorFactory
from stopwordsiso import stopwords, langs
import nltk
from nltk.tokenize import word_tokenize
# Download NLTK data files (only the first time)
# nltk.download('punkt')

import re

# Complete mapping of ISO language codes to NLTK stopwords language names
lang_code_to_nltk = {
    'ar': 'arabic',
    'az': 'azerbaijani',
    'bn': 'bengali',
    'ca': 'catalan',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'fi': 'finnish',
    'fr': 'french',
    'de': 'german',
    'el': 'greek',
    'he': 'hebrew',
    'hu': 'hungarian',
    'id': 'indonesian',
    'it': 'italian',
    'kk': 'kazakh',
    'ne': 'nepali',
    'no': 'norwegian',
    'pt': 'portuguese',
    'ro': 'romanian',
    'ru': 'russian',
    'sl': 'slovene',
    'es': 'spanish',
    'sv': 'swedish',
    'tg': 'tajik',
    'tr': 'turkish',
}

iso_639_1_code_language_map = {
    'xxx': 'unknown',
    'ab': 'Abkhazian',
    'af': 'Afrikaans',
    'ak': 'Akan',
    'sq': 'Albanian',
    'am': 'Amharic',
    'ar': 'Arabic',
    'an': 'Aragonese',
    'hy': 'Armenian',
    'as': 'Assamese',
    'av': 'Avaric',
    'ae': 'Avestan',
    'ay': 'Aymara',
    'az': 'Azerbaijani',
    'ba': 'Bashkir',
    'bm': 'Bambara',
    'eu': 'Basque',
    'be': 'Belarusian',
    'bn': 'Bengali',
    'bh': 'Bihari languages',
    'bi': 'Bislama',
    'bo': 'Tibetan',
    'bs': 'Bosnian',
    'br': 'Breton',
    'bg': 'Bulgarian',
    'my': 'Burmese',
    'ca': 'Catalan; Valencian',
    'cs': 'Czech',
    'ch': 'Chamorro',
    'ce': 'Chechen',
    'zh': 'Chinese',
    'cu': 'Church Slavic',
    'cv': 'Chuvash',
    'kw': 'Cornish',
    'co': 'Corsican',
    'cr': 'Cree',
    'cy': 'Welsh',
    'da': 'Danish',
    'de': 'German',
    'dv': 'Divehi',
    'nl': 'Dutch',
    'dz': 'Dzongkha',
    'el': 'Greek',
    'en': 'English',
    'eo': 'Esperanto',
    'et': 'Estonian',
    'ee': 'Ewe',
    'fo': 'Faroese',
    'fa': 'Persian',
    'fj': 'Fijian',
    'fi': 'Finnish',
    'fr': 'French',
    'fy': 'Western Frisian',
    'ff': 'Fulah',
    'ka': 'Georgian',
    'gd': 'Gaelic',
    'ga': 'Irish',
    'gl': 'Galician',
    'gv': 'Manx',
    'gn': 'Guarani',
    'gu': 'Gujarati',
    'ht': 'Haitian',
    'ha': 'Hausa',
    'he': 'Hebrew',
    'hz': 'Herero',
    'hi': 'Hindi',
    'ho': 'Hiri Motu',
    'hr': 'Croatian',
    'hu': 'Hungarian',
    'ig': 'Igbo',
    'is': 'Icelandic',
    'io': 'Ido',
    'ii': 'Sichuan Yi',
    'iu': 'Inuktitut',
    'ie': 'Interlingue',
    'ia': 'Interlingua',
    'id': 'Indonesian',
    'ik': 'Inupiaq',
    'it': 'Italian',
    'jv': 'Javanese',
    'ja': 'Japanese',
    'kl': 'Kalaallisut',
    'kn': 'Kannada',
    'ks': 'Kashmiri',
    'kr': 'Kanuri',
    'kk': 'Kazakh',
    'km': 'Central Khmer',
    'ki': 'Kikuyu',
    'rw': 'Kinyarwanda',
    'ky': 'Kirghiz',
    'kv': 'Komi',
    'kg': 'Kongo',
    'ko': 'Korean',
    'kj': 'Kuanyama',
    'ku': 'Kurdish',
    'lo': 'Lao',
    'la': 'Latin',
    'lv': 'Latvian',
    'li': 'Limburgan',
    'ln': 'Lingala',
    'lt': 'Lithuanian',
    'lb': 'Luxembourgish',
    'lu': 'Luba-Katanga',
    'lg': 'Ganda',
    'mk': 'Macedonian',
    'mh': 'Marshallese',
    'ml': 'Malayalam',
    'mi': 'Maori',
    'mr': 'Marathi',
    'ms': 'Malay',
    'mg': 'Malagasy',
    'mt': 'Maltese',
    'mn': 'Mongolian',
    'na': 'Nauru',
    'nv': 'Navajo',
    'nr': 'Ndebele',
    'nd': 'Ndebele',
    'ng': 'Ndonga',
    'ne': 'Nepali',
    'nn': 'Norwegian Nynorsk',
    'nb': 'Bokmål',
    'no': 'Norwegian',
    'ny': 'Chichewa',
    'oc': 'Occitan',
    'oj': 'Ojibwa',
    'or': 'Oriya',
    'om': 'Oromo',
    'os': 'Ossetian',
    'pa': 'Panjabi',
    'pi': 'Pali',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ps': 'Pushto',
    'qu': 'Quechua',
    'rm': 'Romansh',
    'ro': 'Romanian',
    'rn': 'Rundi',
    'ru': 'Russian',
    'sg': 'Sango',
    'sa': 'Sanskrit',
    'si': 'Sinhala',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'se': 'Northern Sami',
    'sm': 'Samoan',
    'sn': 'Shona',
    'sd': 'Sindhi',
    'so': 'Somali',
    'st': 'Sotho',
    'es': 'Spanish',
    'sc': 'Sardinian',
    'sr': 'Serbian',
    'ss': 'Swati',
    'su': 'Sundanese',
    'sw': 'Swahili',
    'sv': 'Swedish',
    'ty': 'Tahitian',
    'ta': 'Tamil',
    'tt': 'Tatar',
    'te': 'Telugu',
    'tg': 'Tajik',
    'tl': 'Tagalog',
    'th': 'Thai',
    'ti': 'Tigrinya',
    'to': 'Tonga',
    'tn': 'Tswana',
    'ts': 'Tsonga',
    'tk': 'Turkmen',
    'tr': 'Turkish',
    'tw': 'Twi',
    'ug': 'Uighur',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'uz': 'Uzbek',
    've': 'Venda',
    'vi': 'Vietnamese',
    'vo': 'Volapük',
    'wa': 'Walloon',
    'wo': 'Wolof',
    'xh': 'Xhosa',
    'yi': 'Yiddish',
    'yo': 'Yoruba',
    'za': 'Zhuang; Chuang',
    'zu': 'Zulu',
}

# Ensure reproducibility in language detection
DetectorFactory.seed = 0

# List of languages supported by stopwordsiso
supported_languages = langs()


################################ FUNCTIONS ################################
def remove_pii(text: str) -> str:
    original_input = text
    try:
        # Regular expression pattern for phone numbers
        phone_pattern = r"\b(\+?\d[-.\s()]*\d{3}[-.\s()]*(?:\d{2,}[-.\s()]*)?\d{2,})\b"
    
        # Regular expression pattern for email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    
        # Regular expression pattern for addresses
        address_pattern = r"\b\d+\s+([A-Za-z]+\s*){1,4}(Street|St|Rd|Road|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Square|Sq)\b"
    
        # Function to replace phone numbers with "<someones_phone_number>"
        def replace_phone(match):
            return "<someones_phone_number>"
    
        # Function to replace email addresses with the appropriate dummy values
        def replace_email(match):
            email = match.group()
            if email.endswith("lusha.com"):
                return "rep@lusha.com"
            else:
                return "prospect@somemail.com"
    
        # Function to replace addresses with "<some_address>"
        def replace_address(match):
            return "<some_address>"
        
        # Function to remove url patterns (i.e <https://blah-blah.com/blah-blah>)
        def remove_urls(text):
            # Remove URLs starting with "http://" or "https://"
            text = re.sub(r"http[s]?://\S+", "<some_url>", text)
    
            # Remove URLs enclosed in angle brackets <>
            text = re.sub(r"<[^>]+>", "<some_url>", text)
    
            # Remove additional URL pattern
            text = re.sub(r"\.s\.hubspotemail\.net/[^ ]+", "<some_url>", text)
            return text
        
        def remove_addresses(text):
            # Define the address patterns
            patterns = [
                r'\b\d+\b\s+\w+\b\s+\w+\b',                               # Numeric Street Address
                r'\b\d+\w*\b[-/]?\s*\b\d+\w*\b\s+\w+\b\s+\w+\b',          # Street Address with Unit/Apartment Number
                r'(P\.O\. Box|PO Box)\s+\d+',                             # P.O. Box Address
                r'\b\w+\b\s*,?\s*\b\w{2}\b\s+\d{5}(-\d{4})?',             # City, State, and ZIP/Postal Code
            ]
    
            # Remove address patterns from the text
            for pattern in patterns:
                text = re.sub(pattern, '<some_address>', text)
    
            return text
    
        # Replace phone numbers with "<someones_phone_number>"
        text = re.sub(phone_pattern, replace_phone, text)
    
        # Replace email addresses with dummy values
        text = re.sub(email_pattern, replace_email, text)
    
        # Replace addresses with "<some_address>"
        text = re.sub(address_pattern, replace_address, text)
        
        text = remove_urls(text)
        
        text = remove_addresses(text)
    
        res = text
    except Exception as e:
        print(f"Error:{str(e)}|\t text: {original_input}")
        res = original_input
    finally:
        return res


# Function to detect language and add as a new column
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'xxx'

# Function to preprocess text with language-specific stopwords
def tokenize_text(text, lang):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    #nltk_lang = lang_code_to_nltk.get(lang)
    if lang in supported_languages:
        stop_words = set(stopwords(lang))
        tokens = [word for word in tokens if word not in stop_words]
    return tokens


def extract_topics(df, text_column, num_topics=10, passes=5, tokens_already_processed=False, **kwargs):
    """
    Extract topics from a dataset using LDA topic modeling.

    Parameters:
    - df: pandas DataFrame containing the dataset.
    - text_column: string, name of the column containing the text data.
    - num_topics: int, number of topics to extract (default is 10).
    - passes: int, number of passes through the corpus during training (default is 5).
    - tokens_already_processed: bool, 
        - True if text_column contains tokenized words as lists of strings.
        - False if text_column contains raw text strings that need tokenization.
    - **kwargs: additional keyword arguments to pass to gensim.models.LdaModel.

    Returns:
    - lda_model: trained LDA model.
    - corpus: corpus used for the model.
    - dictionary: gensim dictionary mapping of id to word.
    """
    # If tokens are not already processed, tokenize the text data
    if not tokens_already_processed:
        df['processed_tokens'] = df[text_column].apply(lambda x: word_tokenize(x))
        text_data = df['processed_tokens']
    else:
        text_data = df[text_column]

    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(text_data)

    # Filter out extremes to remove very common and very rare words
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    # Create a Bag-of-Words representation of the documents
    corpus = [dictionary.doc2bow(text) for text in text_data]

    # Train the LDA model
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=num_topics,
                                       passes=passes,
                                       random_state=42,
                                       **kwargs)

    return lda_model, corpus, dictionary


def assign_dominant_topic(lda_model, corpus):
    """
    Assigns the dominant topic to each document in the corpus.
    
    Returns:
    - A list of dominant topic indices for each document.
    - A list of topic contribution percentages for each document.
    """
    dominant_topics = []
    topic_percentages = []
    
    for doc_bow in corpus:
        # Get the topic distribution for the document
        topic_probs = lda_model.get_document_topics(doc_bow, minimum_probability=0.0)
        # Sort the topics by probability
        sorted_topic_probs = sorted(topic_probs, key=lambda x: x[1], reverse=True)
        # Get the dominant topic and its percentage
        dominant_topic, dominant_prob = sorted_topic_probs[0]
        dominant_topics.append(dominant_topic)
        topic_percentages.append(dominant_prob)
    
    return dominant_topics, topic_percentages