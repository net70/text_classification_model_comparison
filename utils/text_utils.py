import pandas as pd
import numpy as np

import gensim
from gensim import corpora

from langdetect import detect, DetectorFactory
from stopwordsiso import stopwords, langs
import stanza
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from lexical_diversity import lex_div as ld
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from collections import defaultdict
from collections import Counter as pCounter
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


def find_n_collocations(df: pd.DataFrame, tokens_col: str, n: int =20, min_frequency: int = 3):
    all_tokens = [token for tokens in df[tokens_col] for token in tokens]
    # Bigram Collocation Finder
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(all_tokens)
    finder.apply_freq_filter(min_frequency)
    bigram_collocations = finder.nbest(bigram_measures.pmi, n)
    
    print(f"\nTop {n} Bigram Collocations:")
    for i, bigram in enumerate(bigram_collocations):
        print(f"{i+1}.{' '.join(bigram)}")

    # Trigram Collocation Finder
    trigram_measures = TrigramAssocMeasures()
    finder = TrigramCollocationFinder.from_words(all_tokens)
    finder.apply_freq_filter(min_frequency)
    trigram_collocations = finder.nbest(trigram_measures.pmi, n)
    
    print(f"\nTop {n} Trigram Collocations:")
    for i, trigram in enumerate(trigram_collocations):
        print(f"{i+1}.{' '.join(trigram)}")


def build_cooccurrence_matrix(tokens_list, window_size=2):
    cooccurrence = defaultdict(pCounter)
    for tokens in tokens_list:
        for i in range(len(tokens)):
            token = tokens[i]
            context = tokens[max(0, i - window_size): i] + tokens[i + 1: i + window_size + 1]
            for context_word in context:
                if token != context_word:
                    cooccurrence[token][context_word] += 1
    return cooccurrence
  

def extract_topics(df, text_column, num_topics=10, passes=5, tokens_already_processed=False, **kwargs):
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


def calculate_cttr(tokens):
    num_tokens = len(tokens)
    num_types = len(set(tokens))
    cttr = num_types / (num_tokens ** 0.5) if num_tokens > 0 else 0
    return cttr


def get_mtld(all_tokens: list):
  mtld = ld.mtld(all_tokens)
  return mtld


def get_stanza_nlp_models(df: pd.DataFrame, text_col: str, language_col: str, model_type: str) -> dict:
  # Filter to supported languages
  if model_type in ['ner']:
    supported_languages = set({'en', 'es', 'fr', 'de', 'it', 'ru', 'zh'})
  else:
    supported_languages = set({'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh'})
    
  df        = df[df[language_col].isin(supported_languages)]
  languages = df[language_col].unique()
  use_gpu   = False if not torch.cuda.is_available() else True
  
  for lang in languages:
    try:
      stanza.download(lang)
      print(f"Downloaded language: {lang}")
    except FileExistsError:
      pass

  # Initialize Stanza pipelines
  stanza_nlp_models = {}
  for lang in languages:
      stanza_nlp_models[lang] = stanza.Pipeline(lang=lang, processors=f'tokenize,{model_type}', use_gpu=use_gpu)

  return stanza_nlp_models


def get_simplified_pos(text: str, language_code: str, pos_nlp_models: dict) -> list:
    nlp = pos_nlp_models.get(language_code)
    if nlp is None:
        return []
    doc = nlp(text)
    simplified_tags = [word.upos for sent in doc.sentences for word in sent.words]
    return simplified_tags


def extract_named_entities(text: str, language_code: str, ner_nlp_models: dict) -> list:
    nlp = ner_nlp_models.get(language_code)
    if nlp is None:
        return []
    doc = nlp(text)
    entities = []
    for sentence in doc.sentences:
        for ent in sentence.ents:
            entities.append({
                'text': ent.text,
                'type': ent.type
            })
    return entities


def get_top_tfidf_words(df: pd.DataFrame, tokenized_col: str, classes_col: str, top_n: int = 20):
    df['processed_text'] = df[tokenized_col].apply(lambda x: ' '.join(x))
  
    # Compute overall TF-IDF
    vectorizer_all = TfidfVectorizer()
    tfidf_matrix_all = vectorizer_all.fit_transform(df['processed_text'])
    feature_names_all = vectorizer_all.get_feature_names_out()
    tfidf_scores_all = tfidf_matrix_all.mean(axis=0).A1
    tfidf_df_all = pd.DataFrame({'term': feature_names_all, 'score_all': tfidf_scores_all})
    
    # Compute TF-IDF per class
    classes = df[classes_col].unique()
    classes = np.append(classes, ['all'])

    top_words_per_class = {}
    
    for cls in classes:
        # Compute TF-IDF for the class
        if cls == 'all':
          class_texts = df['processed_text']
        else:
          class_texts = df[df[classes_col] == cls]['processed_text']
          
        vectorizer_cls = TfidfVectorizer(vocabulary=feature_names_all)
        tfidf_matrix_cls = vectorizer_cls.fit_transform(class_texts)
        tfidf_scores_cls = tfidf_matrix_cls.mean(axis=0).A1
        tfidf_df_cls = pd.DataFrame({'term': feature_names_all, 'score': tfidf_scores_cls})
        
        # Merge with overall TF-IDF
        tfidf_merged = tfidf_df_cls.merge(tfidf_df_all, on='term')
        # Calculate difference
        tfidf_merged['score_diff'] = tfidf_merged['score'] - tfidf_merged['score_all']
        # Sort by score difference
        tfidf_merged = tfidf_merged.sort_values(by='score_diff', ascending=False)
        # Extract top words
        top_words = tfidf_merged.head(top_n)
        top_words_per_class[cls] = top_words
      
    return top_words_per_class

def compute_avg_score(tokens: list, term_score_dict: dict) -> float:
    if not tokens or not isinstance(tokens, list):
        return np.nan  # Return NaN if tokens list is empty or invalid
    scores = [term_score_dict[token] for token in tokens if token in term_score_dict]
    if scores:
        return np.mean(scores)
    else:
        return np.nan


def sentiment_score_from_probs(probs):
    negative_prob = probs[0]
    neutral_prob = probs[1]
    positive_prob = probs[2]
    sentiment_score = (-1 * negative_prob) + (0 * neutral_prob) + (1 * positive_prob)
    return sentiment_score


def get_sentiment_scores_batch(texts: list, model_name: str, device: torch.device, batch_size: int = 16):
    results = []
    total_texts = len(texts)
  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)

    model.to(device)
    model.eval()

    for i in range(0, total_texts, batch_size):
        batch_texts = texts[i:i+batch_size]
        # Preprocess texts
        batch_texts = [text.strip() if isinstance(text, str) else '' for text in batch_texts]
        # Tokenize and encode
        encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        # Move inputs to GPU
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        # Get model outputs
        with torch.no_grad():
            outputs = model(**encoded_input)
        # Get logits and compute probabilities
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        probs = probs.detach().cpu().numpy()
        # Compute sentiment scores and polarities
        for prob in probs:
            sentiment_score = sentiment_score_from_probs(prob)
            if sentiment_score < -0.33:
                polarity = 'negative'
            elif sentiment_score > 0.33:
                polarity = 'positive'
            else:
                polarity = 'neutral'
            results.append({'sentiment_score': sentiment_score, 'polarity': polarity})
    return results