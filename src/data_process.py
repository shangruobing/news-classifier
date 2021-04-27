import string
from nltk.corpus import stopwords


def punctuation_removal(text):
    """删除标点符号"""
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str


def format_data(text):
    """设置数据格式：删除标点、文本小写、删除stopwords"""
    text = text.apply(punctuation_removal)
    text = text.apply(lambda x: x.lower())
    stop = stopwords.words('english')
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
    return text
