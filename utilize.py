import re


def normalize(text):
    text = text.lower()
    text = re.sub(r'[_\-]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点
    text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
    return text