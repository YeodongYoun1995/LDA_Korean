# dataloader.py
import os
import json

# 맥의 경우 한글 조합자의 차이로 인해 정상적으로 읽어들이지 않는 문제 해결.
CATEGORY_MAP = {
    '뉴스': '뉴스',
    '위키': '위키'
}

def load_sentences_with_labels(root_path, category_keyword):
    source_folder = '01.원천데이터'
    all_sentences = []
    all_labels = []

    if category_keyword not in CATEGORY_MAP:
        raise ValueError(f"category_keyword는 '뉴스' 또는 '위키' 중 하나여야 함.")

    file_keyword = CATEGORY_MAP[category_keyword]
    data_root = os.path.join(root_path, source_folder)

    for topic_folder in sorted(os.listdir(data_root)):
        topic_path = os.path.join(data_root, topic_folder)
        if not os.path.isdir(topic_path):
            continue

        topic_label = topic_folder.split('.')[-1]

        for file_name in os.listdir(topic_path):
            if file_keyword in file_name and file_name.endswith(".json"):
                full_path = os.path.join(topic_path, file_name)
                with open(full_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        for entry in data:
                            sentence = entry.get("sentence", "")
                            if sentence:
                                all_sentences.append(sentence)
                                all_labels.append(topic_label)
                    except json.JSONDecodeError:
                        pass

    return all_sentences, all_labels

def load_sentences_by_split(root_path, category_keyword):
    train_docs, train_labels = load_sentences_with_labels(os.path.join(root_path, 'Training'), category_keyword)
    val_docs, val_labels = load_sentences_with_labels(os.path.join(root_path, 'Validation'), category_keyword)
    return train_docs, train_labels, val_docs, val_labels