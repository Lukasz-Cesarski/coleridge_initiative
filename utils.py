import json
import re
import os
import pandas as pd
from typing import Optional, Callable
from datetime import datetime


def r_json(file_path):
    assert os.path.isfile(file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def w_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def r_jsonl(file_path):
    assert os.path.isfile(file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def w_jsonl(data: list, file_path):
    dirname = os.path.dirname(file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as file:
        for d in data:
            json.dump(d, file, ensure_ascii=False)
            file.write('\n')


def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()


def jaccard_similarity(s1, s2):
    l1 = s1.split(" ")
    l2 = s2.split(" ")
    intersection = len(list(set(l1).intersection(l2)))
    union = (len(l1) + len(l2)) - intersection
    return float(intersection) / union


# def find_matches_for_instance(instance: pd.Series, clean_function: Optional[Callable] = None):
#     """
#     1. Find corresponding article
#     2. Localize Dataset Name Instances (dataset_label)
#     3. Prepare data for spacy.displacy
#     """
#     article = r_json(id_to_path[instance["Id"]])
#     docs = []
#     pattern = instance["dataset_label"]
#     if clean_function:
#         pattern = clean_function(pattern)
#     for part in article:
#         title = part["section_title"]
#         txt = part["text"]
#         if clean_function:
#             txt = clean_function(txt)
#         matches = list(re.finditer(pattern, txt))
#         if matches:
#             docs.append({
#                 "text": txt,
#                 "ents": [{"start": m.start(), "end": m.end(), "label": LABEL_DT} for m in matches],
#                 "title": title,
#             })
#     return docs
#
#
# def summarize_cleaning_policy(clean_function=None):
#     start = datetime.now()
#     matched_ids = []
#     missing_ids = []
#     for idx in range(len(df_train)):
#         docs = find_matches_for_instance(df_train.iloc[idx], clean_function=clean_function)
#         if docs:
#             matched_ids.append(idx)
#         else:
#             missing_ids.append(idx)
#     print(f"Processing time {datetime.now() - start}")
#     print(f"Matched ids: {len(matched_ids)}")
#     print(f"Missing ids: {len(missing_ids)}")
#
#     ### error analysis
#     if not missing_ids:
#         print("Every dataset instance matched! GREAT!")
#     else:
#         for idx in range(min(len(missing_ids), 5)):
#             instance = df_train.iloc[missing_ids[idx]]
#             pattern = instance["dataset_label"]
#             article = r_json(id_to_path[instance["Id"]])
#             found = []
#             pat_len = len(pattern)
#             stride = 5
#             fuzz_threshold = 90  # range from 0-100, 100 means match
#             for part in article:
#                 txt = part["text"]
#                 for start_idx in range(0, len(txt), stride):
#                     chunk = txt[start_idx:start_idx + pat_len + stride]
#                     if fuzz.partial_ratio(pattern, chunk) > fuzz_threshold:
#                         found.append(chunk)
#             print(f"Pattern = {pattern}")
#             print(f"Instances = {found}")
#             print()
#     return matched_ids, missing_ids



