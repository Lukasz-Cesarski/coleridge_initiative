import re
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
from utils import r_json, clean_text, w_jsonl
from datetime import datetime




def preprocess_data(data_dir):
    # DIRECTORY TREE
    LABEL_DT = "DT"  # dataset_title

    train_files = glob.glob(join(data_dir, "train/*.json"))
    test_files = glob.glob(join(data_dir, "test/*.json"))
    train_file = join(data_dir, "train.csv")
    id_to_path = {
        os.path.split(path)[-1][:-5]: path for path in train_files
    }
    assert len(id_to_path) == len(train_files)
    assert os.path.isfile(train_file)
    print("Train Atricles found", len(train_files))
    print("Test Articles found", len(test_files))
    df_train = pd.read_csv(train_file)
    start = datetime.now()
    articles = []
    # for path in tqdm(train_files[:100]):
    for path in tqdm(train_files):
        ### ARTICLES ###
        article = {}
        article["path"] = path
        paper_idx = os.path.split(path)[-1][:-5]
        article["idx"] = paper_idx
        paper = r_json(path)
        article["paper"] = paper
        clean_paper = [
            {
                "section_title": part["section_title"],
                "text": clean_text(part["text"]),
                "matches": [],
            } for part in paper
        ]
        article["paper_clean"] = clean_paper

        ### WORDMAP ###
        for raw_part, clean_part in zip(paper, clean_paper):
            raw_txt = raw_part["text"]
            clean_txt = clean_part["text"]
            raw_tokens = raw_txt.split()
            clean_tokens = [clean_text(t) for t in raw_tokens]
            wordmap = []
            filter_tokens = []
            for idx, t in enumerate(clean_tokens):
                if not re.fullmatch(r"^\s*$", t):
                    filter_tokens.append(t)
                    wordmap.extend([idx for _ in range(len(t.split(" ")))])
            # checks
            processed_txt = " ".join(filter_tokens)
            assert clean_txt == processed_txt
            clean_part["wordmap"] = wordmap

        ### INSTANCES ###
        instances_df = df_train.query("Id == @paper_idx")
        instances = [instance_row.to_dict() for _, instance_row in instances_df.iterrows()]
        article["instances"] = instances

        ### MATCHES ###
        for instance in instances:
            pattern = instance["dataset_label"]
            pattern = r"\b" + clean_text(pattern) + r"\b"
            for part in clean_paper:
                txt = part["text"]
                matches = list(re.finditer(pattern, txt))
                if matches:
                    part["matches"].extend(
                        [{"start": m.start(), "end": m.end(), "label": LABEL_DT} for m in matches])

        ### ALIGNMENT ###
        for raw_part, clean_part in zip(paper, clean_paper):
            clean_matches = clean_part["matches"]
            raw_matches = []
            clean_txt = clean_part["text"]
            wordmap = clean_part["wordmap"]
            raw_txt = raw_part["text"]
            clean_cumsum = np.cumsum([0] + [len(t) + 1 for t in clean_txt.split()])
            list_clean_cumsum = clean_cumsum.tolist()
            raw_cumsum = np.cumsum([0] + [len(t) + 1 for t in raw_txt.split()])
            list_raw_cumsum = raw_cumsum.tolist()

            if clean_matches:
                for clean_match in clean_matches:
                    clean_start_char = clean_match["start"]
                    clean_end_char = clean_match["end"]
                    clean_match_txt = clean_txt[clean_start_char:clean_end_char]
                    clean_start_word = list_clean_cumsum.index(clean_start_char)
                    clean_end_word = list_clean_cumsum.index(clean_end_char + 1)
                    clean_match["start_word"] = clean_start_word
                    clean_match["end_word"] = clean_end_word
                    assert clean_match_txt == " ".join(clean_txt.split()[clean_start_word:clean_end_word])
                    raw_start_word = wordmap[clean_start_word]
                    raw_end_word = wordmap[clean_end_word]
                    if raw_start_word == raw_end_word:
                        raw_end_word += 1
                    raw_start_char = list_raw_cumsum[raw_start_word]
                    # print(type(list_raw_cumsum))
                    raw_end_char = list_raw_cumsum[raw_end_word]
                    raw_match = {
                        "start": raw_start_char,
                        "end": raw_end_char,
                        "start_word": raw_start_word,
                        "end_word": raw_end_word,
                        "label": LABEL_DT,
                    }
                    raw_matches.append(raw_match)
            raw_part["matches"] = raw_matches

        articles.append(article)
    print(f"Processing time {datetime.now() - start}")

    return articles


if __name__ == "__main__":
    DATA_DIR = "data"
    PROCESSED_PATH = "output/preprocessed_data.jsonl"
    if not os.path.isfile(PROCESSED_PATH):
        articles = preprocess_data(DATA_DIR)
        w_jsonl(articles, PROCESSED_PATH)
    else:
        size = os.path.getsize(PROCESSED_PATH)/1000**2
        print(f"Data found. Size: {size:.3f} MB")