import csv

import pandas as pd
import json

class Utils:

    doc_features_file_path = "/Users/fsarvi/PycharmProjects/Fair_ranking/data/dummy_data/features.csv"
    libsvm_file_path = "/Users/fsarvi/PycharmProjects/Fair_ranking/AllRank/allrank/arezoo_tests/results/first_exp/reranked_results.txt"

    def __init__(self):
        pass

    def convert_libsvm_format_to_trec_submission(self, libsvm_file_path, doc_id2feature_file_path):
        features = pd.read_csv(doc_id2feature_file_path)

        with open(libsvm_file_path) as f:
            doc_id2features = f.readlines()
        pass

        feature_list = ' '.join([i.split(':')[1] for i in doc_id2features[0].split(',')[2:]])
        features.loc[features['features'] == feature_list]

    def read_text_file_in_lines(self, file_address):
        examples = []
        with open(file_address,
                  errors='ignore') as f:
            examples += [line for line in f]
        return examples

    def write_list_to_text_file(self, data, file_name):
        with open(file_name, 'w') as f:
            for item in data:
                f.write("%s\n" % item)

    def read_jsonl_file(self, file_path):

        result = []
        with open(file_path, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result.append(json.loads(json_str))
        return result

    def save_list_to_csv(self, data, output_file_name):
        if not isinstance(data, list):
            print("input must be a list")
            return
        if not isinstance(data[0], list):
            with open(output_file_name, 'w') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerows(map(lambda x: [x], data))
        else:
            with open(output_file_name, "w") as f:
                writer = csv.writer(f)
                writer.writerows(data)

    def write_dict_to_csv_with_a_row_for_each_key(self, data, output_file_name):
        with open(output_file_name, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in data.items():
                writer.writerow([key, value])

    def read_csv_to_dict_with_first_column_as_keys(self, file_name):
        with open(file_name) as csv_file:
            reader = csv.reader(csv_file)
            mydict = dict(reader)

if __name__ == "__main__":
    pass