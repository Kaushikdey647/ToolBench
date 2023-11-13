import json
import argparse
import os
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="", required=True, help='The directory to output the split files')
    parser.add_argument('--query_file', type=str, default="", required=True, help='The name of the query file')
    parser.add_argument('--index_file', type=str, default="", required=True, help='The name of the index file')
    parser.add_argument('--dataset_name', type=str, default="", required=True, help='The name of the output dataset')
    return parser.parse_args()


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def write_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)


def split_dataset(query_data, test_index_data):
    test_index_set = set(map(int, test_index_data.keys()))
    query_train = []
    query_test = []

    for index, item in tqdm(enumerate(query_data)):
        if "query_id" in item:
            index = item["query_id"]
        if index in test_index_set:
            query_test.append(item)
        else:
            query_train.append(item)

    return query_train, query_test


def process_data(data, doc_id_map, query_id_map):
    documents = []
    pairs = []

    for doc in tqdm(data):
        for api in doc['api_list']:
            document_content = api
            api_identity = [api['tool_name'], api['api_name']]
            doc_id = doc_id_map.setdefault(json.dumps(document_content), len(doc_id_map) + 1)
            documents.append([doc_id, json.dumps(document_content)])

            if api_identity in doc['relevant APIs']:
                query = doc['query']
                if isinstance(query, list):
                    query = query[0]
                query_id = query_id_map.setdefault(query, len(query_id_map) + 1)
                pairs.append(([query_id, query], [query_id, 0, doc_id, 1]))

    return documents, pairs


def shuffle_and_split_pairs(pairs):
    shuffled_pairs = shuffle(pairs, random_state=42)
    queries, labels = zip(*shuffled_pairs)
    return queries, labels

def create_n_save(data,columns,output_dir,save_name, header=False):
    temp_df = pd.DataFrame(data, columns=columns)
    os.makedirs(output_dir, exist_ok=True)
    temp_df.to_csv(output_dir + save_name, sep='\t', index=False, header=header)

def main():

    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    query_data = read_json_file(args.query_file)
    test_index_data = read_json_file(args.index_file)

    query_train, query_test = split_dataset(query_data, test_index_data)

    write_json_file(args.output_dir + '/train.json', query_train)
    write_json_file(args.output_dir + '/test.json', query_test)

    doc_id_map = {}
    query_id_map = {}

    documents_train, train_pairs = process_data(query_train, doc_id_map, query_id_map)
    documents_test, test_pairs = process_data(query_test, doc_id_map, query_id_map)

    train_queries, train_labels = shuffle_and_split_pairs(train_pairs)
    test_queries, test_labels = shuffle_and_split_pairs(test_pairs)

    columns_documents = ['docid', 'document_content']
    columns_queries = ['qid', 'query_text']
    columns_labels = ['qid', 'useless', 'docid', 'label']

    create_n_save(documents_train, columns_documents, args.output_dir, '/corpus.tsv', False)
    create_n_save(train_queries, columns_queries, args.output_dir, '/train.query.txt')
    create_n_save(test_queries, columns_queries, args.output_dir, '/test.query.txt')
    create_n_save(train_labels, columns_labels, args.output_dir, '/qrels.train.tsv')
    create_n_save(test_labels, columns_labels, args.output_dir, '/qrels.test.tsv')


if __name__ == "__main__":
    main()