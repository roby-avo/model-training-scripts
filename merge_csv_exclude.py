import pandas as pd
import os
import zipfile
import argparse
import json
import tempfile
from tqdm import tqdm
from itertools import combinations

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def process_csv_chunk(file, columns_to_remove, chunk_size, progress_bar):
    chunk_iter = pd.read_csv(file, chunksize=chunk_size)
    for chunk in chunk_iter:
        columns_to_remove_existing = [col for col in columns_to_remove if col in chunk.columns]
        chunk.drop(columns=columns_to_remove_existing, inplace=True, errors='ignore')
        yield chunk
        progress_bar.update(len(chunk))

def merge_csv_files_incremental(input_files, output_file, columns_to_remove, ignore_columns, chunk_size=10000):
    processed_chunks = []

    total_rows = sum([sum(1 for row in open(file)) - 1 for file in input_files])
    with tqdm(total=total_rows, desc="Processing files") as pbar:
        for file in input_files:
            for chunk in process_csv_chunk(file, columns_to_remove, chunk_size, pbar):
                processed_chunks.append(chunk)

    merged_df = pd.concat(processed_chunks, ignore_index=True)
    cols_to_consider = [col for col in merged_df.columns if col not in ignore_columns]
    merged_df = merged_df.drop_duplicates(subset=cols_to_consider)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Successfully merged files into {output_file}, removed specified columns, and duplicates.")

def generate_combinations(input_files):
    return [list(comb) for comb in combinations(input_files, len(input_files) - 1)]

def main():
    parser = argparse.ArgumentParser(description='Merge CSV files from a ZIP archive.')
    parser.add_argument('zip_path', type=str, help='Path to the ZIP file containing CSV files.')
    parser.add_argument('output_dir', type=str, help='Directory to save the output CSV files.')
    parser.add_argument('config_file', type=str, help='Path to the JSON config file specifying columns to remove and ignore.')
    parser.add_argument('--chunk_size', type=int, default=10000, help='Number of rows per chunk for incremental processing.')

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    columns_to_remove = config.get('columns_to_remove', [])
    ignore_columns = config.get('ignore_columns', [])

    with tempfile.TemporaryDirectory() as temp_dir:
        extract_zip(args.zip_path, temp_dir)
        input_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.csv')]
        
        for i, excluded_file in enumerate(input_files):
            comb = [file for file in input_files if file != excluded_file]
            excluded_file_name = os.path.basename(excluded_file).replace('.csv', '')
            output_file = os.path.join(args.output_dir, f'merged_excluding_{excluded_file_name}.csv')
            merge_csv_files_incremental(comb, output_file, columns_to_remove, ignore_columns, args.chunk_size)

if __name__ == '__main__':
    main()