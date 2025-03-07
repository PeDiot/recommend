from typing import Literal

import argparse, os, json
from pinecone import Pinecone
import src


BATCH_SIZE = 500


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, choices=["item", "query"], required=True)
    return parser.parse_args()


def process_user_dataset(dataset: src.models.UserDataset, mode: Literal["item", "query"]) -> int:
    try:
        if not dataset.is_valid():
            return 0

        if mode == "item":
            embeddings = encoder.encode_images(dataset.images)
        else:
            embeddings = encoder.encode_text(dataset.texts)

        namespace = dataset.user_id
        table_id = f"{src.enums.PROD_DATASET_ID}.{src.enums.USER_VECTOR_TABLE_ID}"
        vectors, rows = src.pinecone.prepare(dataset.point_ids, dataset.payloads, embeddings)
        
        if src.pinecone.upload(pc_index, vectors, namespace):           
            if src.bigquery.upload(bq_client, table_id, rows):
                return len(rows)
            
        return 0
    
    except Exception as e:
        print(e)
        return 0
        

def main():
    args = parse_args()
    secrets = json.loads(os.getenv("SECRETS_JSON"))
        
    global bq_client, encoder, pc_index
    bq_client = src.bigquery.init_client(secrets["GCP_CREDENTIALS"])    
    encoder = src.encoder.FashionCLIPEncoder()
    pc_client = Pinecone(api_key=secrets.get("PINECONE_API_KEY"))
    pc_index = pc_client.Index(src.enums.INDEX_NAME)
    
    loader = src.bigquery.load_items if args.mode == "item" else src.bigquery.load_queries
    user_groups = loader(bq_client, BATCH_SIZE)

    n, n_success, n_inserted = 0, 0, 0
    
    for user_id, group in user_groups:
        if args.mode == "item":
            dataset = src.models.UserDataset.from_image_rows(user_id, group)
        else:
            dataset = src.models.UserDataset.from_text_rows(user_id, group)
            
        if dataset:
            n_inserted_ = process_user_dataset(dataset, args.mode)
            n_success += min(n_inserted_, 1)
            n_inserted += n_inserted_
            n += 1

        print(
            f"Total users: {n} | "
            f"Success rate: {n_success / n} | "
            f"Inserted: {n_inserted}"
        )

if __name__ == "__main__":
    main()