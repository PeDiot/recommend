import os, json
from pinecone import Pinecone
import src


BATCH_SIZE = None


def process_user_dataset(dataset: src.models.UserDataset) -> int:
    try:
        if not dataset.is_valid():
            return 0

        embeddings = encoder.encode_images(dataset.images)
        namespace = dataset.user_id

        vectors, bq_rows, supabase_rows = src.pinecone.prepare(
            dataset.point_ids, dataset.metadata_list, embeddings
        )

        if not src.pinecone.upload(
            index=pc_index, vectors=vectors, namespace=namespace
        ):
            return 0

        if not src.bigquery.upload(
            client=bq_client,
            dataset_id=src.enums.PROD_DATASET_ID,
            table_id=src.enums.USER_VECTOR_TABLE_ID,
            rows=bq_rows,
        ):
            return 0

        if not src.supabase.upload(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            table_id=src.enums.USER_VECTOR_TABLE_ID,
            rows=supabase_rows,
        ):
            return 0

        return len(vectors)

    except Exception as e:
        print(e)
        return 0


def main():
    secrets = json.loads(os.getenv("SECRETS_JSON"))

    global supabase_url, supabase_key
    supabase_url = secrets["SUPABASE_URL"]
    supabase_key = secrets["SUPABASE_SERVICE_ROLE_KEY"]

    global bq_client, encoder, pc_index
    bq_client = src.bigquery.init_client(secrets["GCP_CREDENTIALS"])
    encoder = src.encoder.FashionCLIPEncoder()
    pc_client = Pinecone(api_key=secrets.get("PINECONE_API_KEY"))
    pc_index = pc_client.Index(src.enums.INDEX_NAME)

    loader_kwargs = {"client": bq_client, "n": BATCH_SIZE}
    loader = src.bigquery.load_items(**loader_kwargs)

    n, n_success, n_inserted = 0, 0, 0

    for user_id, group in loader:
        dataset = src.models.UserDataset.from_image_rows(user_id, group)

        if dataset:
            n_inserted_ = process_user_dataset(dataset)
            n_success += min(n_inserted_, 1)
            n_inserted += n_inserted_
            n += 1

        success_rate = n_success / n if n > 0 else 0

        print(
            f"User: {user_id} | "
            f"Inserted: {n_inserted_} | "
            f"Total users: {n} | "
            f"Total Inserted: {n_inserted} | "
            f"Success rate: {success_rate:.2f}"
        )


if __name__ == "__main__":
    main()
