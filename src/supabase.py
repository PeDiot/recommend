from typing import List, Dict
from supabase import create_client, Client


def init_supabase_client(url: str, key: str) -> Client:
    return create_client(supabase_url=url, supabase_key=key)


def upload(
    supabase_url: str, supabase_key: str, table_id: str, rows: List[Dict]
) -> bool:
    supabase_client = init_supabase_client(supabase_url, supabase_key)

    try:
        response = supabase_client.table(table_id).upsert(rows).execute()
        return len(response.data) == len(rows)

    except Exception as e:
        print(e)
        return False
