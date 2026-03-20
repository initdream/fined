from pymilvus import MilvusClient

client = MilvusClient("milvus.db")

collections = client.list_collections()
print(f"--> Found these collections: {collections}")

if not collections:
    print("The database is empty. Did the indexing script run successfully?")
else:
    my_collection = collections[0]
    print(f"--> Querying collection: '{my_collection}'")

    res = client.query(
        collection_name=my_collection,
        filter="",
        output_fields=["*"],
        limit=3
    )

    import json
    for i, entry in enumerate(res):
        print(f"\n--- Entry {i+1} ---")
        if 'vector' in entry:
            del entry['vector']
        if 'embedding' in entry:
            del entry['embedding']

        print(json.dumps(entry, indent=4, default=str))
