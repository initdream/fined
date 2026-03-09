from pymilvus import MilvusClient

# 1. Connect to the DB
client = MilvusClient("milvus.db")

# 2. Get the ACTUAL list of collections
collections = client.list_collections()
print(f"--> Found these collections: {collections}")

if not collections:
    print("❌ The database is empty. Did the indexing script run successfully?")
else:
    # 3. Pick the first collection found (usually there is only one)
    my_collection = collections[0]
    print(f"--> Querying collection: '{my_collection}'")

    # 4. Run the query using the discovered name
    res = client.query(
        collection_name=my_collection,
        filter="",
        output_fields=["*"],  # Shows all text/metadata
        limit=3
    )

    # 5. Print results nicely
    import json
    for i, entry in enumerate(res):
        print(f"\n--- Entry {i+1} ---")
        # Remove the vector data for cleaner printing (it's too long)
        if 'vector' in entry:
            del entry['vector']
        if 'embedding' in entry:
            del entry['embedding']

        print(json.dumps(entry, indent=4, default=str))
