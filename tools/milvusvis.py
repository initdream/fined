from pymilvus import MilvusClient

# 1. Connect
client = MilvusClient("milvus.db")

# 2. Get the collection name automatically
collections = client.list_collections()
if not collections:
    print("❌ No collections found.")
    exit()

col_name = collections[0]
print(f"--> Checking collection: '{col_name}'")

# 3. Query one entry to inspect the embedding
res = client.query(
    collection_name=col_name,
    output_fields=["*"],  # Get all fields to find the vector field
    limit=1
)

if not res:
    print("❌ Collection is empty.")
    exit()

entry = res[0]

# 4. Identify the vector field (Haystack usually names it 'embedding')
vector_field = "embedding" if "embedding" in entry else None

# If not 'embedding', look for any list of floats
if not vector_field:
    for key, value in entry.items():
        if isinstance(value, list) and len(value) > 10 and isinstance(value[0], float):
            vector_field = key
            break

if vector_field:
    vector_data = entry[vector_field]
    dim = len(vector_data)

    print(f"\n✅ Vector Field Found: '{vector_field}'")
    print(f"📏 Dimensions: {dim}")
    print(f"👀 First 5 values: {vector_data[:5]}")

    # Validation for your specific model
    if dim == 384:
        print("\nSUCCESS: Dimension is 384. This matches 'all-MiniLM-L6-v2'.")
    else:
        print(f"\nWARNING: Dimension is {dim}. 'all-MiniLM-L6-v2' usually outputs 384.")
else:
    print("\n❌ Could not find a vector field in the entry.")
    print("Available keys:", entry.keys())
