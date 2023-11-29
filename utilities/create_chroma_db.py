import os
from tqdm import tqdm
import chromadb
from typing import List, Dict


documents_directory: str = "/Users/hanif/Desktop/LLM_Agent/data/palestine_aljazeera"
collection_name: str = "palestine"
persist_directory: str = "/Users/hanif/Desktop/LLM_Agent/data/database/chroma_db"

def create_chroma_collection():
    documents = []
    metadatas = []
    files = os.listdir(documents_directory)
    for filename in files:
        with open(f"{documents_directory}/{filename}", "r") as file:
            for line_number, line in enumerate(
                tqdm((file.readlines()), desc=f"Reading {filename}"), 1
            ):
                # Strip whitespace and append the line to the documents list
                line = line.strip()
                # Skip empty lines
                if len(line) == 0:
                    continue
                documents.append(line)
                metadatas.append({"filename": filename, "line_number": line_number})


    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=collection_name)

    # Create ids from the current count
    count = collection.count()
    print(f"Collection already contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]

    # Load the documents in batches of 100
    for i in tqdm(
        range(0, len(documents), 100), desc="Adding documents", unit_scale=100
    ):
        collection.add(
            ids=ids[i : i + 100],
            documents=documents[i : i + 100],
            metadatas=metadatas[i : i + 100],
        )

    new_count = collection.count()
    print(f"Added {new_count - count} documents")


create_chroma_collection()