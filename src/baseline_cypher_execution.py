import json
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

load_dotenv()
NEO4J_DATABASE_NAME = os.getenv("NEO4J_DATABASE_NAME")
KG_NAME = None # the folder name in kgmetadata and results should match this name
if NEO4J_DATABASE_NAME == "neo4j":
    KG_NAME = "ickg"
elif NEO4J_DATABASE_NAME == "prokino-kg":
    KG_NAME = "prokino"
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


input_file_path = f'./data/{KG_NAME}/test_queries.json'
output_file_path = input_file_path.replace('.json', '-with_results.json')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def clean_value(value):
    # Replacing non-breaking spaces with regular spaces and strips leading/trailing whitespace.
    return value.replace('\u00a0', ' ').strip() if isinstance(value, str) else value

def run_cypher_query(query):
    try:
        with driver.session(database=NEO4J_DATABASE_NAME) as session:
            if query:
                results = session.run(query)
                result_list = [[clean_value(value) for value in record.values()] for record in results]
                return result_list
            else:
                return []
    except Neo4jError as e:
        print(f"Neo4j query failed: {e}")
        return []

try:
    with open(input_file_path, 'r') as file:
        data = json.load(file)
except json.JSONDecodeError as e:
    print(f"Failed to decode JSON file: {e}")
    exit(1)

for item in data:
    query = item["cypher"]
    cypher_result = run_cypher_query(query)
    item["cypher_result"] = cypher_result

with open(output_file_path, 'w') as file:
    json.dump(data, file, indent=4)

driver.close()

print(f"Updated JSON file saved as {output_file_path}")
