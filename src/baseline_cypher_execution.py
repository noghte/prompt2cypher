import json
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

load_dotenv()
KG_NAME = os.getenv("NEO4J_DATABASE_NAME")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE_NAME = KG_NAME #Assuming the database name is the same as the KG name, otherwise change this line

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
