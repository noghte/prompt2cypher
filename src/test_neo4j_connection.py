import os
import neo4j
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def check_neo4j_connection():
    try:
        with driver.session() as session:
            session.run("MATCH (n) RETURN n LIMIT 1") 
        driver.close()
        return "Neo4j connection successful!"
    except Exception as e:
        return f"Neo4j connection failed. Details: {e}"

if __name__ == "__main__":
    message = check_neo4j_connection()
    print(message)