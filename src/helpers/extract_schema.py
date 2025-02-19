import os
import json
import neo4j
from dotenv import load_dotenv

# Load environment variables
load_dotenv("./backend-flask/.env")

# Database configuration
PROKINO_DATABASE_NAME = "prokino-new"
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def extract_schema(session):
    schema = {"labels": {}, "relationships": []}

    # Node labels extraction remains the same
    label_query = """
    CALL db.schema.nodeTypeProperties()
    YIELD nodeType, propertyName, propertyTypes
    RETURN nodeType, propertyName, propertyTypes
    """
    for record in session.run(label_query):
        labels_combined = record["nodeType"].lstrip(':').split(':`')
        labels = [label.strip('`') for label in labels_combined if label.rstrip('`') != "Resource" and not label.startswith('Resource')]
        if not labels:
            continue
        label_key = ':'.join(labels)
        property_name = record["propertyName"]
        property_types = record["propertyTypes"][0]
        if label_key not in schema["labels"]:
            schema["labels"][label_key] = {}
        schema["labels"][label_key][property_name] = property_types

    # Modified relationship extraction
    relationship_query = """
    CALL db.schema.relTypeProperties()
    YIELD relType, propertyName, propertyTypes
    RETURN DISTINCT relType
    """
    rel_data = session.run(relationship_query)
    for record in rel_data:
        rel_type = record["relType"].lstrip(':').replace('`', '')

        # Modified query to check actual relationship directions
        connection_query = f"""
        MATCH (a)-[r:{rel_type}]->(b)
        WITH labels(a) AS FromLabels, labels(b) AS ToLabels, count(*) as cnt
        RETURN DISTINCT FromLabels, ToLabels, cnt
        ORDER BY cnt DESC
        """
        
        for connection in session.run(connection_query):
            from_labels = [label for label in connection["FromLabels"] 
                         if label != "Resource" and not label.startswith('Resource:')]
            to_labels = [label for label in connection["ToLabels"] 
                        if label != "Resource" and not label.startswith('Resource:')]

            if not from_labels or not to_labels:
                continue

            from_labels_key = ':'.join(from_labels)
            to_labels_key = ':'.join(to_labels)

            # Check if this relationship direction is valid
            validation_query = f"""
            MATCH (a:{from_labels[0]})-[r:{rel_type}]->(b:{to_labels[0]})
            RETURN count(r) as cnt
            """
            count = session.run(validation_query).single()["cnt"]
            
            if count > 0:  # Only add relationships that actually exist in this direction
                schema["relationships"].append({
                    "name": rel_type,
                    "from": from_labels_key,
                    "to": to_labels_key,
                    "count": count
                })

    # Sort relationships by count to prioritize more common relationships
    schema["relationships"].sort(key=lambda x: x.get("count", 0), reverse=True)
    
    # Remove count from final output if you don't want it in the schema
    for rel in schema["relationships"]:
        rel.pop("count", None)

    schema_json = json.dumps(schema, indent=2)
    save_path = "./tools/rdf-to-neo4j/schema.json"
    with open(save_path, "w") as f:
        f.write(schema_json)
    print(f"Schema extracted and saved to {save_path}.")


if __name__ == "__main__":
    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=PROKINO_DATABASE_NAME)
    with driver.session() as session:
        print("Extracting Schema...")
        extract_schema(session)
        print("Extraction complete.")

    driver.close()