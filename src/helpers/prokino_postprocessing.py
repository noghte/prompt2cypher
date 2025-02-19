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

def update_subfamily_labels(session):
    # Query to find all distinct labels ending with 'subfamily'
    query = """
    MATCH (n)
    WHERE ANY(label IN labels(n) WHERE label ENDS WITH 'subfamily')
    RETURN DISTINCT labels(n) AS NodeTypes
    """
    labels_to_change = session.run(query).data()

    # Iterate through each set of labels for each node type
    for entry in labels_to_change:
        for label in entry['NodeTypes']:
            if label.endswith('subfamily'):  # Double check to avoid unnecessary updates
                # Update query to set original type and change label
                update_query = f"""
                MATCH (n:`{label}`)
                SET n.originalType = '{label}', n:Subfamily
                REMOVE n:`{label}`
                """
                session.run(update_query)

def update_family_labels(session):
    # Query to find all distinct labels ending with 'family' but not 'subfamily'
    query = """
    MATCH (n)
    WHERE ANY(label IN labels(n) WHERE label ENDS WITH 'family' AND NOT label ENDS WITH 'subfamily')
    RETURN DISTINCT labels(n) AS NodeTypes
    """
    labels_to_change = session.run(query).data()

    # Iterate through each set of labels for each node type
    for entry in labels_to_change:
        for label in entry['NodeTypes']:
            if label.endswith('family'):  
                # Ensure we do not update nodes that have already been classified as Subfamily
                # Update query to set original type and change label
                # The following query:
                # Ensures the node is not already marked as Subfamily
                # Only updates originalType if it's not already set
                update_query = f"""
                MATCH (n:`{label}`)
                WHERE NOT n:Subfamily  
                SET n.originalType = COALESCE(n.originalType, '{label}'), n:Family
                REMOVE n:`{label}`
                """
                session.run(update_query)



def display_menu():
    print("Select an option:")
    print("1- Update Subfamily nodes")
    print("2- Update Family nodes")
    print("3- Extract Schema")
    print("4- Exit")

def main():
    # Connect to the Neo4j database
    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=PROKINO_DATABASE_NAME)
    with driver.session() as session:
        while True:
            display_menu()
            choice = input("Enter your choice (1-3): ")
            
            if choice == '1':
                print("Updating Subfamily nodes...")
                update_subfamily_labels(session)
                print("Update complete.")
            elif choice == '2':
                print("Updating Family nodes...")
                update_family_labels(session)
                print("Update complete.")
            elif choice == '3':
                print("Exiting...")
                break
            else:
                print("Invalid choice, please try again.")
    
    # Close the Neo4j driver connection
    driver.close()

if __name__ == "__main__":
    main()