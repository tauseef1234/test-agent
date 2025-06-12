import os
from neo4j import GraphDatabase
from langchain_core.runnables import RunnableLambda

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def _verify_customer_in_neo4j(details: dict) -> str:
    """
    Connects to Neo4j and verifies a customer, returning a confirmation or error message.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    query = """
    MATCH (c:Customer)
    WHERE c.first_name = $first_name
    AND c.last_name = $last_name
    AND c.zip_code = $zip_code
    AND c.phone_number = $phone_number
    RETURN c.id AS customer_id, c.name AS full_name
    LIMIT 1
    """

    try:
        with driver.session(database="neo4j") as session:
            result = session.run(query, details)
            record = result.single()

        if record:
            # Return a structured string that the agent can parse/understand
            customer_data = dict(record)
            return f"Verification Successful: Customer Name is {customer_data['full_name']} and Customer ID is {customer_data['customer_id']}."
        else:
            return "Verification Failed: No matching customer found with the provided details. Please ask the user to provide the information again."
    finally:
        driver.close()

# Create a runnable chain from the verification function
customer_verification_chain = RunnableLambda(_verify_customer_in_neo4j)