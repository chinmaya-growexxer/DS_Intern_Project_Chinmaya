import streamlit as st
import snowflake.connector
import configparser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Function to connect to Snowflake
def connect_to_snowflake():
    config = configparser.ConfigParser()
    config.read('config.ini')
    snowflake_config = config['snowflake']
    
    conn = snowflake.connector.connect(
        user=snowflake_config['user'],
        password=snowflake_config['password'],
        account=snowflake_config['account'],
        warehouse=snowflake_config['warehouse'],
        database=snowflake_config['database'],
        schema=snowflake_config['schema'],
        role=snowflake_config['role'],
    )
    return conn

# Function to fetch table and column information from the database schema
def get_db_schema(conn, table_schema, table_name):
    cursor = conn.cursor()
    query = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{table_schema}' AND TABLE_NAME = '{table_name}'"
    cursor.execute(query)
    schema_info = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return schema_info

# Function to create SQL query generation chain
def generate_sql_chain(conn, schema, target_table, question, llm_model, api_key):
    schema_info = get_db_schema(conn, schema, target_table)
    template = f"""
        You are a Snowflake SQL expert. Given an input question, create a syntactically correct Snowflake SQL query to run and return ONLY the generated query and nothing else. 

        Here is the relevant table info: {schema_info}. 
        Pay close attention to which column is in the table and ensure the query includes the table name '{target_table}' in the FROM clause.

        Follow these instructions for creating a syntactically correct SQL query:
        - Only provide the SQL query, without any additional text or characters.
        - Be sure not to query for columns that do not exist in the table.
        - Use the exact column names from the schema: {schema_info}.
        - If the question involves correlation, ensure that two columns are specified and use the CORR function correctly.
        - Always use appropriate aggregation functions such as AVG, SUM, COUNT, etc., when asked for averages, totals, or counts.
        - Pay close attention to the filtering criteria mentioned in the question and incorporate them using the WHERE clause.
        - If the question involves multiple conditions, use logical operators such as AND, OR to combine them effectively.
        - When dealing with date or timestamp columns, use appropriate date functions for extracting specific parts of the date or performing date arithmetic.
        - If the question involves grouping of data, use the GROUP BY clause along with appropriate aggregate functions.
        - Consider using aliases for columns to improve readability of the query, especially in case of complex joins or subqueries.

        Table schema: {schema_info}
        Question: {question}

        SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    llm = ChatGroq(model=llm_model, temperature=0.2, groq_api_key=api_key)

    return (
        RunnablePassthrough(assignments={"schema": schema_info, "question": question})
        | prompt
        | llm
        | StrOutputParser()
    )

# Function to execute a query and fetch results
def execute_query(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    return results

# Function to create natural language response based on SQL query results
def generate_sql_to_nl(conn,question, sql_query, results, llm_model, api_key):
    results_str = "\n".join([str(row) for row in results])
    template = f"""
        Based on the results of the SQL query '{sql_query}', write a natural language response.
        Consider the initial user query'{question}' for context and frame the response accordingly.
        Query Results:
        {results_str}
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    llm = ChatGroq(model=llm_model, temperature=0.2, groq_api_key=api_key)
    return (
        RunnablePassthrough(assignments={"sql_query": sql_query, "results": results_str})
        | prompt
        | llm
        | StrOutputParser()
    )

# def main():
#     st.title("Ask My Database")

#     st.markdown("""
#     <style>
#     .main {background-color: #f0f0f0;}
#     .stButton button {background-color: #4CAF50; color: white; font-size: 16px; padding: 10px 24px;}
#     </style>
#     """, unsafe_allow_html=True)

#     # Read the config file
#     config = configparser.ConfigParser()
#     config.read('config.ini')

#     # Get the Snowflake and API details
#     snowflake_config = config['snowflake']
#     api_config = config['api']
#     llm_model = 'gemma2-9b-it'
#     groq_api_key = api_config['groq_api_key']
#     target_table = snowflake_config['target_table']

#     # Initialize session state
#     if 'sql_query' not in st.session_state:
#         st.session_state.sql_query = None
#     if 'results' not in st.session_state:
#         st.session_state.results = None
#     if 'nlp_response' not in st.session_state:
#         st.session_state.nlp_response = None
#     if 'success_message' not in st.session_state:
#         st.session_state.success_message = None
#     if 'query_history' not in st.session_state:
#         st.session_state.query_history = []

#     # User input
#     user_query = st.text_input("Ask your database a question:", key="user_query_input")

#     if st.button("Submit"):
#         if not user_query:
#             st.error("Please enter a question.")
#         else:
#             # Connect to Snowflake
#             conn = connect_to_snowflake()
#             st.session_state.success_message = "Connected to the database successfully!"

#             # Generate SQL query
#             sql_chain = generate_sql_chain(conn, snowflake_config['schema'], target_table, user_query, llm_model, groq_api_key)
#             sql_query_response = sql_chain.invoke({})
#             sql_query = sql_query_response.strip()
#             print(sql_query)

#             # Execute SQL query
#             results = execute_query(conn, sql_query)
#             if results:
#                 # Generate natural language response
#                 nlp_chain = generate_sql_to_nl(conn, sql_query, results, llm_model, groq_api_key)
#                 nlp_response = nlp_chain.invoke({})
#                 st.session_state.sql_query = sql_query
#                 st.session_state.results = results
#                 st.session_state.nlp_response = nlp_response

#                 # Save query and response to history
#                 st.session_state.query_history.append({
#                     "question": user_query,
#                     "sql_query": sql_query,
#                     "results": results,
#                     "nlp_response": nlp_response
#                 })

#                 st.session_state.success_message = f"**Natural Language Response:**\n{nlp_response}"
#             else:
#                 st.warning("No results found or error occurred.")

#             conn.close()

#     # Display success message
#     if st.session_state.success_message:
#         st.markdown(st.session_state.success_message)

#     # Toggle to show SQL query and results
#     if st.session_state.sql_query and st.session_state.results:
#         show_sql = st.checkbox("Show SQL Query and Results", key="show_sql_checkbox")
#         if show_sql:
#             st.write(f"**Generated SQL Query:**\n{st.session_state.sql_query}")
#             st.write("**Results:**")
#             for row in st.session_state.results:
#                 st.write(row)

#     # Sidebar to show query history
#     st.sidebar.title("Query History")
#     for idx, query in enumerate(st.session_state.query_history):
#         st.sidebar.markdown(f"**Query {idx+1}:** {query['question']}")
#         with st.sidebar.expander(f"Details for Query {idx+1}"):
#             st.write(f"**Generated SQL Query:**\n{query['sql_query']}")
#             st.write("**Results:**")
#             for row in query['results']:
#                 st.write(row)
#             st.write(f"**Natural Language Response:**\n{query['nlp_response']}")

# if __name__ == "__main__":
#     main()



# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# def main():
#     st.title("Ask My Database")

#     st.markdown("""
#     <style>
#     .main {background-color: #f0f0f0;}
#     .stButton button {background-color: #4CAF50; color: white; font-size: 16px; padding: 10px 24px;}
#     </style>
#     """, unsafe_allow_html=True)

#     # Read the config file
#     config = configparser.ConfigParser()
#     config.read('config.ini')

#     # Get the Snowflake and API details
#     snowflake_config = config['snowflake']
#     api_config = config['api']
#     llm_model = 'gemma2-9b-it'
#     groq_api_key = api_config['groq_api_key']
#     target_table = snowflake_config['target_table']

#     # Initialize session state
#     if 'sql_query' not in st.session_state:
#         st.session_state.sql_query = None
#     if 'results' not in st.session_state:
#         st.session_state.results = None
#     if 'nlp_response' not in st.session_state:
#         st.session_state.nlp_response = None
#     if 'success_message' not in st.session_state:
#         st.session_state.success_message = None
    
#     # User input
#     user_query = st.text_input("Ask your database a question:")

#     if st.button("Submit"):
#         if not user_query:
#             st.error("Please enter a question.")
#         else:
#             # Connect to Snowflake
#             conn = connect_to_snowflake()
#             st.session_state.success_message = "Connected to the database successfully!"

#             # Generate SQL query
#             sql_chain = generate_sql_chain(conn, snowflake_config['schema'], target_table, user_query, llm_model, groq_api_key)
#             sql_query_response = sql_chain.invoke({})
#             sql_query = sql_query_response.strip()

#             # Execute SQL query
#             results = execute_query(conn, sql_query)
#             if results:
#                 # Display SQL query and results
#                 st.write(f"**Generated SQL Query:**\n{sql_query}")
#                 st.write("**Results:**")
#                 df = pd.DataFrame(results, columns=["Customer", "Cost"])
#                 st.write(df)

#                 # Visualize top 10 customers with highest cost
#                 top_10_customers = df.nlargest(10, "Cost")
#                 plt.figure(figsize=(10, 6))
#                 sns.barplot(x="Customer", y="Cost", data=top_10_customers, palette="viridis")
#                 plt.title("Top 10 Customers with Highest Cost")
#                 plt.xlabel("Customer")
#                 plt.ylabel("Cost")
#                 st.pyplot()

#                 # Generate natural language response
#                 nlp_chain = generate_sql_to_nl(conn, sql_query, results, llm_model, groq_api_key)
#                 nlp_response = nlp_chain.invoke({})
#                 st.session_state.success_message = f"**Natural Language Response:**\n{nlp_response}"

#                 st.session_state.sql_query = sql_query
#                 st.session_state.results = results
#                 st.session_state.nlp_response = nlp_response
#             else:
#                 st.warning("No results found or error occurred.")

#             conn.close()

#     # Display success message
#     if st.session_state.success_message:
#         st.markdown(st.session_state.success_message)

#     # Toggle to show SQL query and results
#     if st.session_state.sql_query and st.session_state.results:
#         show_sql = st.checkbox("Show SQL Query and Results")
#         if show_sql:
#             st.write(f"**Generated SQL Query:**\n{st.session_state.sql_query}")
#             st.write("**Results:**")
#             df = pd.DataFrame(st.session_state.results, columns=["Customer", "Cost"])
#             st.write(df)

# if __name__ == "__main__":
#     main()


def main():
    st.title("Ask My Database")

    st.markdown("""
    <style>
    .main {background-color: #f0f0f0;}
    .stButton button {background-color: #4CAF50; color: white; font-size: 16px; padding: 10px 24px;}
    </style>
    """, unsafe_allow_html=True)

    # Read the config file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Get the Snowflake and API details
    snowflake_config = config['snowflake']
    api_config = config['api']
    llm_model = 'gemma2-9b-it'
    groq_api_key = api_config['groq_api_key']
    target_table = snowflake_config['target_table']

    # Initialize session state
    if 'sql_query' not in st.session_state:
        st.session_state.sql_query = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'nlp_response' not in st.session_state:
        st.session_state.nlp_response = None
    if 'success_message' not in st.session_state:
        st.session_state.success_message = None
    
    # User input
    user_query = st.text_input("Ask your database a question:")

    if st.button("Submit"):
        if not user_query:
            st.error("Please enter a question.")
        else:
            # Connect to Snowflake
            conn = connect_to_snowflake()
            st.session_state.success_message = "Connected to the database successfully!"

            # Generate SQL query
            sql_chain = generate_sql_chain(conn, snowflake_config['schema'], target_table, user_query, llm_model, groq_api_key)
            sql_query_response = sql_chain.invoke({})
            sql_query = sql_query_response.strip()
            print(sql_query)

            # Execute SQL query
            results = execute_query(conn, sql_query)
            if results:
                # Generate natural language response
                nlp_chain = generate_sql_to_nl(conn,user_query, sql_query, results, llm_model, groq_api_key)
                nlp_response = nlp_chain.invoke({})
                st.session_state.sql_query = sql_query
                st.session_state.results = results
                st.session_state.nlp_response = nlp_response

                st.session_state.success_message = f"**Natural Language Response:**\n{nlp_response}"
            else:
                st.warning("No results found or error occurred.")

            conn.close()

    # Display success message
    if st.session_state.success_message:
        st.markdown(st.session_state.success_message)

    # Toggle to show SQL query and results
    if st.session_state.sql_query and st.session_state.results:
        show_sql = st.checkbox("Show SQL Query and Results")
        if show_sql:
            st.write(f"**Generated SQL Query:**\n{st.session_state.sql_query}")
            st.write("**Results:**")
            for row in st.session_state.results:
                st.write(row)

if __name__ == "__main__":
    main()

