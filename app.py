import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Page title
st.set_page_config(page_title='Interroga file csv')
st.title('Interroga tabelle di dati in formato csv')



# Load CSV file
def load_csv(input_csv):
  df = pd.read_csv(input_csv)
  with st.expander('Guarda la tabella'):
    st.write(df)
  return df


# Generate LLM response
def generate_response(csv_file, input_query):
  llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.2, openai_api_key=openai_api_key)
  df = load_csv(csv_file)
  # Create Pandas DataFrame Agent
  agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
  # Perform Query using the Agent
  response = agent.run(input_query)
  return st.success(response)


# Input widgets
uploaded_file = st.file_uploader('Carica il tuo file csv', type=['csv'])
question_list = [
  'Quante righe ha il file?',
  'Quali sono le colonne del file?',
  'Other']
query_text = st.selectbox('Scegli una domanda:', question_list, disabled=not uploaded_file)
openai_api_key = st.text_input('Chiave di OpenAI API', type='password', disabled=not (uploaded_file and query_text))

# App logic
if query_text is 'Other':
  query_text = st.text_input('Inserisci la tua domanda:', placeholder = 'Scrivi qui la tua domanda ...', disabled=not uploaded_file)
if not openai_api_key.startswith('sk-'):
  st.warning('Inserisci la chiave di Open AI!', icon='⚠')
if openai_api_key.startswith('sk-') and (uploaded_file is not None):
  st.header('Output')
  generate_response(uploaded_file, query_text)

