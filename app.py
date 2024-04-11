import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Page title
st.set_page_config(page_title='🦜🔗 Data App Conversazionale')
st.title('🦜🔗 Data App Conversazionale')

# Aggiunta del campo per caricare la descrizione del file CSV
uploaded_description = st.text_area('Carica la descrizione del file CSV', height=100)

# Load CSV file
def load_csv(input_csv):
  df = pd.read_csv(input_csv)
  with st.expander('See DataFrame'):
    st.write(df)
  return df


# Generate LLM response
def generate_response(csv_file, input_query, uploaded_description):
  system_prompt = f"Sei un archivista e devi analizzare un elenco di fascicoli. Questa è una descrizione del contenuto che dovrai analizzare: {uploaded_description}."
  llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.2, openai_api_key=openai_api_key, system_prompt=system_prompt)
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
  'Quali sono le tipologie di beni?',
  'Quali sono le città menzionate nel file?',
  'Other']
query_text = st.selectbox('scegli una domanda:', question_list, disabled=not uploaded_file)
openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))

# App logic
if query_text is 'Other':
  query_text = st.text_input('Inserisci la tua domanda:', placeholder = 'Scrivi qui la tua domanda ...', disabled=not uploaded_file)
# App logic
if openai_api_key.startswith('sk-') and (uploaded_file is not None) and (uploaded_description is not None):
  st.header('Output')
  generate_response(uploaded_file, uploaded_description, query_text)

