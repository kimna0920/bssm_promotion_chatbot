import streamlit as st
st.title("Title")
st.header("Header")
st.subheader("Sub Header")

st.write("Hello World!")

import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model
@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('bsg_chat.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df
model = cached_model()
df = get_dataset()

if 'generated' not in st.session_state:
    st.session_state['generated']
if 'past' not in st.session_state:
    st.session_state['past']
    
if submited and user_input:
    embedding = model.encode(user_input)
    
    df['distance'] = df['embedding'].map(lambda x:cosine_similarity([embedding],[x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]
    
    st.session_state.past.append(user_input)
    if answer['distance'] > 0.5 :
        st.session_state.generated.append(answer['챗봇'])
    else :
        st.session_state.generated.append("무슨 말인지 모르겠어요.")
        
    for i in range(len(st.session_state['past'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        if len(st.session_state['generated']) > i :
            message(st.session_state['generated'][i], key=str(i) + '_bot')