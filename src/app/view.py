"""TODO"""

import re
import streamlit as st
import requests
import json



st.title(
    "_:blue[Local GenAI Search]_ :sunglasses"
)

question = st.text_input("Ask a question based on your local files", "")


if st.button("Ask a question"):
    
    st.write("The current quesution is \"", question+"\"")
    url = "http://127.0.0.1:8080/ask_localai"

    payload = json.dumps(
        {"query": question}
    )

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    response = requests.request(
        "POST",
        url=url,
        headers=headers,
        data=payload,
        timeout=300
    )

    answer = json.loads(response.text)["answer"]

    rege = re.compile("\Document\ [0-9]+\]|[[0-9]+\]")
    m = rege.findall(answer)
    num = []

    for n in m:

        num = num + [int(s) for s in re.findall(r'\b\d+\b', n)]

    st.markdown(answer)
    documents = json.loads(response.text)['context']
    show_docs = []

    for n in num:
        for doc in documents:
            if int(doc['id']) == n:
                show_docs.append(doc)

    a = 1024

    for doc in show_docs:

        with st.expander(str(doc['id'])+ " - " + doc['path']):

            st.write(doc['context'])

            with open(doc['path'], 'rb') as fo:
                st.download_button("Download file", fo, file_name=doc['path'].split('/')[-1], key=a)
                a = a + 1

