# Create a RAG with LLM and Qdrant using your own data

In this lab you will implement the RAG pattern with your own data. Use the example code in this repository from the [Applied Rag Notebook](./examples/3-applied-rag/embeddings.ipynb) as reference. The end result should be in your own repository containing the complete code for the enhanced RAG pattern based on the example provided.

**Learning Objectives:**

* Implement the RAG pattern with your own data
* Apply your own data to solve a problem using RAG
* Understand how to leverage an LLM and a vector database like Qdrant for useful responses

## Steps

1. Create a new repository in your account for your project. Alternatively, you can use this repository as a starting point by forking the repository. [Use this link to create it in one step.](https://github.com/alfredodeza/learn-retrieval-augmented-generation/generate).
2. use the [Applied Rag notebook](./examples/3-applied-rag/embeddings.ipynb) as a starting point
3. Replace the example data with your own data. Make sure to format it as a list of dictionaries for easier ingestion into the vector database
4. Run the LLM with Llamafile or connect your appliation to a valied OpenAI API endpoint
5. Run the notebook or Python application and verify your work

**Use Llamafile for a full RAG and LLM setup**

The examples for the [Applied Rag notebook](https://github.com/alfredodeza/learn-retrieval-augmented-generation/blob/main/examples/3-applied-rag/embeddings.ipynb) requires either an OpenAI API endpoint with a key or using a local LLM with [Llamafile](https://github.com/Mozilla-Ocho/llamafile).
<br>
I recommend using the [Phi-2 model](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#other-example-llamafiles) which is about 2GB in size. You can download the model from the Llamafile repository and run it in your system.
<br>
Once you have it running you can connect to it with Python or use the [Applied Rag Notebook](./examples/3-applied-rag/embeddings.ipynb). Here is a quick example of how to use the Llamafile with Python:

```python

#!/usr/bin/env python3
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8080/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required" # An API key is not required!
)
completion = client.chat.completions.create(
    model="LLaMA_CPP",
    messages=[
        {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
        {"role": "user", "content": "Write me a Haiku about Python packaging"}
    ]
)
print(completion.choices[0].message)

```

## Concepts Covered

* Retrieval Augmented Generation
* Large Language Models using Llamafile
* Using Vector databases like Qdrant
* Creating embeddings with Sentence Transformers
* Using OpenAI's Python API to connect to the LLM and produce responses

By completing this lab you will have a working example of the RAG pattern with your own data. You will also have a better understanding of how to use LLMs and vector databases to create useful responses for your applications.
