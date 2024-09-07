import os
import pprint
import torch
from dotenv import load_dotenv
from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from typing import Dict, TypedDict
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM



load_dotenv()
TF_ENABLE_ONEDNN_OPTS=0
# Tavily API key
os.environ['Tavily_API_KEY'] = ""

def load_documents_from_file(file_path):
    documents = []
    with open(file_path, 'r') as file:
        urls = file.readlines()
    for url in urls:
        url = url.strip()
        if url:
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)
    return documents

# File path containing URLs
file_path = 'url.txt'
docs = load_documents_from_file(file_path)

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
)
all_splits = text_splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings()

# Index documents
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
llm = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b")



class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """
    keys: Dict[str, any]

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents,
    """
    print("---RETRIEVE---")
    state_dict = state['keys']
    question = state_dict['question']
    local = state_dict['local']
    
    # Debug: Print the question
    print(f"Question: {question}")

    try:
        response = retriever.invoke({"query": question})
        # Debug: Print the response to understand its structure
        print(f"Response: {response}")

        # Adjust according to the actual response structure
        documents = response.get('documents', [])
    except Exception as e:
        print(f"Error during retrieval: {e}")
        documents = []  # Set to empty list or handle the error as needed

    return {"keys": {"documents": documents, "local": local, "question": question}}




def generate(state):
    """
    Generate answer using Gemma 2

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation,
        that contains LLM generation
    """
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    prompt = hub.pull("rlm/rag-prompt")
    prompt+=" Don't give incomplete sentence in asnwer."
    # Prepare the prompt text
    context = "\n\n".join(doc.page_content for doc in documents)
    input_text = f"Context: {context}\nQuestion: {question}"

    # Tokenize and generate
    inputs = tokenizer(input_text, return_tensors="pt")

    # Ensure inputs["input_ids"] is a tensor
    if not isinstance(inputs["input_ids"], torch.Tensor):
        raise TypeError("Expected 'input_ids' to be a tensor")

    outputs = llm.generate(
    inputs["input_ids"],
    max_length=512, 
    num_return_sequences=1,
    do_sample=True, 
    temperature=0.5,  # Adjust this value for more varied responses
    top_k=30,  # Adjust to control sampling
    top_p=0.8   # Adjust to control cumulative probability
)
    generation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "keys": {"documents": documents, "question": question, "generation": generation}
    }

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """
    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]

    from langchain_core.output_parsers import BaseOutputParser

    class CustomOutputParser(BaseOutputParser):
        def get_format_instructions(self):
            return "Provide the binary score as a string with no preamble or explanation."

        def parse(self, output):
            # Implement parsing logic here
            return output.strip()

    parser = CustomOutputParser()

    # Update prompt creation
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved
                    document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question,
        grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out
        erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the
        document is relevant to the question. \n
        Provide the binary score as a string with no preamble or
        explanation and use these instructions to format the output:
        {format_instructions}""",
        input_variables=["question", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )



    chain = prompt | llm | parser

    # Score
    filtered_docs = []
    search = "No"
    for d in documents:
        score = chain.invoke(
            {
                "question": question,
                "context": d.page_content,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        grade = score.strip() 
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            search = "Yes"
            continue

    return {
        "keys": {
            "documents": filtered_docs,
            "question": question,
            "local": local,
            "run_web_search": search,
        }
    }


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]

    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for
                    retrieval. \n
        Look at the input and try to reason about the underlying semantic
        intent / meaning. \n
        Here is the initial question:
        \n ------- \n
        {question}
        \n ------- \n
        Provide an improved question without any premable, only respond
        with the updated question: """,
        input_variables=["question"],
    )
    # Prompt
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {
        "keys": {"documents": documents, "question": better_question,
                 "local": local}
    }

def web_search(state):
    """
    Web search based on the re-phrased question using Tavily API.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Web results appended to documents.
    """
    print("---WEB SEARCH---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]

    try:
        tool = TavilySearchResults()
        docs = tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)
    except Exception as error:
        print(error)

    return {"keys": {"documents": documents, "local": local, "question": question}}

def decide_to_generate(state):
    """
    Determines whether to generate an answer or re-generate a question
    for web search.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """
    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    filtered_documents = state_dict["documents"]
    search = state_dict["run_web_search"]

    if search == "Yes":
        # No relevant documents, so transform query and run web search
        print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE ANSWER---")
        return "generate"



def run_langchain_workflow(question: str):
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate
    workflow.add_node("transform_query", transform_query)  # transform query
    workflow.add_node("web_search", web_search)  # web search

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()
    inputs = {
        "keys": {
            "question": question,
            "local": 'No',
        }
    }
    final_output = None
    for output in app.stream(inputs):
        final_output = output  
    if final_output and 'generate' in final_output and 'keys' in final_output['generate']:
        generation = final_output['generate']['keys'].get('generation', "")
        if "Answer:" in generation:
            answer = generation.split("Answer:", 1)[1].strip()
            return answer
        else:
            return "No answer found in the output."
    else:
        return "Error: No keys found in final output"
