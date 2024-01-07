import gradio as gr
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores.chroma import Chroma



import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


def process_pdf(pdf_file):
    loader = PyMuPDFLoader(pdf_file.name)
    documents = loader.load() 
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    text = text_splitter.split_documents(documents)
    db = Chroma.from_documents(text,embeddings)
    llm = HuggingFaceHub(repo_id = "OpenAssistant/oasst-sft-1-pythia-12b")
    global chain 
    chain = RetrievalQA.from_chain_type(llm = llm, chain_type="stuff", retriever = db.as_retriever())
    return "Document has been loaded successfully"

def answer_query(query):
    question = query
    return chain.run(question)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    pdf_file = gr.File(label='file upload', file_types=['.pdf'])
    with gr.Row(): 
        load_pdf = gr.Button("Load a pdf")
        status = gr.Textbox(label = 'status', placeholder='', interactive= False)
    with gr.Row():
        input = gr.Textbox(label = 'type in your query')
        output = gr.Textbox(label = 'output')

    submit = gr.Button('submit')
    submit.click(answer_query, input, output)
    load_pdf.click(process_pdf, pdf_file, status)

demo.launch()
