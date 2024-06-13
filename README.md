# weaviate-tutorial

**Installation**

```commandline
git clone https://github.com/ulricharmel/weaviate-tutorial.git
cd weaviate-tutorial
pip install -r requirements.txt
```

## Workflow
The workflow code in workflow.py implements a basic RAG, that gets it context from an input pdf file. 
It uses ollama and llama3 to compute embeddings and anwser questions.
Before running the script, create an environment file (.env) using the template .env-examples. 
Make sure to download ollama and pull the model version you specify in your environment file.

The most expensive part of the code is the step that computes and adds the vector embeddings to the weaviate database. Hence use --pages flag to use only a few pgaes from your document. The chunking approach I use is to merge every 4 sentences together as one entry for the database.

```commandline
python workflow.py [input_pdf_file] --pages [number_of_pages]
```
