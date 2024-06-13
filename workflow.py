import os
import argparse
import weaviate
import weaviate.classes.config as wc
import ollama
from dotenv import load_dotenv
from read_pdf import extract_text_from_pdf
import time

load_dotenv()
model = os.getenv("ollama_model")

def get_parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Path to the input pdf file.')
    parser.add_argument("--pages", type=int, help="Number of pages", default=4)

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = get_parser_args()
    print("Input file:", args.input_file)
    sentences = extract_text_from_pdf(args.input_file, args.pages) # Extract the first 20 sentences
    print(f"Document has {len(sentences)} sentences.")

    sentences = sentences[:50]

    # create a Weaviate client
    client = weaviate.connect_to_wcs(
        cluster_url=os.getenv("cluster_url"),  # Replace with your Weaviate Cloud URL
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("auth_credentials")),  # Replace with your Weaviate Cloud key
        # headers={'X-OpenAI-Api-key': os.getenv("OPENAI_APIKEY")}  # Replace with your vectorizer API key
    )

    client.collections.delete(name="Document")
    assert client.is_connected(), "Connection to Weaviate failed"

    collection = client.collections.create(
        name="Document",
        properties=[
            wc.Property(name="text", data_type=wc.DataType.TEXT),
        ],
    )

    t0 = time.time()
    # Add sentences to the document database
    with collection.batch.dynamic() as batch:
        for sentence in sentences:
            # Generate embeddings
            response = ollama.embeddings(model=model, prompt=sentence)

            # Add data object with text and embedding
            batch.add_object(
                properties = {"text" : sentence},
                vector = response["embedding"],
            )
    
    print(f"Time taken to add all sentences: {time.time() - t0:.2f} seconds\n")

    while True:
        prompt = input("Type your question or type exit to end the program: \n")
        if prompt == "exit":
            break

        # Generate an embedding for the prompt and retrieve the most relevant doc
        response = ollama.embeddings(model=model, prompt=prompt)   

        results = collection.query.near_vector(near_vector=response["embedding"], limit=1)
        data = results.objects[0].properties['text']

        print(f"Retrieved context: {data}")
        print(f"--------------------------")

        ## Augment the prompt with the retrieved data
        prompt_template = f"Using this data: {data}. Respond to this prompt: {prompt}"
        output = ollama.generate(
            model = model,
            prompt = prompt_template,
        )

        print(f"{output['response']}\n")


    client.close()




