from pypdf import PdfReader 
import nltk
from nltk.tokenize import sent_tokenize
import tqdm

# Download the necessary NLTK models for sentence tokenization
nltk.download('punkt')

def merge_consecutive_sentences(sentences, chunk_size=4):
    merged_sentences = []
    
    # Iterate through the list in chunks of 'chunk_size' sentences
    for i in range(0, len(sentences), chunk_size):
        # Get the current chunk of sentences
        chunk = sentences[i:i + chunk_size]
        
        # Merge the sentences in the current chunk
        merged_sentence = ' '.join(chunk)
        
        # Add the merged sentence to the result list
        merged_sentences.append(merged_sentence)
    
    return merged_sentences

def extract_sentences(text):
    # Use NLTK's sent_tokenize to split the text into sentences
    sentences = sent_tokenize(text)

    return merge_consecutive_sentences(sentences)

def extract_text_from_pdf(pdf_path, max_pages=4):
    # creating a pdf reader object 
    reader = PdfReader(pdf_path) 
    # getting a specific page from the pdf file 

    sentences = []
    for page in tqdm.tqdm(reader.pages[:max_pages]):
        text = page.extract_text()
        sentences.extend(extract_sentences(text))
    
    return sentences

if __name__ == "__main__":
    # creating a pdf reader object 
    example_file = "/home/ulrich/Downloads/bend_benchmark_paper.pdf" 
    

    # Extract sentences from the text
    sentences = extract_text_from_pdf(example_file)
    print("Number of sentences:", len(sentences))

    # Print the extracted sentences
    for i, sentence in enumerate(sentences[0:10]):
        print(f"Sentence {i+1}: {sentence}")

