import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from retrieval import retrieve_relevant_data, upsert_screenplay_vectors
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

# Download NLTK resources
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
load_dotenv()

# Load LLaMA model
HF_TOKEN = os.getenv("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token=HF_TOKEN)

def generate_screenplay(user_input, genre):
    """Generate a bilingual (Telugu + English) screenplay."""
    tokens = word_tokenize(user_input)
    tagged_words = pos_tag(tokens)
    keywords = [word for word, tag in tagged_words if tag in ["NN", "NNS", "NNP", "NNPS", "JJ"]]
    context = retrieve_relevant_data(keywords, genre)
    prompt = f"""
    You are a professional screenwriter creating a screenplay in the {genre} genre.
    The screenplay should be written in **bilingual format (Telugu in English script + English).**

    ## **Example Scene**
    INT. COFFEE SHOP - NIGHT  

    ARJUN  
    (excited)  
    *"Evaru ra nuvvu? Nen eppudu ne choodaledhu... But you look familiar."*  

    RADHA  
    (smiling)  
    *"Neeku teliyadu kaani, fate lo anni connections untai. Life is strange!"*  

    ## **Scene Details**  
    **User Input:**  
    "{user_input}"  

    **Relevant Context:**  
    "{context}"  

    Now, generate a well-structured screenplay scene.
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    output_tokens = model.generate(
        **inputs,
        max_length=2048,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    screenplay = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    upsert_screenplay_vectors(str(hash(screenplay)), screenplay, genre)

    return screenplay
