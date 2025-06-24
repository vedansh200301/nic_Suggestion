import os
import pandas as pd
import openai
from dotenv import load_dotenv
import time
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   # Add a delay to avoid hitting rate limits
   time.sleep(0.1) 
   try:
       return openai.embeddings.create(input=[text], model=model).data[0].embedding
   except Exception as e:
       print(f"Error getting embedding for text: {text}")
       print(e)
       return None

def create_embeddings():
    # Check for API key
    if not openai.api_key or "YOUR_OPENAI_API_KEY_HERE" in openai.api_key:
        print("ERROR: OpenAI API key is not configured.")
        print("Please set your key in the .env file before running this script.")
        return

    try:
        # Load the NIC codes CSV
        df = pd.read_csv("mca_nic_codes.csv")
        
        # Ensure the 'Description' column exists
        if 'Description' not in df.columns or 'NICCode' not in df.columns:
            print("Error: The CSV must contain 'NICCode' and 'Description' columns.")
            return

        print("Generating embeddings for each NIC code description...")
        
        # Use tqdm for a visible progress bar
        tqdm.pandas(desc="Generating Embeddings")
        df["embedding"] = df['Description'].progress_apply(lambda x: get_embedding(x))
        
        # Drop rows where embedding failed
        df.dropna(subset=['embedding'], inplace=True)

        # Save the DataFrame with embeddings to a new CSV
        df.to_csv("nic_embeddings.csv", index=False)

        print("\nSuccessfully created embeddings and saved them to nic_embeddings.csv")
        print(f"Processed {len(df)} rows.")

    except FileNotFoundError:
        print("Error: mca_nic_codes.csv not found.")
        print("Please make sure the file is in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    create_embeddings() 