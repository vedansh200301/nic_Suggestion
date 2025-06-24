import os
import openai
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_restful import reqparse, Api, Resource
from dotenv import load_dotenv
from flask_swagger_ui import get_swaggerui_blueprint

# Load environment variables
load_dotenv()

# Configure OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app and API
app = Flask(__name__)
CORS(app) # This will enable CORS for all routes
api = Api(app)

# --- Swagger UI Configuration ---
SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI
APP_PREFIX = os.getenv('APP_PREFIX', '')
API_URL = f'{APP_PREFIX}/static/swagger.yml' # Our API definition
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "NIC Code Suggestion API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


# --- Load NIC Codes and Embeddings ---
try:
    nic_df = pd.read_csv("nic_embeddings.csv")
    # Convert string representation of list to actual list of floats
    nic_df['embedding'] = nic_df['embedding'].apply(eval).apply(np.array)
    print("NIC embeddings CSV loaded successfully.")
except FileNotFoundError:
    nic_df = None
    print("WARNING: nic_embeddings.csv not found.")
    print("Please run `python create_embeddings.py` first to generate it.")


# --- Helper function to get embedding for a query ---
def get_query_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   try:
       return openai.embeddings.create(input=[text], model=model).data[0].embedding
   except Exception as e:
       print(f"Error getting query embedding: {e}")
       return None

# --- New search function using embeddings ---
def find_relevant_codes_with_embeddings(description, top_n=5):
    if nic_df is None or nic_df.empty:
        return ""

    query_embedding = get_query_embedding(description)
    if query_embedding is None:
        return ""

    # Calculate cosine similarity
    nic_df["similarity"] = nic_df["embedding"].apply(lambda x: cosine_similarity([x], [query_embedding])[0][0])
    
    # Get top N most similar codes
    relevant_codes = nic_df.sort_values(by="similarity", ascending=False).head(top_n)

    if relevant_codes.empty:
        return ""
        
    context_str = "Consider these potentially relevant NIC codes based on semantic similarity:\n"
    for _, row in relevant_codes.iterrows():
        context_str += f"- Code: {row['NICCode']}, Description: {row['Description']}\n"
        
    return context_str


# --- Frontend Route ---
@app.route('/')
def index():
    return render_template('index.html')

# --- API Resources ---

# Parser for the new single-call suggestion API
suggestion_parser = reqparse.RequestParser()
suggestion_parser.add_argument('business_details', type=str, required=True, help='Detailed business description cannot be blank!')

class SuggestionAPI(Resource):
    def post(self):
        args = suggestion_parser.parse_args()
        context = args['business_details']
        
        # API Key Check
        if not openai.api_key or "YOUR_OPENAI_API_KEY_HERE" in openai.api_key:
            print("ERROR: OpenAI API key is not configured.")
            return {'message': 'OpenAI API key is not configured. Please set it in the .env file.'}, 500

        try:
            csv_context = find_relevant_codes_with_embeddings(context)
            user_prompt = f"Given this detailed business context: '{context}', suggest the 1-2 most relevant Indian NIC codes, with brief reasoning for each."
            system_prompt = f"""You are an expert in Indian NIC codes. You must suggest only 5-digit NIC codes. Your response must be a valid JSON array where each object has 'code' and 'description' keys.
{csv_context}
Based on the user's business context and the provided list of semantically similar codes, select the best NIC code. If none of the provided codes are a good fit, use your general knowledge to find the correct one."""

            print("---PROMPT SENT TO OPENAI---")
            print(system_prompt)
            print("---------------------------")

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
            )
            
            suggestion_text = response.choices[0].message.content
            
            # --- Robust JSON Parsing ---
            # Find the start and end of the JSON array
            try:
                start = suggestion_text.find('[')
                end = suggestion_text.rfind(']') + 1
                if start != -1 and end != 0:
                    json_str = suggestion_text[start:end]
                    suggestions = json.loads(json_str)
                else:
                    raise json.JSONDecodeError("No JSON array found", suggestion_text, 0)
            except json.JSONDecodeError:
                 print(f"OpenAI returned non-JSON response: {suggestion_text}")
                 return {'message': 'The model returned an invalid format. Please try again.'}, 500
            
            return jsonify(nic_codes=suggestions)

        except Exception as e:
            print(f"An error occurred: {e}")
            return {'message': f'Error generating suggestion: {str(e)}'}, 500

# Parser for feedback
feedback_parser = reqparse.RequestParser()
feedback_parser.add_argument('satisfied', type=bool, required=True, help='Satisfaction feedback is required')
feedback_parser.add_argument('additional_info', type=str)

class FeedbackAPI(Resource):
    def post(self):
        args = feedback_parser.parse_args()
        # In a real app, you'd log this feedback to a database for analysis
        print(f"Feedback received: Satisfied={args['satisfied']}, Info='{args['additional_info']}'")
        return {'message': 'Feedback received successfully'}, 200

# --- Add resources to API ---
api.add_resource(SuggestionAPI, '/api/suggest')
api.add_resource(FeedbackAPI, '/api/feedback')

if __name__ == '__main__':
    app.run(debug=True)
