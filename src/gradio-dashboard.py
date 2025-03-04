import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables and data
load_dotenv()
books = pd.read_csv("data/books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].fillna("src/cover-not-found.jpg") + "&fife=w800"

google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    raise ValueError("GEMINI_API_KEY not found. Ensure it is set in your .env file.")

raw_documents = TextLoader("data/tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

persist_directory = "db_books"  # Directory to store your database

if os.path.exists(persist_directory) and os.listdir(persist_directory):
    # Load the existing database from disk
    db_books = Chroma(
        persist_directory=persist_directory,
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    )
else:
    # Create a new database and persist it automatically by providing persist_directory
    db_books = Chroma.from_documents(
        documents,
        GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key),
        persist_directory=persist_directory
    )


def retrieve_semantic_recommendations(query: str, category: str = "All", tone: str = "All", 
                                       initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
    """
    Retrieves book recommendations based on semantic similarity.
    """
    recs = db_books.similarity_search(query, k=initial_top_k)
    book_ids = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(book_ids)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    
    tone_columns = {"Happy": "joy", "Surprising": "surprise", "Angry": "anger", 
                    "Suspenseful": "fear", "Sad": "sadness"}
    if tone in tone_columns:
        book_recs = book_recs.sort_values(by=tone_columns[tone], ascending=False)
    
    return book_recs

def recommend_books(query: str, category: str, tone: str):
    """
    Fetches recommended books and formats them for display.
    """
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    
    for _, row in recommendations.iterrows():
        truncated_description = " ".join(row["description"].split()[:30]) + "..."
        
        # Format author names
        authors = row["authors"].split(";")
        if len(authors) == 2:
            authors_str = f"{authors[0]} and {authors[1]}"
        elif len(authors) > 2:
            authors_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"
        else:
            authors_str = row["authors"]
        
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    
    return results

# Prepare category and tone choices
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Define Gradio UI with an Amazon Books-style design
dashboard = gr.Blocks(theme=gr.themes.Soft())
with dashboard:
    gr.Markdown(
        """
        # 📚 **Discover Your Next Book**  
        Find books based on themes, emotions, and descriptions.  
        """
    )

    with gr.Row(equal_height=True):
        user_query = gr.Textbox(label="🔎 Search Books by Description", placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a 📂 category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an 🎭 emotional tone:", value="All")
        submit_button = gr.Button("Find Recommendations")
    
    gr.Markdown("## 📖 Recommendations")
    output = gr.Gallery(label="Books You Might Like", columns=4, rows=2)

    # Link the button to the recommendation function
    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

if __name__ == "__main__":
    dashboard.launch()