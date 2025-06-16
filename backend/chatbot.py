# chatbot.py

import os
import pandas as pd
from dotenv import load_dotenv

# Updated imports for LangChain V0.2+ and Pydantic V2 compatibility
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field # Updated import for Pydantic V2
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException # Import for parsing errors

# Load Environment Variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise EnvironmentError("⚠️ Hugging Face token not found in .env. Please set HUGGINGFACEHUB_API_TOKEN.")

# Define Pydantic model for structured recommendation output
class RecommendedProduct(BaseModel):
    product_name: str = Field(description="Name of the recommended product.")
    product_category: str = Field(description="Category of the recommended product.")
    product_department: str = Field(description="Department of the recommended product.")
    sale_price: float = Field(description="Price of the recommended product.") # Changed to float for calculations
    stock_quantity: int = Field(description="Available stock quantity of the recommended product.") # Changed to int

class ProductRecommendations(BaseModel):
    recommendations: list[RecommendedProduct] = Field(description="List of recommended products.")

# LLM using HuggingFace (Free)
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    temperature=0.5,
    huggingfacehub_api_token=hf_token,
)

class ChatbotManager:
    def __init__(self, data_file: str):
        print("📄 Loading dataset...")
        try:
            self.df = pd.read_csv(data_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Data file not found at '{data_file}'. Please ensure it exists in the correct directory.")
        except Exception as e:
            raise Exception(f"Error loading dataset from '{data_file}': {e}")

        # --- IMPORTANT: Validate required columns exist ---
        required_columns = ['sale_price', 'stock_quantity', 'order_id', 'name', 'user_id', 
                            'product_department', 'product_name', 'product_category', 'product_brand']
        
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(
                f"Error: The loaded dataset is missing required columns: {', '.join(missing_columns)}. "
                f"Please ensure your CSV file ('{data_file}') has these columns with exact names."
                f"\nAvailable columns: {', '.join(self.df.columns)}"
            )

        # Ensure 'sale_price' is numeric and handle NaNs
        self.df['sale_price'] = pd.to_numeric(self.df['sale_price'], errors='coerce')
        self.df.dropna(subset=['sale_price'], inplace=True) # Drop rows where price couldn't be converted

        # Ensure 'stock_quantity' is numeric and handle NaNs (assuming it might be used numerically)
        self.df['stock_quantity'] = pd.to_numeric(self.df['stock_quantity'], errors='coerce')
        self.df.dropna(subset=['stock_quantity'], inplace=True) # Drop rows where stock couldn't be converted

        self.df['combined_info'] = self.df.apply(lambda row: (
            f"Order ID: {row['order_id']}. "
            f"User: {row['name']} (ID: {row['user_id']}). "
            f"Product Department: {row['product_department']}. "
            f"Product: {row['product_name']}. "
            f"Category: {row['product_category']}. "
            f"Brand: {row['product_brand']}. "
            f"Price: ${row['sale_price']:.2f}. " # Format price for consistency
            f"Stock: {row['stock_quantity']}"
        ), axis=1)

        loader = DataFrameLoader(self.df, page_content_column="combined_info")
        docs = loader.load()

        # Text Splitting
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)

        # Vector DB
        print("📦 Creating vector store...")
        # --- UPDATED: Load embedding model from local path to bypass Hugging Face download issues ---
        # IMPORTANT: Ensure you have manually downloaded all files from
        # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main
        # and placed them into the 'backend/models/all-MiniLM-L6-v2' directory
        embedding_model = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")
        
        self.vectorstore = FAISS.from_documents(texts, embedding_model)
        self.retriever = self.vectorstore.as_retriever()

        # Conversational chain setup
        system_message_template = """You are a helpful shopping assistant. Use the following context and chat history to answer user questions and provide product information. When recommending products in free-form text, briefly include product name, category, and price if available in the context.
If the user asks for specific recommendations based on criteria (like category, brand, price), respond conversationally and explain that you're fetching specific items.
Context: {context}"""

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

        chatbot_prompt = ChatPromptTemplate.from_messages(
            [
                system_message_prompt,
                MessagesPlaceholder(variable_name="chat_history"),
                human_message_prompt,
            ]
        )
        self.document_chain = create_stuff_documents_chain(llm, chatbot_prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)

        # Initialize memory for the conversational chain
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        print("✅ Chatbot initialized.")

    def get_conversational_response(self, query: str):
        """
        Handles general conversational queries using the RAG chain and memory.
        """
        history = self.memory.load_memory_variables({})['chat_history']
        response = self.retrieval_chain.invoke({"question": query, "chat_history": history})
        
        answer_text = response['answer']
        self.memory.save_context({"question": query}, {"answer": answer_text})
        
        return answer_text

    def get_structured_recommendations_from_llm(self, department: str = "any", category: str = "any", brand: str = "any", price_max: float = None):
        """
        Generates structured product recommendations using a dedicated LLM chain
        and Pydantic output parser, grounded by retrieved context.
        """
        parser = PydanticOutputParser(pydantic_object=ProductRecommendations)

        manual_template = """
        You are a product recommendation AI.
        Based on the following criteria, suggest up to three existing products that fit from the provided context.
        For each product, provide its name, category, department, exact sale price, and exact stock quantity.
        Ensure the output is strictly in JSON format matching the schema provided.
        If no relevant products are found for the given criteria in the context, return an empty list in the 'recommendations' field.

        {format_instructions}

        Criteria:
        Product Department: {department}
        Product Category: {category}
        Product Brand: {brand}
        Maximum Price: {price_max}

        Context (relevant products from your database):
        {context}
        """

        # Use the manager's retriever to get relevant products as context for this chain
        search_query = f"products in {department} department, {category} category, brand {brand}"
        if price_max is not None:
            search_query += f", under ${price_max}"
            
        retrieved_docs = self.retriever.get_relevant_documents(search_query)
        context_str = "\n".join([doc.page_content for doc in retrieved_docs])
        if not context_str:
            context_str = "No specific product context found for these criteria. Generate based on general knowledge if possible, or return empty list in JSON."

        prompt_manual = PromptTemplate(
            input_variables=["department", "category", "brand", "price_max", "context"],
            template=manual_template,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        try:
            response_json = (prompt_manual | llm | parser).invoke({
                "department": department,
                "category": category,
                "brand": brand,
                "price_max": price_max if price_max is not None else "any",
                "context": context_str
            })
            
            # Additional filtering/sorting for robustness, as LLM might not perfectly adhere
            if price_max is not None:
                response_json.recommendations = [
                    rec for rec in response_json.recommendations if rec.sale_price <= price_max
                ]
            # Sort by price (lowest first) for consistent output
            response_json.recommendations.sort(key=lambda x: x.sale_price)

            return response_json.recommendations # Return list of RecommendedProduct objects
        except OutputParserException as e:
            print(f"Error parsing manual recommendation output: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during structured recommendation: {e}")
            return []


# --- Global Instance of ChatbotManager ---
# This instance will be imported by app.py
# Using the product transaction data with all expected columns
chatbot_manager = ChatbotManager(data_file='bq-results-20240205-004748-1707094090486.csv')