import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import pandas as pd
import os
import numpy as np # For numerical operations, especially binning
import re # For simple keyword extraction

# --- Import actual chatbot module ---
# Import the chatbot_manager instance directly, as it encapsulates the QA and structured recs logic
from chatbot import chatbot_manager, RecommendedProduct

# Initialize FastAPI application
app = FastAPI(
    title="Product Recommender & Insight Dashboard",
    description="Your go-to app for AI-driven product recommendations, conversational assistance, and data insights.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Configure Jinja2Templates for serving HTML files
templates = Jinja2Templates(directory="templates")

# --- Load Dataset for Dashboard Visualizations ---
# This DataFrame (df) is specifically for the dashboard/visualization endpoints.
# The chatbot_manager loads its own data (which you configured to be the synthetic data).
csv_filename = "bq-results-20240205-004748-1707094090486.csv" # Original data for dashboard
# Or, if you want the dashboard to also use the synthetic data:
# csv_filename = "synthetic_transaction_data_100000.csv" 
csv_path = os.path.join(os.path.dirname(__file__), csv_filename)

df = pd.DataFrame() # Initialize df as an empty DataFrame
try:
    df = pd.read_csv(csv_path)
    # Ensure 'sale_price' and 'stock_quantity' are numeric
    df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
    df['stock_quantity'] = pd.to_numeric(df['stock_quantity'], errors='coerce')
    # Convert product-related text columns to string types explicitly during load
    df['product_category'] = df['product_category'].astype(str)
    df['product_department'] = df['product_department'].astype(str)
    df['product_brand'] = df['product_brand'].astype(str)
    df['product_name'] = df['product_name'].astype(str) # Crucial for product_name labels
    # Drop rows where essential numeric or text data is missing for safety
    df.dropna(subset=['sale_price', 'stock_quantity', 'product_category', 'product_department', 'product_brand', 'product_name'], inplace=True)
    print(f"Dashboard dataset '{csv_filename}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Dashboard dataset file not found at '{csv_path}'. Please ensure it exists.")
except Exception as e:
    print(f"Error loading dashboard dataset: {e}")


# --- Helper function for basic intent detection and parameter extraction ---
def extract_recommendation_params(query: str):
    """
    Analyzes a user query to extract parameters for product recommendations.
    This is a simple rule-based approach; a more advanced system would use NLP.
    """
    department = None
    category = None
    brand = None
    price_max = None

    query_lower = query.lower()

    # Department detection
    if "men's" in query_lower or "mens" in query_lower or "male" in query_lower:
        department = "Men"
    elif "women's" in query_lower or "womens" in query_lower or "female" in query_lower:
        department = "Women"
    elif "kid's" in query_lower or "kids" in query_lower or "children" in query_lower:
        department = "Kids"

    # Category detection (simple keyword matching, extend as needed)
    # Ensure these match the capitalization or exact strings in your product_category data
    categories_keywords = {
        "shirts": "Shirts", "jeans": "Jeans", "dresses": "Dresses", "shoes": "Shoes",
        "watches": "Watches", "t-shirts": "T-Shirts", "shorts": "Shorts",
        "outerwear": "Outerwear & Coats", "intimates": "Intimates", "maternity": "Maternity",
        "socks": "Socks", "jewelry": "Jewelry", "blazers": "Blazers & Jackets",
        "leggings": "Leggings", "hoodies": "Fashion Hoodies & Sweatshirts",
        "sweaters": "Sweaters", "skirts": "Skirts", "swimwear": "Swim",
        "pants": "Pants", "bags": "Handbags", "sunglasses": "Sunglasses",
        "activewear": "Activewear", "sleepwear": "Sleepwear", "accessories": "Accessories"
    }
    for keyword, cat_name in categories_keywords.items():
        if keyword in query_lower:
            category = cat_name
            break

    # Brand detection (simple keyword matching, extend with a more comprehensive list if available)
    # Ideally, you'd load unique brands from your actual product data.
    known_brands_lower = [b.lower() for b in chatbot_manager.df['product_brand'].unique() if pd.notna(b)]
    for b in known_brands_lower:
        if b in query_lower:
            brand = b.title() # Capitalize first letter to match common brand naming
            break

    # Price detection (using regex)
    # Looks for phrases like "under $100", "max price 50", "$250 or less"
    price_match = re.search(r'(under|max|less than|up to)\s*\$?(\d+(\.\d{1,2})?)', query_lower)
    if price_match:
        try:
            price_str = price_match.group(2) # Group 2 captures the number
            price_max = float(price_str)
        except (ValueError, TypeError):
            price_max = None # Handle cases where conversion fails

    return department, category, brand, price_max

# --- FastAPI Routes ---

@app.get("/", response_class=HTMLResponse, summary="Product Recommender Dashboard")
async def index(request: Request):
    """
    Serves the main dashboard page for product recommendations and access to other features.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/visualization", response_class=HTMLResponse, summary="Data Visualization Page")
async def visualization(request: Request):
    """
    Serves the HTML page for Chart.js visualizations.
    Data is fetched by the client-side JavaScript from the /data endpoint.
    """
    if df.empty:
        raise HTTPException(status_code=503, detail="Data for visualizations could not be loaded. Please check the dataset file.")
    return templates.TemplateResponse("visualization.html", {"request": request})

@app.get("/chatbot-interface", response_class=HTMLResponse, summary="Chatbot Interface Page")
async def chatbot_interface(request: Request):
    """
    Serves the dedicated HTML page for interacting with the AI chatbot.
    """
    return templates.TemplateResponse("chatbot_interface.html", {"request": request})


# 🧠 Manual Recommendation Input Model
class ManualItem(BaseModel):
    """
    Pydantic model for manual recommendation input.
    Note: 'price' is now float to match the backend function.
    """
    department: str = Field(..., description="Department, e.g., 'Men', 'Women', 'Kids'")
    category: str = Field(..., description="Product category, e.g., 'Dresses', 'Jeans'")
    brand: str = Field(..., description="Product brand, e.g., 'Nike', 'Adidas'")
    price_max: float | None = Field(None, description="Maximum price for the recommendation (optional)")


# POST: Get Manual Product Recommendation (now returns structured JSON)
@app.post("/manual", summary="Get Manual Product Recommendation", response_model=dict)
async def manual(item: ManualItem):
    """
    Provides a manual product recommendation based on specified item attributes.
    This endpoint calls the structured recommendation function from the chatbot_manager.
    """
    try:
        recommendations_list = chatbot_manager.get_structured_recommendations_from_llm(
            department=item.department,
            category=item.category,
            brand=item.brand,
            price_max=item.price_max
        )
        
        # Convert Pydantic objects to dictionaries for JSON serialization
        recommendations_data = [rec.model_dump() for rec in recommendations_list]
        return {"recommendations": recommendations_data}

    except Exception as e:
        print(f"Error during manual recommendation: {e}")
        # Return a custom error message for the frontend
        raise HTTPException(status_code=500, detail=f"Failed to get manual recommendation: {str(e)}")


# 💬 Conversational Chatbot Endpoint
class ChatQuery(BaseModel):
    """
    Pydantic model for chatbot query input.
    """
    query: str

@app.post("/chatbot", summary="Get Chatbot Answer", response_model=dict)
async def get_answer(query: ChatQuery):
    """
    Engages with the AI chatbot to provide both conversational answers and
    structured product recommendations based on detected user intent.
    """
    user_query = query.query.strip()
    if not user_query:
        return {"response": "Please enter a message.", "products": []}

    # --- Intent Detection and Parameter Extraction ---
    # Keywords to trigger structured recommendation attempt
    product_recommendation_keywords = ["recommend", "show me", "looking for", "find me", "suggest", "products", "items", "buy", "need", "laptop", "shoe", "dress", "shirt", "jeans", "watch"]
    
    # Check if the query contains strong indicators for structured product recommendations
    is_explicit_product_request = any(keyword in user_query.lower() for keyword in product_recommendation_keywords)

    if is_explicit_product_request:
        department, category, brand, price_max = extract_recommendation_params(user_query)
        
        # Call the structured recommendation chain
        recommended_products_list = chatbot_manager.get_structured_recommendations_from_llm(
            department=department,
            category=category,
            brand=brand,
            price_max=price_max
        )

        if recommended_products_list:
            # Convert Pydantic objects to dictionaries for JSON serialization
            products_data = [product.model_dump() for product in recommended_products_list]
            response_text = "Here are some products that might interest you:"
            # The frontend will display the product cards from `products_data`
            return {"response": response_text, "products": products_data}
        else:
            # If no structured recommendations found for specific criteria,
            # try a conversational fallback, or indicate no products found.
            # For now, let's keep it direct.
            response_text = "I couldn't find specific products matching your detailed criteria. Can I help with something else?"
            return {"response": response_text, "products": []}
    else:
        # Fallback to general conversational chain
        bot_response_text = chatbot_manager.get_conversational_response(user_query)
        return {"response": bot_response_text, "products": []}


# 📈 New: Endpoint for Comprehensive Dashboard Data with Filters
class DashboardFilter(BaseModel):
    department: str | None = None
    category: str | None = None
    brand: str | None = None
    min_price: float | None = None
    max_price: float | None = None

@app.post("/dashboard_data_filtered", summary="Get Comprehensive Dashboard Data with Filters")
async def get_dashboard_data_filtered(filters: DashboardFilter):
    """
    Provides comprehensive filtered data for all charts on the professional dashboard.
    """
    if df.empty:
        raise HTTPException(status_code=503, detail="Data not available. Failed to load dataset.")

    filtered_df = df.copy()

    # Apply text filters (case-insensitive contains) if value is provided and not empty
    if filters.department and filters.department.strip():
        filtered_df = filtered_df[filtered_df['product_department'].str.contains(filters.department.strip(), case=False, na=False)]
    if filters.category and filters.category.strip():
        filtered_df = filtered_df[filtered_df['product_category'].str.contains(filters.category.strip(), case=False, na=False)]
    if filters.brand and filters.brand.strip():
        filtered_df = filtered_df[filtered_df['product_brand'].str.contains(filters.brand.strip(), case=False, na=False)]

    # Apply numerical price filters if values are provided
    if filters.min_price is not None:
        # Ensure we only filter on valid numbers
        filtered_df = filtered_df[filtered_df['sale_price'].notna()]
        filtered_df = filtered_df[filtered_df['sale_price'] >= filters.min_price]
    if filters.max_price is not None:
        # Ensure we only filter on valid numbers
        filtered_df = filtered_df[filtered_df['sale_price'].notna()]
        filtered_df = filtered_df[filtered_df['sale_price'] <= filters.max_price]

    if filtered_df.empty:
        # Return 404 if no data matches the filters
        raise HTTPException(status_code=404, detail="No data found for the selected filters. Please adjust your criteria.")

    # --- Chart Data Generation ---

    # 1. Top 10 Product Categories - product_category is already str from initial load
    category_data = filtered_df["product_category"].value_counts().nlargest(10)

    # 2. Product Department Distribution - product_department is already str from initial load
    department_data = filtered_df["product_department"].value_counts()

    # 3. Top 10 Product Brands - product_brand is already str from initial load
    brand_data = filtered_df["product_brand"].value_counts().nlargest(10)

    # 4. Product Price Distribution (Bins)
    # Define price bins and labels.
    price_bins = [0, 50, 100, 250, 500, 1000, 2000, 5000, np.inf] # 9 bins
    price_labels = ['< $50', '$50-100', '$101-250', '$251-500', '$501-1k', '$1k-2k', '$2k-5k', '$5k+'] # 8 labels
    
    # Filter out NaNs from sale_price before binning
    price_ranges = pd.cut(filtered_df['sale_price'].dropna(), bins=price_bins, labels=price_labels, right=False, include_lowest=True)
    price_distribution = price_ranges.value_counts().sort_index()
    # Reindex to ensure all labels are present, even if count is 0
    price_distribution = price_distribution.reindex(price_labels, fill_value=0)


    # 5. Top 10 Products by Sale Price (highest priced)
    # Ensure there are enough products, handle potential NaNs in sale_price before sorting
    top_products_by_price = filtered_df.dropna(subset=['sale_price']).sort_values(by='sale_price', ascending=False).head(10)
    # Corrected: product_name is already str from initial load, no need for .astype(str) here.
    top_products_by_price_labels = [f"{row['product_name']} (${row['sale_price']:.2f})" for index, row in top_products_by_price.iterrows()]
    top_products_by_price_values = top_products_by_price['sale_price'].tolist()

    # 6. Product Stock Quantity (Top 10 products by stock, excluding 0 stock)
    # Filter out 0 stock items if they distort the view of available stock
    # Ensure no NaNs in stock_quantity before filtering and sorting
    active_stock_df = filtered_df[filtered_df['stock_quantity'].notna() & (filtered_df['stock_quantity'] > 0)]
    top_stock_products = active_stock_df.sort_values(by='stock_quantity', ascending=False).head(10)
    # Corrected: product_name is already str from initial load, no need for .astype(str) here.
    stock_labels = top_stock_products['product_name'].tolist()
    stock_values = top_stock_products['stock_quantity'].tolist()


    return {
        "category_labels": category_data.index.tolist(),
        "category_values": category_data.values.tolist(),
        "department_labels": department_data.index.tolist(),
        "department_values": department_data.values.tolist(),
        "brand_labels": brand_data.index.tolist(),
        "brand_values": brand_data.values.tolist(),
        "price_range_labels": price_distribution.index.tolist(),
        "price_range_values": price_distribution.values.tolist(),
        "top_products_by_price_labels": top_products_by_price_labels,
        "top_products_by_price_values": top_products_by_price_values,
        "stock_labels": stock_labels,
        "stock_values": stock_values,
    }


# 📈 Existing: Endpoint for Chart Data based on Recommendation Filters (used by index.html)
# This remains separate as its purpose is more specific to the recommendation result.
class RecommendationInsightFilter(BaseModel):
    category: str | None = None
    brand: str | None = None
    department: str | None = None

@app.post("/product_insight_chart_data", summary="Get Chart Data based on Recommendation Filters (for index page)")
async def get_product_insight_chart_data(filters: RecommendationInsightFilter):
    """
    Provides filtered product data for visualizations based on attributes from a recommendation.
    Returns counts of product names within the specified category/brand/department.
    """
    if df.empty:
        raise HTTPException(status_code=503, detail="Data not available for filtering. Failed to load dataset.")

    filtered_df = df.copy()

    if filters.category and filters.category.strip():
        filtered_df = filtered_df[filtered_df['product_category'].str.contains(filters.category.strip(), case=False, na=False)]
    if filters.brand and filters.brand.strip():
        filtered_df = filtered_df[filtered_df['product_brand'].str.contains(filters.brand.strip(), case=False, na=False)]
    if filters.department and filters.department.strip():
        filtered_df = filtered_df[filtered_df['product_department'].str.contains(filters.department.strip(), case=False, na=False)]

    if filtered_df.empty:
        return {
            "chart_type": "bar",
            "labels": [],
            "values": [],
            "title": "No data for selected filters"
        }

    # product_name is already string type from initial load
    product_name_counts = filtered_df['product_name'].value_counts().nlargest(10)
    chart_title = "Top Products in Recommended Context"
    if filters.category and filters.brand:
        chart_title = f"Top Products in {filters.category} by {filters.brand}"
    elif filters.category:
        chart_title = f"Top Products in {filters.category}"
    elif filters.brand:
        chart_title = f"Top Products by {filters.brand}"
    elif filters.department:
        chart_title = f"Top Products in {filters.department} Department"


    return {
        "chart_type": "bar",
        "labels": product_name_counts.index.tolist(),
        "values": product_name_counts.values.tolist(),
        "title": chart_title
    }


# --- Run the FastAPI app ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)