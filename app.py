import streamlit as st
import pandas as pd
from streamlit_image_select import image_select
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
st.title('SmartShop: Your Go-To for Curated Recommendations')
col1, col2 = st.columns(2)
# Load product details from CSV
df_products = pd.read_csv('product_details.csv')
df_products['Product Details'] = df_products["Category"] + ' ' + df_products["About Product"] + ' ' + df_products["Technical Details"]
df_products.drop(["Category", 'About Product', 'Technical Details'], axis=1, inplace=True)
# Define preprocessing function
def preprocess_text(sentence):
    # Check if the input is a string
    if isinstance(sentence, str):
        # Preprocessing steps
        sentence = sentence.strip()
        sentence = sentence.lower()
        sentence = ''.join(char for char in sentence if not char.isdigit())
        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, '')
        return sentence
    else:
        # If input is not a string (e.g., it's a float), return an empty string
        return ''
# Define function to get similar products
def get_similar_products(product_id, df_products):
    # Preprocess product details
    df_products['Product Details'] = df_products['Product Details'].apply(preprocess_text)
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words="english", min_df=4)
    tfidf_matrix = tfidf.fit_transform(df_products["Product Details"])
    # Compute similarity matrix
    similarity = cosine_similarity(tfidf_matrix)
    # Get index of current product
    product_index = df_products[df_products['Uniqe Id'] == product_id].index[0]
    # Enumerate and sort similar products
    similar_products = list(enumerate(similarity[product_index]))
    similar_products_sorted = sorted(similar_products, key=lambda x: x[1], reverse=True)
    # Retrieve similar product details
    similar_product_details = []
    for product in similar_products_sorted[1:6]:
        similar_product_details.append(df_products.iloc[product[0], 1]),
    return similar_product_details
df_test = pd.read_csv('products_test.csv')
prodcuts = df_test.head(12)
images = prodcuts['Image'].to_list()
captions = prodcuts['Product Name'].to_list()
captions = [x[0:15]+'....' for x in captions]
with col1:
    img = image_select(
        label="Select a product",
        images=images,
        captions=captions,
        return_value="index"
    )
st.session_state['image_id'] = img
if st.session_state['image_id']:
    # st.image(images[st.session_state['image_id']])
    with col2:
        similar_products = get_similar_products(df_test.loc[st.session_state['image_id']]['Uniqe Id'],df_products)
        similar_products = [df_products[df_products['Product Name'] == x]['Image'].values[0] for x in similar_products]
        similar_products_captions = [df_products[df_products['Product Name'] == x]['Product Name'].values[0] for x in similar_products]
        similar_products_captions = [x[0:15]+'....' for x in captions]
        recommendations = image_select(
        label="You might also like",
        images=similar_products,
        captions=similar_products_captions,
        return_value="index"
        )
