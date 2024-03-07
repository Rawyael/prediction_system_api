import streamlit as st
import pandas as pd
from streamlit_image_select import image_select
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

st.title('SmartShop: Your Go-To for Curated Recommendations')
col1, col2 = st.columns(2)

@st.cache_data
def load_data():
    df_products = pd.read_csv('product_details.csv')
    df_products['Product Details'] = df_products["Category"] + ' ' + df_products["About Product"] + ' ' + df_products["Technical Details"]
    df_products.drop(["Category", 'About Product', 'Technical Details'], axis=1, inplace=True)
    return df_products

def preprocess_text(sentence):
    if isinstance(sentence, str):
        sentence = sentence.strip().lower()
        sentence = ''.join(char for char in sentence if not char.isdigit())
        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, '')
        return sentence
    else:
        return ''

@st.cache_data
def get_similar_products(product_id, df_products):
    df_products['Product Details'] = df_products['Product Details'].apply(preprocess_text)
    tfidf = TfidfVectorizer(stop_words="english", min_df=4)
    tfidf_matrix = tfidf.fit_transform(df_products["Product Details"])
    similarity = cosine_similarity(tfidf_matrix)
    product_index = df_products[df_products['Uniqe Id'] == product_id].index[0]
    similar_products = list(enumerate(similarity[product_index]))
    similar_products_sorted = sorted(similar_products, key=lambda x: x[1], reverse=True)
    similar_product_details = []
    for product in similar_products_sorted[1:6]:
        similar_product_details.append(df_products.iloc[product[0], 1])
    return similar_product_details

df_test = pd.read_csv('products_test.csv')
df_products = load_data()

products = df_test.head(12)
images = products['Image'].to_list()
captions = [x[:15] + '....' for x in products['Product Name'].to_list()]

with col1:
    img = image_select(
        label="Select a product",
        images=images,
        captions=captions,
        return_value="index"
    )

st.session_state['image_id'] = img

if st.session_state['image_id'] is not None:
    with col2:
        similar_products = get_similar_products(df_test.loc[st.session_state['image_id']]['Uniqe Id'], df_products)
        if similar_products:
            similar_product_images = [df_products[df_products['Product Name'] == x]['Image'].values[0] for x in similar_products]
            similar_product_captions = [x[:15] + '....' for x in similar_products]
            recommendations = image_select(
                label="You might also like",
                images=similar_product_images,
                captions=similar_product_captions,
                return_value="index"
            )
        else:
            st.warning("No similar products found.")
