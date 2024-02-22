#!/usr/bin/env python
# coding: utf-8

# # Ecommerce Dataset Analysis: Unveiling Insights for Strategic Decision-Making
# 

# ## Introduction:
#  This report presents an analysis of the Amazon Sales Dataset. The dataset contains information about various products, including their attributes, customer reviews, and sales data. The analysis aims to uncover patterns, trends, and relationships within the data to inform strategic decision-making and improve overall business performance.

# ## Dataset Overview:
# The dataset comprises several columns, including:
# 
# - Product attributes such as product ID, name, category, prices, discount percentage, and rating.
# - Customer reviews, including user ID, review ID, review title, review content, and user name.
# - Additional information, such as about the product, image links, and product links.

# ## Data Preprocessing
# 

# In[3]:


import pandas as pd
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a pandas DataFrame
ecommerce_data = pd.read_csv('amazon.csv')
pd.set_option('display.max_colwidth', 80)
ecommerce_data.head()


# ## Cleaning up Data
# To clean the dataset, I performed several preprocessing steps. These include:
# 
# - Converting price-related columns to numeric data types.
# - Removing special characters and converting text to lowercase for text-based - analysis.
# - Handling missing values.
# - Extracting main category information from the category column for simplified analysis
# - Tokenizing and removing stop words for text analysis.

# In[6]:


# 1. checking and handling missing values
sd = ecommerce_data.dropna()
# 2. Removing duplicates 
sd = sd.drop_duplicates()
# 3. Formatting some column with the right data type after careful evaluation of it contents since all information in the dataset are formatted as "object".

# Remove currency symbols and commas (',') from 'discounted_price' column and convert to float
sd['discounted_price'] = sd['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)

# Remove currency symbols and commas (',') from 'actual_price' column and convert to float 
sd['actual_price'] = sd['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)

# Remove '%' from 'discount_percentage' and convert to float
sd['discount_percentage'] = sd['discount_percentage'].astype(str).str.replace('%', '').astype(float)

# Convert 'rating' to float and handle error 
sd['rating'] = pd.to_numeric(sd['rating'].astype(str).str.replace('|', '', regex=True), errors='coerce')

# Remove commas (',') from 'rating_count' and convert to int (count can't be in float)
sd['rating_count'] = sd['rating_count'].astype(str).str.replace(',', '').astype(int)


# In[7]:


# 4. Extracting main categories from the 'category' column
sd['main_category'] = sd['category'].str.replace(r'\|.*', '', regex=True)

# Display the DataFrame information
print(sd.info())


# In[8]:


# 5. Cleaning and preprocessing text without Stemming or lemmatization
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # Converting to lowercase
    text = text.lower()
    # Removing punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    # Splitting text into words and removing stopwords
    words_without_stopwords = [word for word in text.split() if word not in stop_words]
    # Joining words with a single space between each word
    text_without_stopwords = ' '.join(words_without_stopwords)
    return text_without_stopwords



# Applying clean_text to specified columns of the dataframe
sd['category'] = sd['category'].apply(clean_text)
sd['review_title'] = sd['review_title'].apply(clean_text)
sd['review_content'] = sd['review_content'].apply(clean_text)


# In[9]:


# Display the updated DataFrame and handle missing values and duplicate
sd = sd.dropna()
sd = sd.drop_duplicates()
sd


# ## EDA
# 
# ## Product Insights:
# 
# ### Which products have the highest and lowest ratings?

# In[10]:


# Which products have the highest and lowest ratings?
highest_rated_product = sd.loc[sd['rating'].idxmax()][['product_name','main_category', 'discounted_price','actual_price', 'discount_percentage', 'rating']]
lowest_rated_product = sd.loc[sd['rating'].idxmin()][['product_name','main_category', 'discounted_price','actual_price', 'discount_percentage', 'rating']]

rating_comparison_table = pd.concat([highest_rated_product, lowest_rated_product], axis=1)
rating_comparison_table.columns = ['Highest Rated Product', 'Lowest Rated Product']

rating_comparison_table


# In[11]:


# Is there a correlation between discount percentage and rating?
# Calculate the correlation between the two columns
correlation = sd['discount_percentage'].corr(sd['rating'])
print("Correlation between discount percentage and rating:", correlation)


plt.figure(figsize=(8, 6))
plt.scatter(sd['discount_percentage'], sd['rating'])
plt.title('Correlation between discount percentage and rating')
plt.xlabel('discount percentage')
plt.ylabel('rating')
plt.grid(True)
plt.show()


# ### How do rating counts vary across different product categories?
# 

# In[12]:


# How do rating counts vary across different product categories?
rating_counts_by_category = sd.groupby('main_category')['rating_count'].mean()
rating_counts_by_category_df = rating_counts_by_category.reset_index(name='Average Rating Count')

# Print the DataFrame
print("\nAverage rating counts by category:")
rating_counts_by_category_df


# ### What are the most common words used in product descriptions (about_product)?
# 
# 

# In[13]:


# What are the most common words used in product descriptions (about_product)?
from collections import Counter
import re

words = ' '.join(sd['about_product']).lower()
words = re.findall(r'\b\w+\b', words)
word_counts = Counter(words).most_common(10)
print("Most common words in product descriptions:")
word_counts


# ## Customer Behavior Analysis:
# 
# 
# ### How many unique users have made purchases?

# In[14]:


# How many unique users have made purchases?
unique_users = sd['user_id'].nunique()
print("Number of unique users:", unique_users)


# ### What is the average number of products purchased per user?
# 

# In[15]:


# What is the average number of products purchased per user?
average_products_per_user = sd.groupby('user_id')['product_id'].count().mean()
print("Average number of products purchased per user:", average_products_per_user)


# ### How do user ratings correlate with the number of reviews they provide?

# In[17]:


# How do user ratings correlate with the number of reviews they provide?
correlation_reviews_ratings = sd['rating'].corr(sd['rating_count'])
print("\nCorrelation between ratings and number of reviews:", correlation_reviews_ratings)


# ## Marketing and Promotion Analysis:
# 
# 
# ### Which products have the highest discount percentages?

# In[18]:


# Which products have the highest discount percentages?
highest_discount_products = sd.loc[sd['discount_percentage'].idxmax()]
highest_discount_products_df = pd.DataFrame(highest_discount_products[['product_name', 'discount_percentage']]).transpose()
print("Product with the highest discount percentage:")
highest_discount_products_df


# ## User Engagement and Experience:
# 
# ### What are the most common review titles and review contents?

# In[19]:


# What are the most common review titles?
common_review_titles = sd['review_title'].value_counts().head()
common_review_titles_df = common_review_titles.reset_index()
common_review_titles_df.columns = ['Review Title', 'Frequency']
print("\nMost common review titles:")
common_review_titles_df


# In[20]:


# What are the most common review contents?
common_review_contents = sd['review_content'].value_counts().head()
common_review_contents_df = common_review_contents.reset_index()
common_review_contents_df.columns = ['Review Contents', 'Frequency']
print("\nMost common review contents:")
common_review_contents_df


# ### How do user ratings correlate with the length of review content?
# 

# In[21]:


# How do user ratings correlate with the length of review content?
sd['review_content_length'] = sd['review_content'].apply(lambda x: len(str(x)))
correlation_rating_content_length = sd['rating'].corr(sd['review_content_length'])
print("Correlation between ratings and length of review content:", correlation_rating_content_length)


# ## Visual Content Analysis:
# 
# 
# ### Which products have the most frequently accessed image links (img_link)?

# In[22]:


# Which products have the most frequently accessed image links (img_link)?
most_frequent_images = ecommerce_data['img_link'].value_counts().head()
most_frequent_images_df = most_frequent_images.reset_index()
most_frequent_images_df.columns = ['Image Link', 'Frequency']
print("\nProducts with the most frequently accessed image links:")
most_frequent_images_df


# ## Analysis Findings:
# 
# ### 1. Product Insights:
# 
# Identified products with the highest and lowest ratings.
# Examined the correlation between discount percentage and rating, finding a weak negative correlation (-0.155).
# Explored the variation of rating counts across different product categories.
# 
# ### 2. Customer Behavior Analysis:
# 
# Determined the number of unique users who made purchases and calculated the average number of products purchased per user.
# Analyzed user behavior patterns based on the time of day or day of the week **but the dataset doesn't contain date information**.
# Investigated the correlation between user ratings and the number of reviews provided, finding a weak positive correlation (0.102).
# 
# ### 3. Marketing and Promotion Analysis:
# 
# Evaluated the impact of discounts on sales volume **dataset does not contain number of sales/sales volume information**.
# Identified products with the highest discount percentages.
# Explored the correlation between user ratings and the length of review content, finding a weak positive correlation (0.077).
# 
# ### 4. Visual Content Analysis:
# 
# Determined products with the most frequently accessed image links.

# ## Recommendations
# 
# Based on the findings above, here are some recommendations:
# 
# ### Product Performance Enhancement: 
# 
# - Identify the factors contributing to the high ratings of products like "Syncwire LTG to USB Cable for Fast Charging" and replicate those features in other products.
# 
# - Address the issues highlighted in products with low ratings, such as "Khaitan ORFin Fan heater for Home and kitchen," to improve customer satisfaction.
# 
# #### Discount Strategy Optimization: 
# 
# - Analyze the impact of discount percentages on sales volume to determine the effectiveness of discounting strategies.
# 
# - Explore offering discounts on products with lower ratings or lower sales volume to stimulate demand.
# 
# ### Customer Engagement and Review Management: 
# 
# - Encourage customers to provide detailed reviews by offering incentives or rewards, as there is a positive correlation between user ratings and the number of reviews.
# 
# - Monitor and respond to customer reviews promptly to address any concerns or issues raised by customers.
# 
# ### Marketing and Promotion Tactics: 
# 
# - Highlight products with the highest discount percentages in marketing campaigns to attract price-sensitive customers.
# 
# - Utilize visual content analysis insights to prioritize the creation and promotion of product images that are accessed most frequently by customers.
# 
# ### Product Description Optimization: 
# 
# - Optimize product descriptions by focusing on the most common words used by customers to improve searchability and relevance.
# 
# - Ensure that product descriptions highlight key features and benefits effectively to enhance customer understanding and decision-making.

# ## Conclusion
# 
# In conclusion, this analysis provides valuable insights into various aspects of the ecommerce business, including product performance, customer behavior, marketing effectiveness. By implementing the recommended strategies, the company can enhance its offerings, improve customer satisfaction, and drive business growth in a competitive market environment. Regular monitoring and adaptation of strategies based on evolving market dynamics will be crucial for sustained success.
