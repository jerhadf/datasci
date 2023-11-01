# The API URL will be of the following format:

# https://api.crunchbase.com/api/v4/entities/organizations/crunchbase?user_key=INSERT_YOUR_API_KEY_HERE


# With Crunchbase Basic, you have access to a limited set of Organization data fields (as covered in Organization Attributes) using the 3 API endpoints below:

# Organization Search endpoint: https://api.crunchbase.com/api/v4/searches/organizations
# Organization Entity Lookup endpoint: https://api.crunchbase.com/api/v4/entities/organizations/{permlaink}
# Autocomplete endpoint: https://api.crunchbase.com/api/v4/autocompletes

# Since I am searching for organizations in LA, I will be using “POST /search/organizations” URL.

# Now let's get to the actual coding part

# Step 2: Request data using python

# Import all necessary Packages

# import requests
# import json
# import pandas as pd
# from pandas.io.json import json_normalize
# We will use the request module to send an API request to Crunchbase.

# Define your API user key

# userkey = {"user_key":"INSERT_YOUR_API_KEY_HERE"}

# EXAMPLE: GET info for tesla motors

# Information to retrieve for Tesla Motors:

# company firmographic - categories, short description, website and facebook URLs (field_ids = website,facebook,categories,short_description)
# who are their founder(s)? (card_ids=founders)
# when was the company founded? (field_ids=founded_on)
# what funding rounds did Tesla Motors raised? (card_ids=raised_funding_rounds)
# what's Tesla's Crunchbase Rank among companies across Crunchbase Graph? (field_ids=rank_org_company)

# GET https://api.crunchbase.com/api/v4/entities/organizations/tesla-motors?card_ids=founders,raised_funding_rounds&field_ids=categories,short_description,rank_org_company,founded_on,website,facebook,created_at&user_key=INSERT_KEY_HERE
