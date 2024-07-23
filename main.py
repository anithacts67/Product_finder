import json
import gcsfs
import pickle
import re
import pandas as pd
import numpy as np
import datetime
import db_dtypes
from google.cloud import bigquery
from google.cloud import storage
from vertexai.language_models import TextEmbeddingModel
from vertexai.language_models import TextEmbeddingInput
from pyarrow import parquet
from io import BytesIO
import rapidfuzz

# attempt to cache data outside of the cloud functions entry point
try:
    storage_client = storage.Client(project = 'data-genai-prd')
    bucket = storage_client.bucket('store_product_finder')

    file_path = 'update_for_v4/' ##### DELETE THIS TO GO BACK TO OLD EMBEDDINGS

    # product hierarchy
    df_h = pd.read_parquet(BytesIO(bucket.blob(file_path + 'product_hierarchy.parquet').download_as_bytes()))

    # sales
    df_sales = pd.read_parquet(BytesIO(bucket.blob(file_path + 'sales.parquet').download_as_bytes()))

    # product embeddings
    df_emb = pd.read_parquet(BytesIO(bucket.blob(file_path + 'product_embeddings.parquet').download_as_bytes()))
    
    df_search_scores = pd.read_parquet(BytesIO(bucket.blob(file_path + 'online_search_scores.parquet').download_as_bytes()))

except Exception as e99:
    print('Trouble loading in cached data.')
    print(e99)


def check_the_past(store, product):
    
    # by default assume we have never seen the request before
    response = (False, '')

    try:
        # have we asked the request before?
        client = bigquery.Client(project = 'data-genai-prd')
        query = "SELECT * FROM `data-genai-prd.product_finder_app.search_request_history` WHERE 1=1 AND upload_datetime > '2024-02-29 10:10:00.000000'"
        df_hist = client.query(query).to_dataframe()
        df_hist = df_hist.astype({'store': 'str', 'product': 'str', 'user': 'str', 'response': 'str'})
        df_hist = df_hist.loc[(df_hist.store == str(store)) & (df_hist['product'] == str(product))].sort_values('upload_datetime', ascending = False)

        if df_hist.shape[0] > 0:
            print("""We've seen this product before!""")
            response = (True, df_hist.response.values[0])
        else:
            print("""We've not had this request before, continue as normal.""")

    except Exception as e:
        print("Check history - Failed to check history table.")
   
    return response


def upload_response(store, product, user, text_response):
    
    try:
        # package details of the webhook response
        df_upload = pd.DataFrame({'store': [str(store)], 'product': [str(product)], 'user': [str(user)], 'response': [str(text_response)], 'upload_datetime': [datetime.datetime.now()]})

        # upload response in a GCP table
        client = bigquery.Client(project = 'data-genai-prd')
        dataset_id = 'data-genai-prd.product_finder_app'
        dataset_ref = bigquery.dataset.Dataset(dataset_id)
        table_ref = dataset_ref.table("search_request_history")

        job_config = bigquery.LoadJobConfig(

            schema = [
                bigquery.SchemaField("store", bigquery.enums.SqlTypeNames.STRING),
                bigquery.SchemaField("product", bigquery.enums.SqlTypeNames.STRING),
                bigquery.SchemaField("user", bigquery.enums.SqlTypeNames.STRING),
                bigquery.SchemaField("response", bigquery.enums.SqlTypeNames.STRING),
                bigquery.SchemaField("upload_datetime", bigquery.enums.SqlTypeNames.TIMESTAMP),
            ],
        )

        job = client.load_table_from_dataframe(df_upload, table_ref, job_config = job_config)

    except Exception as e:
        print("GCP Upload - Failed to upload to GCP.")

    return


def fuzz_off(user_str, prod_str): # change vegetarian to vegan
    
    if user_str == prod_str:
        return 110
    
    user_str = user_str.replace('vegetarian','').replace('vegan','').replace('plant based','').replace('meat free','') # these will get a seperate boost
    
    if len(user_str) == 0:
        return 0

    user_tokens = user_str.split(' ')
    prod_tokens = prod_str.split(' ')
    # print(tokens)
    scores = []
    for user_token in user_tokens:
        token_results = []
        for prod_token in prod_tokens:
            if user_token == prod_token:
                token_score = 100 
            else:
                token_score = rapidfuzz.fuzz.ratio(user_token, prod_token) 
            token_results.append(token_score)
            # print(user_token, prod_token, token_score)
        partial_score = rapidfuzz.fuzz.partial_ratio(user_token, prod_str) 
        final_token_score = (partial_score + max(token_results))/2 
        # print(user_token, max(token_results), partial_score, final_token_score)
        
        scores.append(final_token_score * len(user_token))
    
    return sum(scores)/len(user_str.replace(' ', ''))


def help_me_find(user_input, store, store_name, vol_w=2, acc_w=1, type_w=0.5, online_w=1):

    try:
        # aisle locations
        storage_client = storage.Client(project = 'data-genai-prd')
        bucket = storage_client.bucket('store_product_finder')
        df_loc = pd.read_parquet(BytesIO(bucket.blob(f'aisle_locations_live/aisle_locations_{store}.parquet').download_as_bytes()))
        
    except Exception as e1:
        print('Embedding function - Failed reading in supporting data and embeddings.')
        print(e1)

    # create an embedding for the user's input
    try:
        # declare model and pull inout string embedding
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@002")

        text_emb_input = TextEmbeddingInput(task_type = 'RETRIEVAL_QUERY', text = user_input)
        user_emb = np.array(model.get_embeddings([text_emb_input])[0].values).reshape(1, -1)

        # task_type	Description
        # RETRIEVAL_QUERY	Specifies the given text is a query in a search/retrieval setting.
        # RETRISEVAL_DOCUMENT	Specifies the given text is a document in a search/retrieval setting.
        # SEMANTIC_SIMILARITY	Specifies the given text will be used for Semantic Textual Similarity (STS).
        # CLASSIFICATION	Specifies that the embeddings will be used for classification.
        # CLUSTERING	Specifies that the embeddings will be used for clustering.

    except Exception as e2:
        print("Embedding function - Failed creating embedding from user input.")
        print(e2)

    try:
        # create an accuracy store
        df_final = df_loc.sort_values('aisle', ascending = True).groupby(['bsns_unit_cd', 'sku_item_nbr'], as_index = False).first()

        # which products most relate to our input string?
        accuracy = dict((zip(df_emb.index, ((df_emb.to_numpy() - user_emb) ** 2).sum(axis = 1))))

        df_final['broad_product_type_match_score'] = 1
        df_final['broad_product_type_match'] = None

        # calculate adjusted accuracy score for each product
        user_input_clean = re.sub(r'[^\w\s]', '', user_input).lower()
        user_input_clean_words = user_input_clean.split(' ')

        for c in ['description','trading_group','category','department','class','subclass']:
            df_final['a_' + c] = df_final[c].str.lower().map(accuracy)
            df_final['m_string_match_' + c] = df_final[c].str.lower().str.contains(user_input_clean)
            # df_final['m_' + c] = df_final[c].apply(lambda x: fuzz_off(user_input_clean, re.sub(r'[^\w\s]', '', x).lower()))
        df_final['m_description'] = df_final['description'].apply(lambda x: fuzz_off(user_input_clean, re.sub(r'[^\w\s]', '', x).lower()))

        for c in ['category','department','class', 'subclass']:
            df_final['broad_product_type_match'] = np.where(df_final['a_' + c] < df_final['broad_product_type_match_score'], 
                                                                  c, df_final['broad_product_type_match'])
            df_final['broad_product_type_match_score'] = np.where(df_final['a_' + c]  < df_final['broad_product_type_match_score'], 
                                                                  df_final['a_' + c], df_final['broad_product_type_match_score'])



        # display(df_final[df_final.m_description > 70])

        # df_final['accuracy'] = 0.75*df_final.a_description + 0.125*df_final.a_department + 0.125*df_final.a_class
        df_final['accuracy'] = 0.75*df_final.a_description + 0.25*df_final.broad_product_type_match_score
        df_final['accuracy_adjusted'] = df_final.accuracy

        df_final['cat_boost'] = np.where((df_final.broad_product_type_match_score == 0), 0.2, 0)
        df_final['str_boost'] = np.where(df_final.m_description>70, 0.2*((df_final.m_description-70)/30), 0)
        df_final['boost'] = df_final[['cat_boost', 'str_boost']].max(axis=1)
        df_final['accuracy_adjusted'] = df_final.accuracy - df_final.boost

        # df_final['accuracy_adjusted'] = np.where(df_final.m_description>70, df_final['accuracy_adjusted'] - (0.2*((df_final.m_description-60)/40)), df_final['accuracy_adjusted'])
        # df_final['boost'] = np.where(df_final.m_description>70, (0.2*((df_final.m_description-60)/40)), 0)

        df_final['accuracy_adjusted'] = np.where(df_final.description.str.contains('morrisons', case = False), df_final.accuracy_adjusted-0.02, df_final.accuracy_adjusted)
        # df_final['accuracy_adjusted'] = np.where(df_final.trading_group.str.contains('deli', case = False), df_final.accuracy_adjusted+0.05, df_final.accuracy_adjusted)
        
        
        ### add boost to accuracy for online click-throughs
        
        # client = bigquery.Client(project = 'data-genai-prd')
        # df_search_term_boost = client.query(f"""SELECT * FROM `data-genai-prd.product_finder_app.online_search_scores` WHERE search_term_clean = '{user_input_clean}'""").to_dataframe()
        
        df_search_term_boost = df_search_scores[df_search_scores.search_term_clean == user_input_clean]
        print(f'website searched items: {df_search_term_boost.shape[0]}')
        df_final = pd.merge(df_final, df_search_term_boost[['sku_item_nbr', 'search_term_score']], on='sku_item_nbr', how='left')
        df_final.search_term_score = df_final.search_term_score.fillna(0) * 0.4
        df_final.accuracy_adjusted = df_final.accuracy_adjusted - df_final.search_term_score
        
         # add frozen filter
        if 'frozen' in user_input_clean:
            df_final = df_final.loc[df_final.trading_group == 'Frozen']

        # reduce accuracy of canned/frozen items if user search contains 'fresh'
        if 'fresh' in user_input_clean_words:
            df_final.accuracy_adjusted = np.where(ddf_final.trading_group.isin(['Frozen', 'Canned & Packet']), df_final.accuracy_adjusted + 0.1, df_final.accuracy_adjusted)

        # reduce accuracy of pet items unless search includes pet term
        pet_boost = 0.1
        for pet_word in ['pet', 'pets', 'dog', 'cat', 'dogs', 'cats']:
            if pet_word in user_input_clean_words:
                pet_boost = 0
        df_final.accuracy_adjusted = np.where(df_final.category == 'Pet (c)', df_final.accuracy_adjusted + pet_boost, df_final.accuracy_adjusted)

        # bake beean boost
        if user_input_clean == 'beans':
            df_final.accuracy_adjusted = np.where(df_final.subclass == 'Baked Beans', df_final.accuracy_adjusted - 0.1, df_final.accuracy_adjusted)

        vegetarian_class = ['Alternative Milks', 'Meat Free', 'Frozen Meat Free', 'Alternative Long Life Milk', 'Dairy Free Spread', 
                            'Vegetarian Fresh  Meal Solutions',
                            'Convenience Meals - Meat Free',
                            'Cooking Aids - Meat Free',
                            'Homebaking - Meat Free',
                            'International Foods - Meat Free',
                            'Vegetarian']

        vegetarian_subclass = ['Meat Free', 'Frozen - Meat Free', 'Convenience Meals - Meat Free',
                                   'Cooking Aids - Meat Free', 'Homebaking - Meat Free',
                                   'International Foods - Meat Free', 'Vegetarian Branded', 'Vegetarian OL', 'Vegetarian']

        vegan_class = ['Alternative Milks', 'Meat Free', 'Frozen Meat Free', 'Alternative Long Life Milk', 'Dairy Free Spread', 
        'Sunflower Spread',
        'Olive Based Spread',
        'Olive Based',
        'Vegetarian Fresh  Meal Solutions',
        'Convenience Meals - Meat Free',
        'Cooking Aids - Meat Free',
        'Homebaking - Meat Free',
        'International Foods - Meat Free',
        'Vegetarian']


        if ('vegetarian' in user_input_clean) or ('meat free' in user_input_clean):
            df_final.accuracy_adjusted = np.where(df_final['class'].isin(vegetarian_class)|
                                                  df_final['subclass'].isin(vegetarian_subclass)|
                                                  df_final.description.str.lower().str.contains('vegetarian')|
                                                  df_final.description.str.lower().str.contains('meat free')|
                                                  df_final.description.str.lower().str.contains('plant revolution')|
                                                  df_final.description.str.lower().str.contains('plant based')|
                                                  df_final.description.str.lower().str.contains('quorn')|
                                                  df_final.description.str.lower().str.contains('veggie')|
                                                  df_final.description.str.lower().str.contains('vegan'), df_final.accuracy_adjusted - 0.2, df_final.accuracy_adjusted)

        if ('vegan' in user_input_clean) or ('plant based' in user_input_clean):
            df_final.accuracy_adjusted = np.where((df_final['class'].isin(vegan_class)|
                                                  df_final['subclass'].isin(vegetarian_subclass)|
                                                  df_final.description.str.lower().str.contains('vegan')|
                                                   df_final.description.str.lower().str.contains('meat free')|
                                                   df_final.description.str.lower().str.contains('plant revolution')|
                                                  df_final.description.str.lower().str.contains('plant based')|
                                                  df_final.description.str.lower().str.contains('quorn')|
                                                  df_final.description.str.lower().str.contains('veggie'))&
                                                  ~df_final.subclass.isin(['Goat Milk', 'GIB Milk & Cream (Fresh)']), df_final.accuracy_adjusted - 0.2, df_final.accuracy_adjusted)


        # which columns are we interested in keeping?
        l_columns = ['aisle', 'side', 'bay', 'gondola', 'gondola_aisle', 'accuracy', 'accuracy_adjusted', 'sku_item_name',
                     'sku_item_nbr',
                     'a_description', 'description', 'm_description', 'boost',
                     # 'trading_group', 'a_trading_group',
                     'a_category', 'category', 
                     'a_department', 'department', #'m_department',
                     'class','a_class',#'m_class',
                     'subclass', 'a_subclass',
                     'l_dept', 'broad_product_type_match_score', 'broad_product_type_match', 'search_term_score'
                    ]
        df_final = df_final[l_columns]

        # filter out nan values
        df_final = df_final.loc[df_final.accuracy_adjusted.notna()]

    except Exception as e3:

        print('Embedding function - Failed to create similarity scores.')
        print(e3)

    try:        
        # which columns do we want to use to create the user input?
        l_print_columns = ['aisle', 'side', 'bay', 'gondola','gondola_aisle',
                           'l_dept', 'category', 'department', 'class', 'subclass', 'description', 'a_description', 'm_description', 'boost', 'accuracy', 'accuracy_adjusted', 'vol_weight', 'acc_weight', 'type_weight', 'online_weight', 'weight',
                           'volume', 'broad_product_type_match', 'broad_product_type_match_score', 'search_term_score']

        # filter output to display the most related products
        pctile = np.percentile(df_final.accuracy_adjusted, 0.1) # cutoff used to determine if user request is a broad or specific ask

        print(pctile)

        # for veg in ['vegetarian', 'vegan', 'plant based', 'meat free']:
        #     if veg in user_input_clean:
        #         print('yes')
        #         pctile = 0.05
        #         break

        if pctile < -0.05: # broad ask
            # print('broad ask')
            ask_type = 'broad'
            print(f'Ask Type: {ask_type}')
            prod_count = df_loc.shape[0]
            broad_search_cutoff = int(np.array((0.000556 * prod_count + 13.333)).clip(min = 20, max = 30).round(0))
            
            # df_final.accuracy_adjusted =  np.where(df_final.product_rank < 50, df_final.accuracy_adjusted - df_final.search_term_score * 0.15 + 0.05, df_final.accuracy_adjusted)

            accuracy_range = df_final.sort_values('accuracy_adjusted',ascending = True).head(n = broad_search_cutoff).accuracy_adjusted.std()
            print(accuracy_range)
            


            df_output = (df_final.sort_values('accuracy_adjusted',ascending = True)
                                  .head(n = broad_search_cutoff)
                                  .merge(df_sales, how = 'inner', on = 'sku_item_nbr')
                                  .assign(
                                      vol_weight = lambda x: x.volume.rank(ascending = False) * vol_w,
                                      acc_weight = lambda x: x.accuracy_adjusted.rank(ascending = True) * acc_w,
                                      type_weight = lambda x: x.broad_product_type_match_score.rank(ascending = True) * type_w,
                                      online_weight = lambda x: x.search_term_score.rank(ascending=False) * online_w,
                                      weight = lambda x: x.acc_weight + x.type_weight + x.vol_weight + x.online_weight)
                                  .sort_values('weight', ascending = True)
                                  .loc[lambda x: x.accuracy_adjusted < 0.145]
                                  .head(n = 50)
                                  [l_print_columns]
                         )

        else:
            # print('specific ask')
            ask_type = 'specific'
            print(f'Ask Type: {ask_type}')
            
            # df_final = pd.merge(df_final, df_search_term_boost[['sku_item_nbr', 'search_term_score']], on='sku_item_nbr', how='left')
            # df_final.search_term_score = df_final.search_term_score.fillna(0)
            # df_final.accuracy_adjusted = df_final.accuracy_adjusted - df_final.search_term_score*0.2
            
            df_output = (df_final.sort_values('accuracy_adjusted',ascending=True)
                              .loc[lambda x: x.accuracy_adjusted < 0.155]
                             .assign(
                                     vol_weight=None,
                                     acc_weight = lambda x: x.accuracy_adjusted.rank(ascending = True) * acc_w,
                                     type_weight=None,
                                     online_weight = lambda x: x.search_term_score.rank(ascending=False) * online_w,
                                     weight = lambda x: x.acc_weight + x.online_weight)
                              .sort_values('weight', ascending = True)
                              .head(50)
                              .merge(df_sales, how = 'inner', on = 'sku_item_nbr')
                              [l_print_columns]
                     )
            # df_output['weight'] = None ## to avoid errors when weight not assigned


    except Exception as e4:
        print('Embedding function - Failed to create final product selection.')
        print(e4)


    try:
        aisle = df_output.iloc[0].aisle
        gondola_aisle = df_output.iloc[0].gondola_aisle
        bay = df_output.iloc[0].bay
        side = df_output.iloc[0].side
        max_bays = df_loc.loc[df_loc.aisle == aisle].bay.max()
        l_dept = df_output.iloc[0].l_dept

        side_map = {'L': 'left', 'R': 'right'}
        end_map = {'L': lambda x: f'start of aisle {x} facing the checkouts.',
                   'R': lambda x: f'end of the aisle {x} facing the back of the store.'}
        back_end_map = {'L': lambda x: f'back of the store across from aisle {x}.',
                        'R': lambda x: f'end of the aisle {x} facing the back of the store.'}

        d_message = {
            97: f"In {store_name} store '{df_output.iloc[0].description}' can be found at the start of aisle {gondola_aisle} facing the checkouts.\n\n",
            96: f"In {store_name} store '{df_output.iloc[0].description}' can be found at the {end_map.get(side)(gondola_aisle)}\n\n",
            95: f"In {store_name} store '{df_output.iloc[0].description}' can be found at the {back_end_map.get(side)(gondola_aisle)}\n\n",
            94: f"In {store_name} store '{df_output.iloc[0].description}' can be found on the same fixture as the {', '.join(l_dept)} in the Fresh Produce Section.\n\n",
            93: f"In {store_name} store '{df_output.iloc[0].description}' can be found in the Horticulture area at the front of the store.\n\n",
            92: f"In {store_name} store '{df_output.iloc[0].description}' can be found in the Beers, Wines and Spirits section of the store.\n\n",
            91: f"In {store_name} store '{df_output.iloc[0].description}' can be found on the same aisle as the {', '.join(l_dept[:3])} etc.\n\n",   # fresh pre-pack aisle
            90: f"In {store_name} store '{df_output.iloc[0].description}' can be found at the Fresh Fish Counter.\n\n",
            89: f"In {store_name} store '{df_output.iloc[0].description}' can be found at the Fresh Food To Go Counter.\n\n",
            88: f"In {store_name} store '{df_output.iloc[0].description}' can be found at the Deli Counter.\n\n",
            87: f"In {store_name} store '{df_output.iloc[0].description}' can be found at the Oven Baked Counter.\n\n",
            86: f"In {store_name} store '{df_output.iloc[0].description}' can be found at the in-store Bakery.\n\n",
            85: f"In {store_name} store '{df_output.iloc[0].description}' can be found at the in-store Bakery.\n\n",
            84: f"In {store_name} store '{df_output.iloc[0].description}' can be found at the Butchery Counter.\n\n",
            83: f"In {store_name} store '{df_output.iloc[0].description}' can be found on the Frozen aisles.\n\n",
            77: f"In {store_name} store '{df_output.iloc[0].description}' can be found at the in-store Party shop.\n\n",
            76: f"In {store_name} store '{df_output.iloc[0].description}' can be found in the Nutmeg section of the store.\n\n",
            75: f"In {store_name} store '{df_output.iloc[0].description}' can be found in the Free From aisle.\n\n",
            72: f"In {store_name} store '{df_output.iloc[0].description}' can be found at the Customer Service Kiosk at the front of the store.\n\n",
            70: f"In {store_name} store '{df_output.iloc[0].description}' can be found down the Seaonal Aisle.\n\n"
        }


        text_response = d_message.get(aisle)

        if text_response == None:
            text_response = f"In {store_name} store '{df_output.iloc[0].description}' can be found on aisle {aisle}.\nWith your back to the checkouts they can be found on the {side_map.get(side)}-hand side, {bay} bays from the checkout.\n\n"

        aisle_numbers = [97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 77, 76, 75, 72, 70, 61]
        aisle_names = ["(End of Aisle near Checkouts)",
                       "(Middle Aisle Section on Promotional Ends)",
                       "(End of Aisle facing the back of the Store)",
                       "(Fresh Fruit & Veg Section)",
                       "(Plants and Flowers Section)",
                       "(Beers, Wines and Spirits Section)",
                       "(Fresh Food Pre-Packed Aisle (Cheese, juices))",
                       "(Fishmonger Counter (Around the side/back of store))",
                       "(Fresh To Go Counter (Around the side/back of store))",
                       "(Deli Counter (Around the side/back of store))",
                       "(Oven Baked Counter (Around the side/back of store))",
                       "(Cake Shop (Around the side/back of store))",
                       "(Bakery Section (Around the side/back of store))",
                       "(Butchery Counter (Around the side/back of store))",
                       "(Frozen Section/Aisle)",
                       "(Party Zone (Around the side/back of store))",
                       "(Nutmeg Clothing Section)",
                       "(Free From Section (Around the side/back of store))",
                       "(Kiosk Section (Tobacco))",
                       "(Seasonal Block)",
                       "(Brew Coffee Shop)"]
        d_aisle_map = {i: j for i, j in zip(aisle_numbers, aisle_names)}
        location_string = (lambda aisle, bay: d_aisle_map.get(aisle, f'(Aisle: {aisle}, Bay: {bay})'))

        # add a list of other relevant products
        if len(df_output) > 1:
            l_top_products = (f"If you like, below are some other related products I can help you find:\n" +
                              ',\n'.join([str(ix + 1) + f'. {row.description} {location_string(row.aisle, row.bay)}' for ix, (_, row) in enumerate(df_output.iloc[1:].iterrows()) if ix < 5]) + '.')

            text_response = text_response + l_top_products

    except Exception as e5:
        print("Embedding function - Failed to build chat message from final product selection.")
        print(e5)

    return text_response#, df_output, df_final



def hello_chat(request):
    print(request)
    
    if request.headers.get('secret_header_key') == 'OKE4ZdOZw4sb8olR7U1lEOFCYLsF3A5olmuP9JdvqY0uWgZT':

        # attempt to pull information from the webook request
        parameters_identified = False
        
        try:
            request_json = request.get_json()
            print(request_json)

            # pull parameters from json request
            product = request_json['queryResult']['parameters']['product'].lower()
            store_name = request_json['queryResult']['parameters']['store']
            user_name = request_json['originalDetectIntentRequest']['payload']['data']['event']['user']['displayName']
            email = request_json['originalDetectIntentRequest']['payload']['data']['event']['user']['email']

            # find other store identification parameters
            df_map = pd.read_parquet(BytesIO(bucket.blob('store_mappings.parquet').download_as_bytes()))
            d_map = df_map.set_index('store_name').to_dict()['bsns_unit_cd']
            store_id = d_map[store_name]
            store = int(store_id)

            parameters_identified = True
            print(f"Product: {product}, Store Name {store_name}, Store Id: {store_id}, Store: {store}, User: {user_name}, Email: {email}")
            
        except Exception as e1:
            print("Main function - Failed to extract users parameters.")
            print(e1)
        
        # determine what our text response should be
        if parameters_identified:

            # have we been sent this request before?
            #seen_before, past_response = check_the_past(store_id, product)
            seen_before = False

            if seen_before:
                text_response = past_response

            else:
                try:
                    text_response = help_me_find(product, store, store_name)

                    # upload text response to GCP
                    upload_response(store_id, product, email, text_response)

                except Exception as e2:
                    print("Main function - Failed to generate a text response.")
                    print(e2)
                    text_response = "I'm sorry we were unable to help you, please try again."

        d_response_json = {"fulfillmentMessages": [{"text": {"text": [text_response]}}]}
        print(text_response)

    else:
        d_response_json = {"fulfillmentMessages": [{"text": {"text": ["User request not authenticated. We are sorry we are unable to help you."]}}]}

    return json.dumps(d_response_json)
