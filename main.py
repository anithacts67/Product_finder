import os
import json
from google.cloud import bigquery
from google.cloud import storage
import gcsfs
import datetime


def dialogflow_connect(request):
    parameters_identified = False
    request_json = request.get_json()
    print(request_json)
    #res = {"parameters": {"messages": [{"text": {"text": [text]}}]}}
    
    tag = request_json["fulfillmentInfo"]["tag"]
    if tag == "full info":
        
        #text= "please wait"
        string_field_0=request_json['intentInfo']['parameters']['product']['resolvedValue']
        #store_name=request_json['intentInfo']['parameters']['store_entries_en']['resolvedValue']
        print(string_field_0)
        #print(store_name)
        # find other store identification parameters
        client = bigquery.Client()
    query = f"""
       SELECT string_field_0, string_field_1, string_field_2 
       FROM `cog01j2gvb54gvrcnk6c95qw452aw.demo_dataset_store.product_new` 
       WHERE product_name = '{string_field_0}'
    """
    query_job = client.query(query)
    result = query_job.result()
    for row in result:
        
        text_response = f"Product {row.string_field_0} is located at aisle {row.string_field_2} in {row.string_field_1}."
        
   
                  
    result={"fulfillment_response": {"messages": [{"text": {"text": [text_response]}}]}}
    return result

       
