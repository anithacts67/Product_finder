from google.cloud import dialogflowcx_v3beta1
from google.protobuf.json_format import MessageToJson
import json



def chat(input):
    # Create a client
    
    client = dialogflowcx_v3beta1.SessionsClient(client_options={"api_endpoint": "us-central1-dialogflow.googleapis.com"})
    
    query_input = dialogflowcx_v3beta1.QueryInput()
    query_input.text.text = input
    query_input.language_code = "en"



    request = dialogflowcx_v3beta1.DetectIntentRequest(
        session="projects/cog01j2gvb54gvrcnk6c95qw452aw/locations/us-central1/agents/de99d644-2cb8-4618-9166-58c5a4902a7a/environments/Draft/sessions/*",
        query_input=query_input,
    )


    # Making request
    response = client.detect_intent(request=request)
    print(response) # object as <class 'google.cloud.dialogflow_v2.types.DetectIntentResponse'>
    response_json = MessageToJson(response._pb) # to json string
    #print(response_json) 
    response_dict= json.loads(response_json) # to convert into dict
    #print(response_dict)
    print(response_dict['queryResult']['responseMessages'][0]['text']['text'][0]) 
    print(response_dict['queryResult']['intentDetectionConfidence']) 
    print(response_dict['queryResult']['intent']['displayName'])
    
    # Parsing the response without coverting to json or dict
    '''query_result = response.query_result
    response_messages= query_result.response_messages 
    intent = query_result.intent
    display_name = intent.display_name
    confidence = query_result.intent_detection_confidence
  
    print(f"response: {response_messages[0]}")
    print(f"Intent: {display_name}")
    print(f"Confidence: {confidence}")'''
#calling chat function
chat("hi")

       
