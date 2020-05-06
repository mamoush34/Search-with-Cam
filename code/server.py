import json 
import numpy as np
import math
import http
import bottle
from bottle import run, post, request, response
import config


bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024 * 1024 * 1024




@post('/predict')
def get_keywords_post():
    def action(phrase):
        client = authenticate_client()
        return get_keywords(client, phrase)
    return jsonify('phrase', action)




def jsonify(key, action):
    received_data = dict(request.json) #turn json file into dictionary
    if key in received_data: #if there is key in the attribute
        result = {'result' : action(received_data[key]), 'status' : 'success'}
        return json.dumps(result)
    return json.dumps({'status' : 'failed'})
    


if __name__ == "__main__":  
    #run the server
    run(host=config.HOST, port=config.PORT)