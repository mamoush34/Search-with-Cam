import json 
import asyncio
import websockets
import os
from analyze import segmentation, predict
from keras.models import load_model


model = None; 

async def receiver(websocket, path):
    """
    Socket that handles segmentation and predict queries
    INPUT: websocket - the tcp connection 
           path - path to the image
    OUTPUT: none (except this function sends result back to nodejs client)
    """
    jsonString = await websocket.recv()
    jsonObject = json.loads(jsonString);
    print(jsonObject["type"])
    if (jsonObject["type"] == "segmentation"):
        filename = jsonObject["filename"]
        path = "../communication/rawimage/" + filename
        norm_points = segmentation(path)
        obj = {
            "type" : "segmentation", 
            "boxes" : norm_points, 
            "iou" : 0.65,
        }
        string = json.dumps(obj)
        await websocket.send(string)
    if (jsonObject["type"] == "predict"):
        filename = jsonObject["filename"]
        path = "../communication/rawimage/" + filename
        global model
        norm_points, labels, percentages = predict(model, path)
        print("done")
        obj = {
            "type" : "predict", 
            "boxes" : norm_points, 
            "labels" : labels,
        }
        string = json.dumps(obj)
        print("sent")
        await websocket.send(string)
    
        






if __name__ == "__main__":  
    #run the socket server
    start_server = websockets.serve(receiver, "localhost", 1234)
    #need to load in the model.
    print("Loading R-CNN model...")
    model = load_model("./saved_models/rcnn_vgg16_1.h5")

    print("Server running.")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()