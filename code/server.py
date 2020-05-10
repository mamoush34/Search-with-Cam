import json 
import asyncio
import websockets
import os
from analyze import segmentation


async def receiver(websocket, path):
    jsonString = await websocket.recv()
    jsonObject = json.loads(jsonString);
    if (jsonObject["type"] == "segmentation"):
        filename = jsonObject["filename"]
        path = "../../communication/rawimage/" + filename
        norm_points = segmentation(path)
        obj = {
            "type" : "segmentation", 
            "boxes" : norm_points, 
            "iou" : 0.65,
        }
        string = json.dumps(obj)
        await websocket.send(string)
        

    elif (jsonObject["type"] == "predict"):
        print(jsonObject)
        



    # await websocket.send(greeting)



if __name__ == "__main__":  
    #run the socket server
    start_server = websockets.serve(receiver, "localhost", 1234)

    print("Server running.")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()