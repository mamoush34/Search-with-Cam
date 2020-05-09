import * as React from "react";
import "./main_view.scss";
import { observable, action } from "mobx";
import { observer } from "mobx-react";
import { json } from "express";


@observer
export default class MainView extends React.Component {

    private socket:WebSocket = new WebSocket("ws://localhost:1051");
    @observable resultList:JSX.Element[] = [];
    
    componentDidMount() {
        if ("WebSocket" in window){
            console.log("Websocket is supported")
        }
        this.socket.onopen = (event) => {
            //parse the event data
            /**
             * Should be a JSON data that has:
             * {path: path to segmented image
             *  iou: value
             *  segmentation: num of seg
             * }
             */
            console.log("recieved");
            this.socket.send("Hello, Server!")
        }
        this.socket.onmessage = (event) => {
            const data = event.data
            let jsonData = JSON.parse(data)
            if (jsonData.type = "segmentation") {
                const elements:JSX.Element[] = [];
                const div = <div className ="container">{elements}</div>
                const img = <img src={jsonData.imgpath} style={{height: "224px", width:"224px"}}></img>
                const header = <h2>Segmented Image</h2>   
                const message = <h3>IOU: {jsonData.iou}</h3>   
                elements.push(header);
                elements.push(message);
                elements.push(img);
                this.resultList.push(div);
            } else if (jsonData.type = "result") {
                const elements:JSX.Element[] = [];
                const div = <div className ="container">{elements}</div>
                const img = <img src={jsonData.imgpath} style={{height: "224px", width:"224px"}}></img>
                const header = <h2>Segmented Image</h2>   
                const message = <h3>IOU: {jsonData.iou}</h3>   
                elements.push(header);
                elements.push(message);
                elements.push(img);
                this.resultList.push(div);
            }
        }
        
    }

    @action
    onFileLoad = (e:any) => {
        e.preventDefault();
        const uploadform = document.getElementById("uploadform")! as HTMLFormElement;
        uploadform.submit();
        //@ts-ignore
        const files = uploadform.elements["rawimage"].files
        if (files.length != 0) {
            this.resultList = [];
            const file = files[0]
            const reader:FileReader = new FileReader();
            reader.onload = () => {
                const elements:JSX.Element[] = [];
                const div = <div className ="container">{elements}</div>
                const img = <img src={(reader.result as string)} style={{height: "280px", width:"400px"}}></img>
                const text = <h2>Your raw input image</h2>   
                elements.push(text)
                elements.push(img)
                this.resultList.push(div)
            }
            reader.readAsDataURL(file);
        }
    }

    @action 
    onResult = () => {

    }

    @action 
    onSegmentation = () => {

    }


    render() {
        return (
            <div>
                <div className="query-container">
                    <div>
                        <h1 className="wrapper">Search With Cam!</h1>
                        <h2>R-CNN with Visual Information</h2>
                    </div>
                    <div>
                        <form action="/upload" method="post" encType="multipart/form-data" name="uploadform" id ="uploadform">
                            <input type="file" name= "rawimage" accept="image/jpeg, image/jpg, image/png"/> 
                        </form>
                        <button onClick = {this.onFileLoad}>Analyze Image</button> 
                    </div>
                   
                </div>
                <div className="container"> 
                    {this.resultList.map(el => {return el;})}
                </div>
            </div>
            
        );
    }

}