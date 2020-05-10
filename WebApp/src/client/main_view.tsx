import * as React from "react";
import "./main_view.scss";
import { observable, action, runInAction } from "mobx";
import { observer } from "mobx-react";
import { json } from "express";


@observer
export default class MainView extends React.Component {

    private socket:WebSocket = new WebSocket("ws://localhost:1234");
    private file = undefined;
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
        }
        this.socket.onmessage = (event) => {
            const data = event.data
            let jsonData = JSON.parse(data)
            if (jsonData.type = "segmentation") {
                console.log("Segmentation!");
                const reader:FileReader = new FileReader();
                reader.onload = () => {
                    runInAction(() => {
                        const elements:JSX.Element[] = [];
                        const div = <div className ="container">{elements}</div>
                        const img = document.createElement("img");
                        img.src = reader.result as string;
                        img.style.height = "280";
                        img.style.width = "400";
                        let canvasref = React.createRef<HTMLCanvasElement>()
                        const canvas = <canvas id="segmentation-canvas" ref={canvasref} height="280" width="400" ></canvas>
                        

                        const header = <h2>Segmented Image</h2>   
                        const message = <h3>IOU: {jsonData.iou}</h3>   
                        elements.push(header);
                        elements.push(message);
                        elements.push(canvas);
                        this.resultList.push(div)
                        this.forceUpdate()
                        let c:HTMLCanvasElement = document.getElementById("segmentation-canvas") as HTMLCanvasElement;
                        let ctx = c.getContext("2d")!;
                        ctx.drawImage(img, 0, 0, 400, 280)

                        let boxes:number[] = jsonData.boxes
                        console.log(boxes)
                        ctx.beginPath();
                        for (let i = 0; i < boxes.length; i += 4){
                            
                            const xmin = boxes[i] * 400
                            const xmax = boxes[i + 1] * 400
                            const ymin = boxes[i + 2] * 280
                            const ymax = boxes[i + 3] * 280
                            const width = xmax - xmin
                            const height = ymax - ymin;
                            ctx.strokeStyle = "green";
                            ctx.lineWidth = 1;
                            ctx.rect(xmin, ymin, width, height)
                            ctx.stroke()
                        }
                        
                    })
                    
                }
                reader.readAsDataURL(this.file!);
            } else if (jsonData.type = "predict") {
                console.log("Prediction!")
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
            this.file = file
            const reader:FileReader = new FileReader();
            reader.onload = () => {
                const elements:JSX.Element[] = [];
                const div = <div className ="container">{elements}</div>
                const img = <img src={(reader.result as string)} style={{height: "280px", width:"400px"}}></img>
                const text = <h2>Your raw input image</h2>   
                elements.push(text)
                elements.push(img)
                this.resultList.push(div)
                //let the python server know the file has been loaded
                let json1 = `{"type" : "segmentation", "filename": "${file.name}"}`
                let json2 = `{"type" : "predict", "filename": "${file.name}"}`
                this.socket.send(json1);
                this.socket.send(json2);
            }
            reader.readAsDataURL(file);
        }
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