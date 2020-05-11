import * as React from "react";
import "./main_view.scss";
import { observable, action, runInAction } from "mobx";
import { observer } from "mobx-react";
import { json } from "express";


@observer
export default class MainView extends React.Component {

    private socket1:WebSocket = new WebSocket("ws://localhost:1234/1");
    private socket2:WebSocket = new WebSocket("ws://localhost:1234/2");
    private file = undefined;
    @observable resultList:JSX.Element[] = [];
    
    componentDidMount() {
        this.socket1.onopen = (event) => {
            console.log("Client has been connected to the python server.")
        }
        this.socket1.onmessage = (event) => {
            const data = event.data
            let jsonData = JSON.parse(data)

            if (jsonData.type = "segmentation") {
                const reader:FileReader = new FileReader();
                reader.onload = () => {
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
                    this.drawBoxes(ctx, jsonData, "green")        
                }
                reader.readAsDataURL(this.file!);
            }
        }
        this.socket2.onmessage = (event) => {
            const data = event.data
            let jsonData = JSON.parse(data)
            if (jsonData.type = "predict") {
                const reader:FileReader = new FileReader();
                reader.onload = () => {
                    const elements:JSX.Element[] = [];
                    const div = <div className ="container">{elements}</div>
                    const img = document.createElement("img");
                    img.src = reader.result as string;
                    img.style.height = "280";
                    img.style.width = "400";
                    let canvasref = React.createRef<HTMLCanvasElement>()
                    const canvas = <canvas id="predict-canvas" ref={canvasref} height="280" width="400" ></canvas>
                    const header = <h2>Object Detected</h2>   
                    elements.push(header);
                    elements.push(canvas);
                    this.resultList.push(div)
                    this.forceUpdate()
                    let c:HTMLCanvasElement = document.getElementById("predict-canvas") as HTMLCanvasElement;
                    let ctx = c.getContext("2d")!;
                    ctx.drawImage(img, 0, 0, 400, 280)
                    this.drawBoxes(ctx, jsonData, "red")
                    const labels:string[] = jsonData.labels
                    if (labels.length != 0){
                        this.resultList.push(<h2>Wikipedia link: https://en.wikipedia.org/wiki/ + {labels[0]}</h2>)
                    }
                }
                reader.readAsDataURL(this.file!); 
            }
        }
    }

    @action 
    drawBoxes = (ctx:CanvasRenderingContext2D, jsonData:any, color:string) => {
        let boxes:number[] = jsonData.boxes
        let labels:string[] = []
        let percentages:number[] = []
        if (color == "red"){
            labels = jsonData.labels
            percentages = jsonData.percentages
        }
        ctx.beginPath();
        for (let i = 0; i < boxes.length; i += 4){
            const xmin = boxes[i] * 400
            const xmax = boxes[i + 1] * 400
            const ymin = boxes[i + 2] * 280
            const ymax = boxes[i + 3] * 280
            const width = xmax - xmin
            const height = ymax - ymin;
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.rect(xmin, ymin, width, height)      
            if (color == "red"){
                ctx.font = "bold 14px Arial";
                ctx.fillStyle = color
                const info = labels[i / 4]
                ctx.fillText(info, xmin, ymin)
            }      
            ctx.stroke()
        }
    }

    @action
    onFileLoad = (e:any) => {
        e.preventDefault();
        const uploadform = document.getElementById("uploadform")! as HTMLFormElement;
        uploadform.submit()
        
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
                this.socket1.send(json1);
                let json2 = `{"type" : "predict", "filename": "${file.name}"}`                
                this.socket2.send(json2);
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
                {this.resultList.map(el => {return el;})}
            </div>
            
        );
    }

}