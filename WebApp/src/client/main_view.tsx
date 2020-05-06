
import * as React from "react";
import "./main_view.scss";
import { observable, action } from "mobx";
import { observer } from "mobx-react";



@observer
export default class MainView extends React.Component {

    @observable resultList:JSX.Element[] = [];

    componentDidMount() {
        let segmentationSocket:WebSocket = new WebSocket("/getsegmentation")
        let resultSocket:WebSocket = new WebSocket("/getresult");
        segmentationSocket.onmessage = (event) => {
            //parse the event data
            /**
             * Should be a JSON data that has:
             * {path: path to segmented image
             *  iou: value
             *  segmentation: num of seg
             * }
             */
        }
        resultSocket.onmessage = (event) => {
            //parse the event data. 
            /**
             * Should be a JSON data that has:
             * {boundingboxes: arrays of [xmin, xmax, ymin, ymax]}
             * 
             */
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
                const img = <img src={(reader.result as string)}></img>
                const text = <h2>Your raw input image</h2>   
                this.resultList.push(text);
                this.resultList.push(img);
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
                <div className={"container"}>
                    <h1 className="wrapper">Search With Cam!</h1>
                    <div>
                        <h2 >R-CNN with Visual Information</h2>
                        <h3 > Click on "Analyze Image" below to input your image.</h3>
                    </div>
                    <form action="/upload" method="post" encType="multipart/form-data" name="uploadform" id ="uploadform">
                        <input type="file" name= "rawimage" accept="image/jpeg, image/jpg, image/png"/> 
                        <input type="submit" value="Upload a file" onClick={this.onFileLoad}/>
                    </form>
                    {/* <button onClick = {() => {document.getElementById('selectedFile')!.click()}}>Analyze Image</button>  */}
                         
                </div>
                <div className="container"> 
                    {this.resultList.map(el => {return el;})}
                </div>
            </div>
            
        );
    }

}