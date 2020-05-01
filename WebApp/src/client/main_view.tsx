
import * as React from "react";
import "./main_view.scss";
import { observable, action } from "mobx";
import { observer } from "mobx-react";

export interface MainViewProps {
    background: string;
}

@observer
export default class MainView extends React.Component<MainViewProps> {

    render() {
        const { background } = this.props;
        return (
            <div
                className={"container"}
                style={{ background }}
                
            >
                <span className={"welcome"}>Here's the information you're looking for!</span>
            </div>
        );
    }

}