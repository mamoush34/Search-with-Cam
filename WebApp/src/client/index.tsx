import * as React from "react";
import * as ReactDOM from "react-dom";
import MainView from "./main_view";
import {Utilities} from "./utilities"

ReactDOM.render(<MainView background={Utilities.url("wooden_background.jpg")} />, document.getElementById("root"));