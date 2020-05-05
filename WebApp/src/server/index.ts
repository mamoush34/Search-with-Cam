import * as express from "express";
import { resolve } from "path";
import * as cors from "cors";

const port = 1050;

const static_path = resolve(__dirname, "../../static");
const content_path = resolve(__dirname, "../../src/index.html");

const server = express();

const {spawn} = require('child_process')
const path  = require('path')

server.use(cors());
server.use(express.static(static_path));
server.use((req, _res, next) => {
    console.log(req.originalUrl);
    next();
});

console.log(`Server listening on port ${port}...`);
server.get("/", (_req, res) => res.redirect("/home"));
server.get("/home", (_req, res) => res.sendFile(content_path));


function runScript() {
    return spawn('python', [path.join(__dirname, '../code/run.py')])
}

const script_process = runScript()

script_process.stdout.on('data', (data: any) => {
    console.log(`data:${data}`);
})
script_process.stderr.on('data', (data) => {
    console.log(`error:${data}`);
  });
  script_process.on('close', () => {
    console.log("Closed");
  });



server.listen(port);
