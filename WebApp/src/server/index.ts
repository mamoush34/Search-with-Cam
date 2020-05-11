import * as express from "express";
import { resolve } from "path";
import * as cors from "cors";
import * as multer from "multer";
const WebSocket = require('ws');

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


let storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, '../communication/rawimage')
  }, 
  filename: (req, file, cb) => {
    //check to see if there are duplicates //TODO: 
    cb(null, file.originalname)
  }
})

let upload = multer({ storage: storage})

server.post('/upload', upload.single('rawimage'), (req, res, next) => {
  if (!req.file) {
    return next(new Error('File has not been uploaded'))
  }
  // return res.send(req.file);
});




 
const wss = new WebSocket.Server({ port: 1051 });
 
wss.on('connection', function connection(ws:any) {
  ws.on('message', function incoming(message:any) {
    console.log('received: %s', message);
    ws.send("Received Images.");
  });
 
  ws.send('something');
});


server.listen(port);
