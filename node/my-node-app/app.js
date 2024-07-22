var express = require('express');
var app = express();

const hostname = 'localhost';
const port = 3000;


app.get('/', function (req, res) {
  res.send('Hello World! My First Program in Node...');
});
app.listen(3000, function () {
console.log(`Server running at ${port}: http://${hostname}:${port}/`);
});