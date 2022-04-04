/*
    This file can be used to ingest and retrieve a (dummy) model from
	Triabase.
    Copyright (C) 2021  Stavroulla Koumou

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.html.
*/

const http = require('http')
const fs = require('fs')

var myArgs = process.argv.slice(2);
const MODEL = myArgs[1]; //darknet | pytorch | tfv2 | tfjs | tflite
const RUN = myArgs[0]; // submit | retrieve
const TEST = myArgs[2]; // true | false
const TEST_MODEL_SIZE = parseInt(myArgs[3]) * 1024 * 1024; //MB
const AVG_COUNT = parseInt(myArgs[4]);
const INTERVAL = parseInt(myArgs[5]);

function darknet(){
	let architecture = fs.readFileSync('./data/experiments/darknet/yolov3.cfg', {encoding: 'base64'});
	let weights = fs.readFileSync('./data/experiments/darknet/yolov3.weights', {encoding: 'base64'});
	let data = JSON.stringify({
		"token": "token",
		"data": {
			"id": "darknet",
			"tag1":"yolo",
			"tag2":"object_detection",
			"serialization_encoding":"base64",
			"model": [
				{
					"metadata":
					{
						"identifier":"yolov3",
						"original_format":"cfg"
					},
					"serialized_data": architecture
				}
			],
			"weights":[
				{
					"metadata":
					{
						"identifier":"yolov3",
						"original_format":"weights"
					},
					"serialized_data": weights
				}
			],
			"initialization":[ ],
			"checkpoints":[ ]
		}
	})
	
	return data;
}

function pytorch(){
	let model = fs.readFileSync('./data/experiments/pytorch/yolov5s.pt', {encoding: 'base64'});
	let data = JSON.stringify({
		"token": "token",
		"data": {
			"id": "pytorch",
			"tag1":"yolo",
			"tag2":"object_detection",
			"serialization_encoding":"base64",
			"model": [
				{
					"metadata":
					{
						"identifier":"yolov5s",
						"original_format":"pt"
					},
					"serialized_data": model
				}
			],
			"weights":[ ],
			"initialization":[ ],
			"checkpoints":[ ]
		}
	})
	
	return data;
}

function tfv2(){
	let variables1 = fs.readFileSync('./data/experiments/tf-v2/variables/variables.data-00000-of-00001', {encoding: 'base64'});
	let variables2 = fs.readFileSync('./data/experiments/tf-v2/variables/variables.index', {encoding: 'base64'});
	let model = fs.readFileSync('./data/experiments/tf-v2/saved_model.pb', {encoding: 'base64'});
	let data = JSON.stringify({
		"token": "token",
		"data": {
			"id": "tfv2",
			"tag1":"ssd_mobilenet_v2",
			"tag2":"object_detection",
			"serialization_encoding":"base64",
			"model": [
				{
					"metadata":
					{
						"identifier":"saved_model",
						"original_format":"pb"
					},
					"serialized_data": model
				}
			],
			"weights":[ ],
			"initialization":[ ],
			"checkpoints":[
				{
					"metadata":
					{
						"identifier":"variables",
						"original_format":"data-00000-of-00001"
					},
					"serialized_data": variables1
				},
				{
					"metadata":
					{
						"identifier":"variables",
						"original_format":"index"
					},
					"serialized_data": variables2
				}
			 ]
		}
	})
	
	return data;
}

function tfjs(){
	let shard1 = fs.readFileSync('./data/experiments/tfjs/group1-shard1of17', {encoding: 'base64'});
	let shard2 = fs.readFileSync('./data/experiments/tfjs/group1-shard2of17', {encoding: 'base64'});
	let shard3 = fs.readFileSync('./data/experiments/tfjs/group1-shard3of17', {encoding: 'base64'});
	let shard4 = fs.readFileSync('./data/experiments/tfjs/group1-shard4of17', {encoding: 'base64'});
	let shard5 = fs.readFileSync('./data/experiments/tfjs/group1-shard5of17', {encoding: 'base64'});
	let shard6 = fs.readFileSync('./data/experiments/tfjs/group1-shard6of17', {encoding: 'base64'});
	let shard7 = fs.readFileSync('./data/experiments/tfjs/group1-shard7of17', {encoding: 'base64'});
	let shard8 = fs.readFileSync('./data/experiments/tfjs/group1-shard8of17', {encoding: 'base64'});
	let shard9 = fs.readFileSync('./data/experiments/tfjs/group1-shard9of17', {encoding: 'base64'});
	let shard10 = fs.readFileSync('./data/experiments/tfjs/group1-shard10of17', {encoding: 'base64'});
	let shard11 = fs.readFileSync('./data/experiments/tfjs/group1-shard11of17', {encoding: 'base64'});
	let shard12 = fs.readFileSync('./data/experiments/tfjs/group1-shard12of17', {encoding: 'base64'});
	let shard13 = fs.readFileSync('./data/experiments/tfjs/group1-shard13of17', {encoding: 'base64'});
	let shard14 = fs.readFileSync('./data/experiments/tfjs/group1-shard14of17', {encoding: 'base64'});
	let shard15 = fs.readFileSync('./data/experiments/tfjs/group1-shard15of17', {encoding: 'base64'});
	let shard16 = fs.readFileSync('./data/experiments/tfjs/group1-shard16of17', {encoding: 'base64'});
	let shard17 = fs.readFileSync('./data/experiments/tfjs/group1-shard17of17', {encoding: 'base64'});
	let model = fs.readFileSync('./data/experiments/tfjs/tensorflowjs_model.pb', {encoding: 'base64'});
	let architecture = fs.readFileSync('./data/experiments/tfjs/model.json', {encoding: 'base64'});
	let weights = fs.readFileSync('./data/experiments/tfjs/weights_manifest.json', {encoding: 'base64'});

	let data = JSON.stringify({
		"token": "token",
		"data": {
			"id": "tfjs",
			"tag1":"ssd_mobilenet_v2",
			"tag2":"object_detection",
			"serialization_encoding":"base64",
			"model": [
				{
					"metadata":
					{
						"identifier":"tensorflowjs_model",
						"original_format":"pb"
					},
					"serialized_data": model
				},
				{
					"metadata":
					{
						"identifier":"model",
						"original_format":"json"
					},
					"serialized_data": architecture
				},
				{
					"metadata":
					{
						"identifier":"shard1of17",
						"original_format":"bin"
					},
					"serialized_data": shard1
				},
				{
					"metadata":
					{
						"identifier":"shard2of17",
						"original_format":"bin"
					},
					"serialized_data": shard2
				},
				{
					"metadata":
					{
						"identifier":"shard3of17",
						"original_format":"bin"
					},
					"serialized_data": shard3
				},
				{
					"metadata":
					{
						"identifier":"shard4of17",
						"original_format":"bin"
					},
					"serialized_data": shard4
				},
				{
					"metadata":
					{
						"identifier":"shard5of17",
						"original_format":"bin"
					},
					"serialized_data": shard5
				},
				{
					"metadata":
					{
						"identifier":"shard6of17",
						"original_format":"bin"
					},
					"serialized_data": shard6
				},
				{
					"metadata":
					{
						"identifier":"shard7of17",
						"original_format":"bin"
					},
					"serialized_data": shard7
				},
				{
					"metadata":
					{
						"identifier":"shard8of17",
						"original_format":"bin"
					},
					"serialized_data": shard8
				},
				{
					"metadata":
					{
						"identifier":"shard9of17",
						"original_format":"bin"
					},
					"serialized_data": shard9
				},
				{
					"metadata":
					{
						"identifier":"shard10of17",
						"original_format":"bin"
					},
					"serialized_data": shard10
				},
				{
					"metadata":
					{
						"identifier":"shard11of17",
						"original_format":"bin"
					},
					"serialized_data": shard11
				},
				{
					"metadata":
					{
						"identifier":"shard12of17",
						"original_format":"bin"
					},
					"serialized_data": shard12
				},
				{
					"metadata":
					{
						"identifier":"shard13of17",
						"original_format":"bin"
					},
					"serialized_data": shard13
				},
				{
					"metadata":
					{
						"identifier":"shard14of17",
						"original_format":"bin"
					},
					"serialized_data": shard14
				},
				{
					"metadata":
					{
						"identifier":"shard15of17",
						"original_format":"bin"
					},
					"serialized_data": shard15
				},
				{
					"metadata":
					{
						"identifier":"shard16of17",
						"original_format":"bin"
					},
					"serialized_data": shard16
				},
				{
					"metadata":
					{
						"identifier":"shard17of17",
						"original_format":"bin"
					},
					"serialized_data": shard17
				}
			],
			"weights":[
				{
					"metadata":
					{
						"identifier":"weights_manifest",
						"original_format":"json"
					},
					"serialized_data": weights
				}
			 ],
			"initialization":[ ],
			"checkpoints":[ ]
		}
	})
	
	return data;
}

function tflite(){
	let model = fs.readFileSync('./data/experiments/tflite/lite-model_efficientdet_lite0_detection_default_1.tflite', {encoding: 'base64'});
	let data = JSON.stringify({
		"token": "token",
		"data": {
			"id": "tflite",
			"tag1":"efficientdet",
			"tag2":"object_detection",
			"serialization_encoding":"base64",
			"model": [
				{
					"metadata":
					{
						"identifier":"lite-model_efficientdet_lite0_detection_default_1",
						"original_format":"tflite"
					},
					"serialized_data": model
				}
			],
			"weights":[ ],
			"initialization":[ ],
			"checkpoints":[ ]
		}
	})
	
	return data;
}

function test_model(size, name){
	var test_data = Buffer.allocUnsafe(size).fill('s').toString('utf8');
	

	let data = JSON.stringify({
		"token": "token",
		"data": {
			"id": "test_"+name,
			"tag1":"tag1",
			"tag2":"tag2",
			"serialization_encoding":"utf8",
			"model": [
				{
					"metadata":
					{
						"identifier":"test",
						"original_format":"test"
					},
					"serialized_data": test_data
				}
			],
			"weights":[ ],
			"initialization":[ ],
			"checkpoints":[ ]
		}
	})
	
	return data;
}

function submit(model_id, name){
	
	if (TEST === 'true'){
		var data = test_model(TEST_MODEL_SIZE, name);
	}else if (model_id === 'darknet'){
		var data = darknet();
	}else if (model_id === 'pytorch'){
		var data = pytorch();
	}else if (model_id === 'tfv2'){
		var data = tfv2();
	}else if (model_id === 'tfjs'){
		var data = tfjs();
	}else if (model_id === 'tflite'){
		var data = tflite();
	}	

	const options = {
	  hostname: '10.16.30.89',
	  port: 3000,
	  path: '/model/submit',
	  method: 'PUT',
	  headers: {
		'Content-Type': 'application/json',
		'Content-Length': data.length
	  }
	}

	const req = http.request(options, res => {
	  console.log(`statusCode: ${res.statusCode}`)

	  res.on('data', d => {
		console.log(d.toString());
	  })
	})

	req.on('error', error => {
	  console.error(error)
	})

	req.write(data)
	req.end()
}

function retrieve(model_id){

	const options = {
		hostname: '10.16.30.89',
		port: 3000,
		path: '/model?id='+model_id+'&token=token',
		method: 'GET'
	  }
	  
	  const req = http.request(options, res => {
		console.log(`statusCode: ${res.statusCode}`)
	  
		res.on('data', d => {
		  //process.stdout.write(d)
		})
	  })
	  
	  req.on('error', error => {
		console.error(error)
	  })
	  
	  req.end()

}

if (RUN === 'submit'){

	let i = 0;
	let a = new Date();
	while(i < AVG_COUNT){
		if (new Date().getTime() >= a.getTime()){
			submit(MODEL, i);
			a.setSeconds(a.getSeconds()+INTERVAL);
			console.log(a.getSeconds());
			i++;
		}
	}

	//submit(MODEL, '');
}else{
	retrieve(MODEL);
}


//node simulator.js submit test100 true 100 10 1
