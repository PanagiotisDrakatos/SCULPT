/*
    This file implements the Application Layer's REST server which communicates
    Triabase's Blockchain Network.
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

'use strict';
const express = require('express');
const HashMap = require('hashmap');
const cors = require('cors');
const compression = require('compression');
const Client = require('fabric-client');
const { exit } = require('process');
const bodyParser = require('body-parser');

const compressJSON = require('compress-json');  // compress-json
const cjson = require('compressed-json');       // compressed-json
const jsonpack = require('jsonpack/main')       // jsonpack
const { stringify, parse } = require('zipson'); // zipson


const chunker = require('buffer-chunks');
const MD5 = require('crypto-js/md5');


var app = express();
var submit_tx_map_meta = new HashMap();
var submit_tx_map_delete = new HashMap();
var submit_tx_map_cleanup = new HashMap();


var myArgs = process.argv.slice(2);
const COMPRESSION_ALGO = myArgs[0]; //compress-json | compressed-json | jsonpack | zipson
const CHUNK_SIZE = parseInt(myArgs[1]);
const MAX_CHUNK_SIZE = CHUNK_SIZE* 1024 * 1024; //MB -> bytes
const DEBUG  = myArgs[2]; // true | false
const AVG = myArgs[3]; //true | false
const AVG_COUNT = parseInt(myArgs[4]);
var AVG_COUNTER = AVG_COUNT;
var AVG_SUBMIT_LATENCY = 0;
var AVG_SUBMIT_PROCESSING_TIME = 0;

app.use(cors());
//app.use(compression());
app.use(express.json({limit: '1000mb', extended: true}));
app.use(express.urlencoded({limit: '1000mb', extended: true}));

//////////////////////////////////////////////BLOCKCHAIN CLIENT///////////////////////////////////////////////

const ORG_NAME = 'org1';
const USER_NAME = 'Admin';
const PEER_NAME = 'peer0.org1.example.com';
const CHANNEL_NAME = 'mychannel';
const CHAINCODE_ID = 'contract_models';
const DELETE_PAGE_SIZE = 10;
const DELETE_BOOKMARK = '';
var LAST = '';
//var START_TIMER = 0;


// Constants for profile
const CONNECTION_PROFILE_PATH = './profiles/dev-connect.yaml';

var CLIENT_CONNECTION_PROFILE_PATH;
// Client section configuration
if (ORG_NAME === 'org1') {
    CLIENT_CONNECTION_PROFILE_PATH = './profiles/org1-client.yaml';
} else if (ORG_NAME === 'org2') {
    CLIENT_CONNECTION_PROFILE_PATH = './profiles/org2-client.yaml';
} else {
    console.log("Exiting : Ivalid Organization name");
    exit(0);
}


// Variable to hold the client
var client = {}
// Variable to hold the channel
var channel = {}

async function submitMeta(tx_data, token, model_id) {

    let peerName = channel.getChannelPeer(PEER_NAME)

    var tx_id = client.newTransactionID();
    let tx_id_string = tx_id.getTransactionID();

    var request = {
        targets: peerName,
        chaincodeId: CHAINCODE_ID,
        fcn: 'SubmitMeta',
        args: [ token, model_id, JSON.stringify(tx_data)],
        chainId: CHANNEL_NAME,
        txId: tx_id
    };

    if (DEBUG === 'true'){
        console.log("#0 Transaction id is: " + tx_id_string)
        console.log("#1 Transaction proposal successfully sent to channel.")
    }
    try{
        let results = await channel.sendTransactionProposal(request);
        
        // Array of proposal responses
        var proposalResponses = results[0];

        var proposal = results[1];

        var all_good = true;
        for (var i in proposalResponses) {
            let good = false
            if (proposalResponses && proposalResponses[i].response &&
                proposalResponses[i].response.status === 200) {
                good = true;
                if (DEBUG === 'true'){
                    console.log(`\tChaincode invocation proposal response #${i} was good`);
                }
            } else {
                console.log(`\tChaincode invocation proposal response #${i} was bad!`);
            }
            all_good = all_good & good
        }
        if (DEBUG === 'true'){
            console.log("#2 Looped through the proposal responses all_good=", all_good)
        }

        await setupTxListener(tx_id_string, 'meta', model_id, new Date().getTime())//commented out for debugging
        if (DEBUG === 'true'){
            console.log('#3 Registered the Tx Listener')
        }

        var orderer_request = {
            txId: tx_id,
            proposalResponses: proposalResponses,
            proposal: proposal
        };

        await channel.sendTransaction(orderer_request);
        if (DEBUG === 'true'){
            console.log("#4 Transaction has been submitted.")
        }

    }catch{
        submit_tx_map_meta.set(model_id, {is_completed: 'true', status: 'UNAUTHORIZED', pending: -1});
        return
    }
    

}

async function submitPage(tx_data,  token, model_id, start_timer) {

    let peerName = channel.getChannelPeer(PEER_NAME)

    var tx_id = client.newTransactionID();
    let tx_id_string = tx_id.getTransactionID();

    var request = {
        targets: peerName,
        chaincodeId: CHAINCODE_ID,
        fcn: 'SubmitPage',
        args: [ token, model_id, JSON.stringify(tx_data)],
        chainId: CHANNEL_NAME,
        txId: tx_id
    };

    if (DEBUG === 'true'){
        console.log("#0 Transaction id is: " + tx_id_string)
        console.log("#1 Transaction proposal successfully sent to channel.")
    }
    try{
        let results = await channel.sendTransactionProposal(request);
        
        // Array of proposal responses
        var proposalResponses = results[0];

        var proposal = results[1];

        var all_good = true;
        for (var i in proposalResponses) {
            let good = false
            if (proposalResponses && proposalResponses[i].response &&
                proposalResponses[i].response.status === 200) {
                good = true;
                if (DEBUG === 'true'){
                    console.log(`\tChaincode invocation proposal response #${i} was good`);
                }
            } else {
                console.log(`\tChaincode invocation proposal response #${i} was bad!`);
            }
            all_good = all_good & good
        }
        if (DEBUG === 'true'){
            console.log("#2 Looped through the proposal responses all_good=", all_good)
        }

        await setupTxListener(tx_id_string, 'page', model_id, start_timer)//commented out for debugging
        if (DEBUG === 'true'){
            console.log('#3 Registered the Tx Listener')
        }

        var orderer_request = {
            txId: tx_id,
            proposalResponses: proposalResponses,
            proposal: proposal
        };

        await channel.sendTransaction(orderer_request);
        if (DEBUG === 'true'){
            console.log("#4 Transaction has been submitted.")
        }

    }catch(e){
        console.log(e);
        submit_tx_map_meta.set(model_id, {is_completed: 'true', status: 'UNAUTHORIZED', pending: -1});
        return
    }
    

}

async function getLastPages(page_size, bookmark,  token, model_id) {

    if (bookmark === "''" || bookmark === '""'){
        bookmark = '';
    }

    let peerName = channel.getChannelPeer(PEER_NAME)

    let request = {
         targets: peerName,
         chaincodeId: CHAINCODE_ID,
         fcn: 'QueryLatestModelWithPagination',
         args: [ token, model_id, page_size.toString(), bookmark]
     };

     // send the query proposal to the peer
    let response = await channel.queryByChaincode(request);
    
    return response.toString();

}

async function getMeta(token, model_id) {

    let peerName = channel.getChannelPeer(PEER_NAME)

    let request = {
         targets: peerName,
         chaincodeId: CHAINCODE_ID,
         fcn: 'QueryMetaData',
         args: [token, model_id]
     };

    // send the query proposal to the peer
    var response = await channel.queryByChaincode(request);
    
    return response.toString();
     
}

async function queryAdHocMeta(query_string, page_size, bookmark, token) {

    if (bookmark === "''" || bookmark === '""'){
        bookmark = '';
    }

    let peerName = channel.getChannelPeer(PEER_NAME)

    let request = {
         targets: peerName,
         chaincodeId: CHAINCODE_ID,
         fcn: 'QueryMetaDataWithPagination',
         args: [token, query_string, page_size.toString(), bookmark]
     };

     // send the query proposal to the peer
    let response = await channel.queryByChaincode(request);
    return response.toString();

}

async function getTag1Meta(tag1, page_size, bookmark, token){
    return await queryAdHocMeta('{"selector":{"tag1":"' + tag1 + '"}, "use_index":["_design/indexTag1", "indexTag1"]}', page_size, bookmark, token);
}
async function getTag2Meta(tag2, page_size, bookmark, token){
    return await queryAdHocMeta('{"selector":{"tag2":"' + tag2 + '"}, "use_index":["_design/indexTag2", "indexTag2"]}', page_size, bookmark, token);
}
async function getTag12Meta(tag1, tag2, page_size, bookmark, token){
    return await queryAdHocMeta('{"selector":{"$and":[{"tag1":"' + tag1 + '"},{"tag2":"' + tag2 + '"}]}, "use_index":["_design/indexTag12", "indexTag12"]}', page_size, bookmark, token);
}

async function deleteKeys(keys, token, model_id, type) {

    let peerName = channel.getChannelPeer(PEER_NAME)

    var tx_id = client.newTransactionID();
    let tx_id_string = tx_id.getTransactionID();

    var request = {
        targets: peerName,
        chaincodeId: CHAINCODE_ID,
        fcn: 'DeleteKeys',
        args: [token, keys.toString()],
        chainId: CHANNEL_NAME,
        txId: tx_id
    };

    if (DEBUG === 'true'){
        console.log("#0 Transaction id is: " + tx_id_string)
        console.log("#1 Transaction proposal successfully sent to channel.")
    }
    try{
        let results = await channel.sendTransactionProposal(request);
        
        // Array of proposal responses
        var proposalResponses = results[0];

        var proposal = results[1];

        var all_good = true;
        for (var i in proposalResponses) {
            let good = false
            if (proposalResponses && proposalResponses[i].response &&
                proposalResponses[i].response.status === 200) {
                good = true;
                if (DEBUG === 'true'){
                    console.log(`\tChaincode invocation proposal response #${i} was good`);
                }
            } else {
                console.log(`\tChaincode invocation proposal response #${i} was bad!`);
            }
            all_good = all_good & good
        }
        if (DEBUG === 'true'){
            console.log("#2 Looped through the proposal responses all_good=", all_good)
        }

        await setupTxListener(tx_id_string, type, model_id, new Date().getTime())
        if (DEBUG === 'true'){
            console.log('#3 Registered the Tx Listener')
        }

        var orderer_request = {
            txId: tx_id,
            proposalResponses: proposalResponses,
            proposal: proposal
        };

        await channel.sendTransaction(orderer_request);
        if (DEBUG === 'true'){
            console.log("#4 Transaction has been submitted.")
        }

    }catch(e){
        console.log(e);
        if (type === 'delete'){
            submit_tx_map_delete.set(model_id, {is_completed: 'true', status: 'UNAUTHORIZED'});
        }else if (type === 'cleanup'){
            submit_tx_map_cleanup.set(model_id, {is_completed: 'true', status: 'UNAUTHORIZED'});
        }
        
        return
    }
    

}

async function getKeys(queryString, page_size, bookmark,  token, model_id) {

    let peerName = channel.getChannelPeer(PEER_NAME)

    let request = {
         targets: peerName,
         chaincodeId: CHAINCODE_ID,
         fcn: 'GetKeys',
         args: [token, model_id, queryString, page_size.toString(), bookmark]
     };

    // send the query proposal to the peer
    var response = await channel.queryByChaincode(request);
    
    return response.toString();
     
}

async function getDeleteKeys(token, model_id){
    return getKeys(`{"selector":{"model_id":"` + model_id + `"}}`, DELETE_PAGE_SIZE, DELETE_BOOKMARK, token, model_id);
}

async function getCleanUpKeys(token, model_id) {

    let peerName = channel.getChannelPeer(PEER_NAME)

    let request = {
         targets: peerName,
         chaincodeId: CHAINCODE_ID,
         fcn: 'GetCleanUpKeys',
         args: [ token, model_id, DELETE_PAGE_SIZE.toString(), DELETE_BOOKMARK]
     };

    // send the query proposal to the peer
    var response = await channel.queryByChaincode(request);
    
    return response.toString();
     
}

async function checkToken(token) {

    let peerName = channel.getChannelPeer(PEER_NAME)

    let request = {
         targets: peerName,
         chaincodeId: CHAINCODE_ID,
         fcn: 'CheckClientToken',
         args: [token]
     };

     // send the query proposal to the peer
    let response = await channel.queryByChaincode(request);
    return response.toString();
}

async function setupTxListener(tx_id_string, type, model_id, start_timer) {
    
    try{
        let event_hub = channel.getChannelEventHub(PEER_NAME);

        event_hub.registerTxEvent(tx_id_string, (tx, code, block_num) => {
             
            if (DEBUG === 'true'){
                console.log("#5 Received Tx Event")
                console.log('The chaincode invoke chaincode transaction has been committed on peer %s', event_hub.getPeerAddr());
                console.log('Transaction %s is in block %s', tx, block_num);
            }
            
            if (code !== 'VALID') {
                if (type == "delete"){
                    submit_tx_map_delete.set(model_id, {is_completed: 'true', status: 'CORRUPTED'});
                }else if (type == "cleanup"){
                        submit_tx_map_cleanup.set(model_id, {is_completed: 'true', status: 'FAILED'});
                }else{
                    console.log('submit-INVALID', new Date().getTime() - start_timer);// the time it took before the corruption was detected
                    submit_tx_map_meta.set(model_id, {is_completed: 'true', status: 'CORRUPTED', pending: -1});
                }
            }

            if (type == "delete"){
                submit_tx_map_delete.set(model_id, {is_completed: 'true', status: 'VALID'});
            }if (type == "cleanup"){
                submit_tx_map_cleanup.set(model_id, {is_completed: 'true', status: 'VALID'});
            }else{
                let status = submit_tx_map_meta.get(model_id);
                if (status.status === 'PENDING'){
                    let pending = status.pending - 1;
                    if (pending === 0){
                        if (AVG === 'true'){
                            if (AVG_COUNTER === 0){
                                console.log('avg_submit_time_valid', AVG_SUBMIT_LATENCY/AVG_COUNT);
                            }else{
                                AVG_SUBMIT_LATENCY += new Date().getTime() - start_timer;
                            }
                        }else{
                            console.log('submit_time_valid', new Date().getTime() - start_timer);// the time it took to submit all the pages
                        }
                        
                        submit_tx_map_meta.set(model_id, {is_completed: 'true', status: 'VALID', pending: pending});
                    }else{
                        if (AVG === 'true'){
                            AVG_SUBMIT_LATENCY += new Date().getTime() - start_timer;
                        }
                        submit_tx_map_meta.set(model_id, {is_completed: 'false', status: 'PENDING', pending: pending});
                    }
                }
            }
            
            if (type == 'delete'){
                console.log(model_id, 'delete', submit_tx_map_delete.get(model_id))
            }else if (type == 'cleanup'){
                console.log(model_id, 'cleanup', submit_tx_map_cleanup.get(model_id))
            }else if (type === 'meta'){
                console.log(model_id+'_metadata', submit_tx_map_meta.get(model_id))
            }else{
                console.log(model_id, submit_tx_map_meta.get(model_id))
            }
            
            //for (const [key, value] of submit_tx_map_meta.entries()) {
            //  console.log(key, value);
            //}
            
        },
            // 3. Callback for errors
            (err) => {
                return JSON.stringify({});
                //console.log(err);
            },
            { unregister: true, disconnect: (true && (LAST === 'true')) }
        );

        event_hub.connect();
    }catch{
        throw new Error('Listener Error.')
    }
}

async function setupClient() {

    const client = Client.loadFromConfig(CONNECTION_PROFILE_PATH)

    client.loadFromConfig(CLIENT_CONNECTION_PROFILE_PATH)

    await client.initCredentialStores()
        .then((done) => {
            console.log("initCredentialStore(): ", done)
        })

    let userContext = await client.loadUserFromStateStore(USER_NAME)
    if (userContext == null) {
        throw new Error("User NOT found in credstore: " + USER_NAME)
    }

    client.setUserContext(userContext, true)

    return client
}

async function setupChannel() {

    
    try {
        channel = await client.getChannel(CHANNEL_NAME, true)
    } catch (e) {
        throw new Error("Could NOT create channel: " + CHANNEL_NAME)
    }

    return channel
}


/////////////////////////////////////////////////REST SERVER/////////////////////////////////////////////////////



app.get("/model", wrapAsync(async function(req, res, next){
    let start_timer = new Date().getTime();//milliseconds
    if (req.query.hasOwnProperty('token')){    
        
        try {
            if (req.query.hasOwnProperty('id')){

                let model = {};
                let page_counter = 0;
                let bookmark = '';
                
                let retrieve_preprocessing_time = new Date().getTime();
                let sub = 0;
                while(true){ //get all the data pages
                    try{
                        let start_timer2 = new Date().getTime();//milliseconds
                        let response = await getLastPages(1, bookmark, req.query.token, req.query.id);
                        sub += new Date().getTime() - start_timer2;
                        if (page_counter == 0 && response.includes('authorization')){
                            console.log('retrieve_time', new Date().getTime() - start_timer);
                            res.status(401).send({'message':'Unauthorized: authorization not granded: token "' + req.query.token
                                                    +'" is invalid'});
                            return
                        }
                        let page = JSON.parse(response);
                        model[page.records[0].Key] = page.records[0].Value;
                        bookmark = page.bookmark;
                        page_counter++;
                    }catch{
                        break;
                    }
                }
                var data = "";
                for (let i = 0; i < page_counter; i++){
                    let temp_digest = MD5(model[i].data).toString();
                    if (!(temp_digest === model[i].digest)){
                        console.log('retrieve_time', new Date().getTime() - start_timer);
                        res.status(500).send({'message':'Internal Server Error: Model data are corrupted.'});
                        return
                    }
                    data += model[i].data;
                }
                
                var recovered = {};

                if (COMPRESSION_ALGO === 'compress-json'){
                    recovered = compressJSON.decompress(JSON.parse(data));
                }else if (COMPRESSION_ALGO === 'compressed-json'){
                    recovered = cjson.decompress.fromString(data);
                }else if (COMPRESSION_ALGO === 'jsonpack'){
                    recovered = jsonpack.unpack(JSON.parse(data));
                }else if(COMPRESSION_ALGO === 'zipson'){
                    recovered = parse(JSON.parse(data));
                }else{
                    recovered = JSON.parse(data);
                }

                let response = await getMeta(req.query.token, req.query.id);
                let meta = JSON.parse(response);

                let ledger_entry = {
                    tag1: meta.tag1,
                    tag2: meta.tag2,
                    serialization_encoding: meta.serialization_encoding,
                    model: recovered.model,
                    weights: recovered.weights,
                    initialization: recovered.initialization,
                    checkpoints: recovered.checkpoints
                }
                
                console.log('retrieve_time_valid', new Date().getTime() - start_timer); //includes the time needed to get the data from the blockchain
                console.log('retrieve_preprocessing_time', new Date().getTime() - retrieve_preprocessing_time - sub);
                res.status(200).send(ledger_entry);
            }else{
                console.log('retrieve_time', new Date().getTime() - start_timer);
                res.status(400).send({'message':'Bad Request Error: Ensure valid query parameters are provided.'});
            }
        }catch(e){
            console.log(e);
            console.log('retrieve_time', new Date().getTime() - start_timer);
            res.status(404).send({'message':'Not Found: Model with id "' + req.query.id + '" does not exist.'});
            return
        }

    }else{
        console.log('retrieve_time', new Date().getTime() - start_timer);
        res.status(400).send({'message':'Bad Request Error: Ensure query parameter "token" is provided.'});
        return
    }

}));

app.put('/model/submit', wrapAsync(async function (req, res, next){
    AVG_COUNTER -= 1;
    let start_timer = new Date().getTime();//milliseconds
    if (!req.body.hasOwnProperty('token')){
        console.log('submit_time', new Date().getTime() - start_timer);
        res.status(400).send({'message':'Bad Request Error: Property "token" is missing from request body.'});
        return
    }
    

    if (!req.body.hasOwnProperty('data')){
        console.log('submit_time', new Date().getTime() - start_timer);
        res.status(400).send({'message':'Bad Request Error: Property "data" is missing from request body.'});
        return
    }
    if (!req.body.data.hasOwnProperty('id') || !req.body.data.hasOwnProperty('tag1')
        || !req.body.data.hasOwnProperty('tag2') || !req.body.data.hasOwnProperty('serialization_encoding')
        || !req.body.data.hasOwnProperty('model') || !req.body.data.hasOwnProperty('weights')
        || !req.body.data.hasOwnProperty('initialization') || !req.body.data.hasOwnProperty('checkpoints')){
            console.log('submit_time', new Date().getTime() - start_timer);
            res.status(400).send({'message':'Bad Request Error: Ensure all the required json properties are provided.'});
            return
    }
    
    // create metadata page|entry
    var Tx_Meta = {};
    Tx_Meta.model_id = req.body.data.id;
    Tx_Meta.tag1 = req.body.data.tag1;
    Tx_Meta.tag2 = req.body.data.tag2;
    Tx_Meta.serialization_encoding = req.body.data.serialization_encoding;
    

    // create data pages
    var Tx_Pages = {};
    Tx_Pages.model = req.body.data.model;
    Tx_Pages.weights = req.body.data.weights;
    Tx_Pages.initialization = req.body.data.initialization;
    Tx_Pages.checkpoints = req.body.data.checkpoints;

    var all_pages = '';

    console.log('uncompressed_data_size', Buffer.byteLength(JSON.stringify(Tx_Pages)));
    if (COMPRESSION_ALGO === 'compress-json'){
        all_pages = JSON.stringify(compressJSON.compress(Tx_Pages));
    }else if (COMPRESSION_ALGO === 'compressed-json'){
        all_pages = cjson.compress.toString(Tx_Pages);
    }else if (COMPRESSION_ALGO === 'jsonpack'){
        all_pages = JSON.stringify(jsonpack.pack(Tx_Pages));
    }else if(COMPRESSION_ALGO === 'zipson'){
        all_pages = JSON.stringify(stringify(Tx_Pages));
    }else{
        all_pages = JSON.stringify(Tx_Pages);
    }
    console.log('compressed_data_size', Buffer.byteLength(all_pages));

    
    var buf = Buffer.from(all_pages, 'utf8')
    if (CHUNK_SIZE === -1){
        var chunks = [all_pages];
    }else{
        var chunks = chunker(buf, MAX_CHUNK_SIZE);
    }
    console.log('page_number', chunks.length);// excludes the metadata entry
    var pages = [];
    for (let i = 0; i < chunks.length; i++){
        let Tx_Page = {
            model_id: req.body.data.id,
            page_id: i,
            page_bytes: chunks[i].length,
            data: chunks[i].toString(),
            digest: MD5(chunks[i].toString()).toString()
        }   
        pages.push(Tx_Page);
    }
    Tx_Meta.page_number = pages.length;
    if (AVG === 'true'){
        if (AVG_COUNTER === 0){
            console.log('avg_submit_preprocessing_time', AVG_SUBMIT_PROCESSING_TIME/AVG_COUNT);
        }else{
            AVG_SUBMIT_PROCESSING_TIME += new Date().getTime() - start_timer;
        }
    }else{
        console.log('submit_preprocessing_time', new Date().getTime() - start_timer); //does not include the submission to the blockchain delay
    }
    try {
        res.status(200).send({'message':'OK'});
        LAST = 'false';
        submit_tx_map_meta.set(req.body.data.id, {is_completed: 'false', status: 'PENDING', pending: pages.length+1});
        await submitMeta(Tx_Meta, req.body.token, req.body.data.id);
        for (let i = 0; i < pages.length; i++){
            if (i == pages.length-1){
                LAST = 'true';
                //START_TIMER = start_timer;
            }
            await submitPage(pages[i],req.body.token, req.body.data.id, start_timer);
        }
    }catch(e){
        console.log(e);
        submit_tx_map_meta.set(req.body.data.id, {is_completed: 'false', status: 'ERROR', pending: -1});
        if (AVG === 'true'){
            if (AVG_COUNTER === 0){
                console.log('avg_submit_time', AVG_SUBMIT_LATENCY/AVG_COUNT);
            }else{
                AVG_SUBMIT_LATENCY += new Date().getTime() - start_timer;
            }
        }else{
            console.log('submit_time', new Date().getTime() - start_timer);
        }        
        res.status(500).send({'message':'Internal Server Error: Failed to submit model ' + req.body.data.id + '.'});
        return
    }

}));

app.get('/model/submit/check', wrapAsync(async function(req, res, next){

    if (req.query.hasOwnProperty('token')){    
        
        try {
            let response = await checkToken(req.query.token);
            if (response.includes('false')){
                res.status(401).send({'message':'Unauthorized: authorization not granded: token "' + req.query.token
                                        +'" is invalid'});
                return
            }
            if (!req.query.hasOwnProperty('id')){
                res.status(400).send({'message':'Bad Request Error: Ensure query parameter "id" is provided.'});
                    return
            }

            
            if (!submit_tx_map_meta.has(req.query.id)){
                res.status(404).send({'message':'Not Found: Model with id "' + req.query.id + '" does not exist.'});
                return
            }
            res.status(200).send(submit_tx_map_meta.get(req.query.id));
        }catch(e){
            res.status(500).send({'message':'Internal Server Error: Failed to validate token "' + req.query.token + '".'});
            return
        }

    }else{
        res.status(400).send({'message':'Bad Request Error: Query parameter "token" is missing.'});
        return
    }

}));

app.get("/model/delete", wrapAsync(async function(req, res, next){
    if (req.query.hasOwnProperty('token')){    
        
        try {
            if (req.query.hasOwnProperty('id')){
                let response = await getDeleteKeys(req.query.token, req.query.id);
                console.log(response);
                if (response.includes('does not exist')){
                    submit_tx_map_delete.set(req.query.id, {is_completed: 'true', status: 'ERROR'});
                    res.status(404).send({'message':'Not Found: Model with id "' + req.query.id + '" does not exist.'});
                    return
                }
                if (response.includes('authorization')){
                    res.status(401).send({'message':'Unauthorized: authorization not granded: token "' + req.query.token
                                            +'" is invalid'});
                    return
                }  
                submit_tx_map_delete.set(req.query.id, {is_completed: 'false', status: 'PENDING'});
                await deleteKeys(response, req.query.token, req.query.id, 'delete');

                res.status(200).send({'message': 'OK'});
            }else{
                res.status(400).send({'message':'Bad Request Error: Ensure valid query parameters are provided.'});
            }
        }catch(e){
            submit_tx_map_delete.set(req.query.id, {is_completed: 'true', status: 'ERROR'});
            res.status(404).send({'message':'Not Found: Model with id "' + req.query.id + '" does not exist.'});
            return
        }
    }else{
        res.status(400).send({'message':'Bad Request Error: Ensure query parameter "token" is provided.'});
        return
    }
}));

app.get('/model/delete/check', wrapAsync(async function(req, res, next){

    if (req.query.hasOwnProperty('token')){    
        
        try {
            let response = await checkToken(req.query.token);
            if (response.includes('false')){
                res.status(401).send({'message':'Unauthorized: authorization not granded: token "' + req.query.token
                                        +'" is invalid'});
                return
            }
            if (!req.query.hasOwnProperty('id')){
                res.status(400).send({'message':'Bad Request Error: Ensure query parameter "id" is provided.'});
                    return
            }

            
            if (!submit_tx_map_delete.has(req.query.id)){
                res.status(404).send({'message':'Not Found: Delete record for model with id "' + req.query.id + '" does not exist.'});
                return
            }
            res.status(200).send(submit_tx_map_delete.get(req.query.id));
        }catch(e){
            res.status(500).send({'message':'Internal Server Error: Failed to validate token "' + req.query.token + '".'});
            return
        }

    }else{
        res.status(400).send({'message':'Bad Request Error: Query parameter "token" is missing.'});
        return
    }

}));

app.get("/model/cleanup", wrapAsync(async function(req, res, next){
    if (req.query.hasOwnProperty('token')){    
        
        try {
            if (req.query.hasOwnProperty('id')){
                let response = await getCleanUpKeys(req.query.token, req.query.id);
                if (response.length === 0){
                    submit_tx_map_cleanup.set(req.query.id, {is_completed: 'true', status: 'VALID'});
                    res.status(200).send({'message': 'OK'});
                    return;
                }
                if (response.includes('does not exist')){
                    submit_tx_map_cleanup.set(req.query.id, {is_completed: 'true', status: 'ERROR'});
                    res.status(404).send({'message':'Not Found: Model with id "' + req.query.id + '" does not exist.'});
                    return
                }
                if (response.includes('authorization')){
                    res.status(401).send({'message':'Unauthorized: authorization not granded: token "' + req.query.token
                                            +'" is invalid'});
                    return
                }  
                submit_tx_map_cleanup.set(req.query.id, {is_completed: 'false', status: 'PENDING'});
                await deleteKeys(response, req.query.token, req.query.id, 'cleanup');

                res.status(200).send({'message': 'OK'});
            }else{
                res.status(400).send({'message':'Bad Request Error: Ensure valid query parameters are provided.'});
            }
        }catch(e){
            submit_tx_map_cleanup.set(req.query.id, {is_completed: 'true', status: 'ERROR'});
            console.log(e);
            res.status(404).send({'message':'Not Found: Model with id "' + req.query.id + '" does not exist.'});
            return
        }
    }else{
        res.status(400).send({'message':'Bad Request Error: Ensure query parameter "token" is provided.'});
        return
    }
}));


app.get('/model/cleanup/check', wrapAsync(async function(req, res, next){

    if (req.query.hasOwnProperty('token')){    
        
        try {
            let response = await checkToken(req.query.token);
            if (response.includes('false')){
                res.status(401).send({'message':'Unauthorized: authorization not granded: token "' + req.query.token
                                        +'" is invalid'});
                return
            }
            if (!req.query.hasOwnProperty('id')){
                res.status(400).send({'message':'Bad Request Error: Ensure query parameter "id" is provided.'});
                    return
            }

            
            if (!submit_tx_map_cleanup.has(req.query.id)){
                res.status(404).send({'message':'Not Found: Cleanup record for model with id "' + req.query.id + '" does not exist.'});
                return
            }
            res.status(200).send(submit_tx_map_cleanup.get(req.query.id));
        }catch(e){
            res.status(500).send({'message':'Internal Server Error: Failed to validate token "' + req.query.token + '".'});
            return
        }

    }else{
        res.status(400).send({'message':'Bad Request Error: Query parameter "token" is missing.'});
        return
    }

}));

app.get("/metadata", wrapAsync(async function(req, res, next){
    if (req.query.hasOwnProperty('token')){    

        try {
            if (req.query.hasOwnProperty('id')){
                
                let response = await getMeta(req.query.token, req.query.id);
                if (response.includes('authorization')){
                    res.status(401).send({'message':'Unauthorized: authorization not granded: token ' + req.query.token
                                            +' is invalid'});
                    return
                }
                var ledger_entry = JSON.parse(response);
                res.status(200).send(ledger_entry);
            }else{
                res.status(400).send({'message':'Bad Request Error: Ensure valid query parameters are provided.'});
            }
        }catch(e){
            res.status(404).send({'message':'Not Found: Model with id "' + req.query.id + '" does not exist.'});
            return
        }

    }else{
        res.status(400).send({'message':'Bad Request Error: Ensure query parameter "token" is provided.'});
        return
    }
}));

app.get("/metadata/tags", wrapAsync(async function(req, res, next){

    if (req.query.hasOwnProperty('token')){    
        
        if (!req.query.hasOwnProperty('page_size') || !req.query.hasOwnProperty('bookmark')){        
            res.status(400).send({'message':'Bad Request Error: Ensure  valid "page_size" and "bookmark" query' +
                                    ' parameters are provided.'});
                return
        }

        if (!req.query.hasOwnProperty('tag1') || !req.query.hasOwnProperty('tag2')){               
            res.status(400).send({'message':'Bad Request Error: Ensure valid query parameters are provided.'});
            return
        }

        try {          
            var results = await getTag12Meta(req.query.tag1, req.query.tag2, req.query.page_size, req.query.bookmark, req.query.token);
            try {
                results = JSON.parse(results);
            }catch(e){
                if (results.includes('authorization')){
                    res.status(401).send({'message':'Unauthorized: authorization not granded: token "' + req.query.token
                                            +'" is invalid'});
                    return
                }         
                results = {records: [], fetchedRecordsCount: 0, bookmark: ""};
            }
            res.status(200).send(results);
        }catch(e){
            res.status(500).send({'message':'Internal Server Error: Failed to execute tag query.'});
            return
        }
    }else{
        res.status(400).send({'message':'Bad Request Error: Ensure query parameter "token" is provided.'});
        return
    }

}));

app.get("/metadata/tags/tag1", wrapAsync(async function(req, res, next){

    if (req.query.hasOwnProperty('token')){    
        
        if (!req.query.hasOwnProperty('page_size') || !req.query.hasOwnProperty('bookmark')){        
            res.status(400).send({'message':'Bad Request Error: Ensure  valid "page_size" and "bookmark" query' +
                                    ' parameters are provided.'});
                return
        }
        
    
        if (!req.query.hasOwnProperty('tag1')){               
            res.status(400).send({'message':'Bad Request Error: Ensure valid query parameters are provided.'});
            return
        }
        
        try {          
            var results = await getTag1Meta(req.query.tag1, req.query.page_size, req.query.bookmark, req.query.token);

            try {
                results = JSON.parse(results);
            }catch{
                if (results.includes('authorization')){
                    res.status(401).send({'message':'Unauthorized: authorization not granded: token "' + req.query.token
                                            +'" is invalid'});
                    return
                }         
                results = {records: [], fetchedRecordsCount: 0, bookmark: ""};
            }
            res.status(200).send(results);
        }catch(e){
            console.log(e);
            res.status(500).send({'message':'Internal Server Error: Failed to execute tag query.'});
            return
        }
    }else{
        res.status(400).send({'message':'Bad Request Error: Ensure query parameter "token" is provided.'});
        return
    }

}));

app.get("/metadata/tags/tag2", wrapAsync(async function(req, res, next){

    if (req.query.hasOwnProperty('token')){    

        if (!req.query.hasOwnProperty('page_size') || !req.query.hasOwnProperty('bookmark')){        
            res.status(400).send({'message':'Bad Request Error: Ensure  valid "page_size" and "bookmark" query' +
                                    ' parameters are provided.'});
                return
        }
        if (!req.query.hasOwnProperty('tag2')){               
            res.status(400).send({'message':'Bad Request Error: Ensure valid query parameters are provided.'});
            return
        }
        

        try {          
            var results = await getTag2Meta(req.query.tag2, req.query.page_size, req.query.bookmark, req.query.token);

            try {
                results = JSON.parse(results);
            }catch{
                if (results.includes('authorization')){
                    res.status(401).send({'message':'Unauthorized: authorization not granded: token "' + req.query.token
                                            +'" is invalid'});
                    return
                }         
                results = {records: [], fetchedRecordsCount: 0, bookmark: ""};
            }
            res.status(200).send(results);
        }catch(e){
            console.log(e);
            res.status(500).send({'message':'Internal Server Error: Failed to execute tag query.'});
            return
        }
    }else{
        res.status(400).send({'message':'Bad Request Error: Ensure query parameter "token" is provided.'});
        return
    }

}));

app.use(function(error, req, res, next) {
    // Any request to this server will get here
    console.log(error);
    res.end();
});

function wrapAsync(fn) {
    return function(req, res, next) {
      // Make sure to `.catch()` any errors and pass them along to the `next()`
      // middleware in the chain, in this case the error handler.
      fn(req, res, next).catch(next);
    };
  }

async function main() {
   
    //establish blockchain connection
    client = await setupClient();
    channel = await setupChannel();

    
    //start REST server
    var server = app.listen(3000, function () {
        var host = server.address().address;
        var port = server.address().port;
        console.log("Chunk size set to %dMB", CHUNK_SIZE);
        console.log("Example app listening at http://%s:%s", host, port);
    });
    
}

if (require.main === module){
    main();
}




//node --max-old-space-size=4096 server.js compress-json 10 false true 10