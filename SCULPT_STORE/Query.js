require('node-go-require');
var DEFAULT_PAGE_SIZE=100
var GoSimpleChainCode = require('./chaincode/src/models/models.go');
var exported_obj=GoSimpleChainCode.SimpleChaincode.NewSimpleChaincode(new contractapi.Contract())
var trasnactionContext=GoSimpleChainCode.trasnactionContext;
var chaincodeStub=GoSimpleChainCode.chaincodeStub;
var stateQueryIterator=GoSimpleChainCode.stateQueryIteratorGoSimpleChainCode;
module.exports = {
    CreateClientToken: function(timestamp) {
        console.time('CreateClientToken');
        exported_obj.CreateClientToken(trasnactionContext,timestamp);
        console.timeEnd('CreateClientToken');
    
    },
    SubmitMeta: function(timestamp,server_state,sampled_train_data) {
        console.time('SubmitMeta');
        exported_obj.SubmitMeta(trasnactionContext,timestamp,server_state,sampled_train_data);
        console.timeEnd('SubmitMeta');
    
    },
    SubmitPage: function(timestamp,server_state,sampled_train_data) {
        console.time('SubmitPage');
        exported_obj.SubmitPage(trasnactionContext,timestamp,server_state,sampled_train_data);
        console.timeEnd('SubmitPage');
    
    },
    QueryMetaData: function(timestamp,server_state) {
        console.time('QueryMetaData');
        exported_obj.SubmitPage(trasnactionContext,timestamp,server_state);
        console.timeEnd('QueryMetaData');
    
    },
    QueryMetaDataWithPagination: function(timestamp,server_state) {
        console.time('MetaDataWithPagination');
        exported_obj.QueryMetaDataWithPagination(trasnactionContext,timestamp,server_state,DEFAULT_PAGE_SIZE,"");
        console.timeEnd('MetaDataWithPagination');
    
    },
    getMetaQueryResultForQueryStringWithPagination: function(timestamp,server_state) {
        console.time('MetaQueryResultForQueryStringWithPagination');
        exported_obj.getMetaQueryResultForQueryStringWithPagination(trasnactionContext,server_state,DEFAULT_PAGE_SIZE,"");
        console.timeEnd('MetaQueryResultForQueryStringWithPagination');
    
    },
};