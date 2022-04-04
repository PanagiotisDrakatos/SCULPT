const grpc = require("@grpc/grpc-js");
const PROTO_PATH = "./federated.proto";
var protoLoader = require("@grpc/proto-loader");

const options = {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
};
var packageDefinition = protoLoader.loadSync(PROTO_PATH, options);
const newsProto = grpc.loadPackageDefinition(packageDefinition);
const {CreateClientToken, SubmitMeta,SubmitPage, QueryMetaData, QueryMetaDataWithPagination,getMetaQueryResultForQueryStringWithPagination}  = require('./Query.js');
const server = new grpc.Server();
var news =[
    { id: "1", title: "Note 1", body: "Content 1", postImage: "Post image 1" },
    { id: "2", title: "Note 2", body: "Content 2", postImage: "Post image 2" },
  ];

server.addService(newsProto.FederatedService.service, {
  Search: (_, callback) => {
    const timestamp=_.request.timestamp;
    const round=_.request.round;
    const server_state = _.request.server_state;
    const sampled_train_data = _.request.sampled_train_data;
    const clients_participated = _.request.clients_participated;
    const broadcasted_bits = _.request.broadcasted_bits;
    const aggregated_bits = _.request.aggregated_bits;

    console.log("Timestamp:"+timestamp);
    console.log("round:"+round);
    console.log("server_state:"+server_state);
    console.log("sampled_train_data:"+sampled_train_data);
    console.log("clients_participated:"+clients_participated);
    console.log("broadcasted_bits:"+broadcasted_bits);
    console.log("aggregated_bits:"+aggregated_bits);
    callback(null, {response: 'Server Received'});
  },
  getAllNews: (_, callback) => {
    const timestamp=_.request.timestamp;
    const round=_.request.round;
    const server_state = _.request.server_state;
    const sampled_train_data = _.request.sampled_train_data;
    const clients_participated = _.request.clients_participated;
    const broadcasted_bits = _.request.broadcasted_bits;
    const aggregated_bits = _.request.aggregated_bits;

    CreateClientToken(timestamp);
    SubmitMeta(timestamp,server_state,sampled_train_data)
    SubmitPage(timestamp,server_state)
    QueryMetaData(timestamp,server_state);
    QueryMetaDataWithPagination(timestamp,server_state)
    getMetaQueryResultForQueryStringWithPagination(timestamp,server_state);
    callback(null,news);
  },
  getNews: (_, callback) => {
    const newsId = _.request.id;
    const newsItem = news.find(({ id }) => newsId == id);
    callback(null, newsItem);
  },
  deleteNews: (_, callback) => {
    const newsId = _.request.id;
   // news = news.filter(({ id }) => id !== newsId);
    callback(null, {});
  },
  editNews: (_, callback) => {
    const newsId = _.request.id;
    const newsItem = news.find(({ id }) => newsId == id);
    newsItem.body = _.request.body;
    newsItem.postImage = _.request.postImage;
    newsItem.title = _.request.title;
    callback(null, newsItem);
  },
  addNews: (call, callback) => {
    let _news = { id: Date.now(), ...call.request };
    news.push(_news);
    console.log(call.title);
    callback(null, _news);
  },
});
//const {method, otherMethod}  = require('./Query.js');
//console.log(method("asda"));
server.bindAsync(
  "127.0.0.1:50051",
  grpc.ServerCredentials.createInsecure(),
  (error, port) => {
    console.log("Server at port:", port);
    console.log("Server running at http://127.0.0.1:50051");
    server.start();
  }
);