const client = require("./client");


client.getAllNews({}, (error, response) => {
    console.log(error);
   // if (!error) throw error;
    console.log(response);
  });