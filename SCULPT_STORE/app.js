const crypto = require("crypto");
const moment = require("moment");
const secret = "abcdefg";
const hash = crypto
  .createHmac("sha256", secret)
  .update("I love cupcakes")
  .digest("hex");

function main() {
  console.log(hash);
  console.log(moment().format("MMMM Do YYYY, h:mm:ss a"));
}
main();
