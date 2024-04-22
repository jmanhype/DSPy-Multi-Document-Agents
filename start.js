const { exec } = require("child_process");
const url = "http://localhost:5656";

setTimeout(() => {
  const platform = process.platform;
  let command;

  if (platform === "darwin") {
    command = "open";
  } else if (platform === "win32") {
    command = "start";
  } else {
    command = "xdg-open";
  }

  console.log("Executing...");
  const child = exec(command + " " + url, {detached: true});
  child.unref();
}, 8000);
