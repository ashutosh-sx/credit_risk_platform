const localtunnel = require('localtunnel');
const fs = require('fs');

(async () => {
  console.log("Starting localtunnel via Node.js API...");
  try {
    const tunnel = await localtunnel({ 
      port: 8501
    });

    console.log("FOUND SYSTEM URL:", tunnel.url);

    // Save the URL to tunnel_url.txt
    fs.writeFileSync("tunnel_url.txt", tunnel.url);
    console.log("Successfully wrote URL to tunnel_url.txt!");

    tunnel.on('close', () => {
      console.log("Tunnel closed.");
    });

    tunnel.on('error', (err) => {
      console.error("Tunnel error:", err);
    });

  } catch (err) {
    console.error("Error starting localtunnel:", err);
  }
})();
