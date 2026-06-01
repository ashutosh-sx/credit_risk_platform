import subprocess
import time
import sys
import os

def start_tunnel():
    print("Starting localtunnel process...")
    # Start localtunnel subprocess in real-time
    process = subprocess.Popen(
        ["npx", "-y", "-p", "localtunnel", "lt", "--port", "8501"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    
    print("Reading stdout...")
    url = None
    
    # Read stdout line-by-line in real-time
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print("STDOUT:", line.strip())
        if "your url is" in line.lower():
            url = line.strip().split("is:")[-1].strip()
            print("FOUND SYSTEM URL:", url)
            
            # Save the URL to a local text file
            with open("tunnel_url.txt", "w") as f:
                f.write(url)
            print("Successfully wrote URL to tunnel_url.txt!")
            break
            
    # Keep running in background
    if url:
        print("Tunnel is established! Keeping process active in background...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            process.terminate()
    else:
        err = process.stderr.read()
        print("Stderr errors:", err)

if __name__ == "__main__":
    start_tunnel()
