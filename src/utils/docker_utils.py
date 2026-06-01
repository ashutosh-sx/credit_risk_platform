import os
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger("docker_utils")

def is_running_in_docker() -> bool:
    """
    Detects if the application is currently running inside a Docker container.
    """
    # Check for .dockerenv file in root
    if Path("/.dockerenv").exists():
        return True
        
    # Check cgroup signatures
    try:
        with open("/proc/1/cgroup", "rt") as f:
            if "docker" in f.read():
                return True
    except Exception:
        pass
        
    return False

def resolve_data_path(file_name: str) -> str:
    """
    Resolves data paths dynamically. Under Docker, raw data might be mounted under
    /data, whereas locally it could be in the relative ./data folder.
    """
    if is_running_in_docker():
        # Inside docker, volume is mounted to root level /data
        docker_path = Path("/data") / file_name
        if docker_path.exists():
            return str(docker_path)
            
    # Default local lookup fallback
    local_path = Path(__file__).resolve().parents[2] / "data" / file_name
    return str(local_path)
