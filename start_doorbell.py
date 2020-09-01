import socket
import subprocess as sp
from pathlib import Path
import time

import yaml

from detection_code.door_camera import DoorCamera


def isOpen(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False


def main():
    config_path = Path("config.yaml")
    assert config_path.exists() and config_path.is_file()
    config = yaml.safe_load(config_path.open())
    host = config.get("server").get("host")
    port = config.get("server").get("port")
    motion_settings = config.get("motion_settings")
    try:
        t = sp.Popen(f"uvicorn --workers 1 --host {host} --port {port} face_recognition_server:server".split(),
                     cwd=f"{config_path.parent.absolute()}")
        door_camera = DoorCamera(host=host, port=port, motion_settings=motion_settings)
        while not isOpen(host, port):
            time.sleep(0.5)
        door_camera.run_camera()

    finally:
        sp.run("fuser -k 8000/tcp ", shell=True)


if __name__ == "__main__":
    main()
