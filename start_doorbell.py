import subprocess as sp
import time
from pathlib import Path

import yaml

from detection_code.door_camera import DoorCamera


def main():
    config_path = Path("config.yaml")
    assert config_path.exists() and config_path.is_file()
    config = yaml.load(config_path.open())
    host = config.get("server").get("host")
    port = config.get("server").get("port")
    motion_settings = config.get("motion_settings")
    try:
        t = sp.Popen(f"uvicorn --workers 1 --host {host} --port {port} face_recognition_server:server", shell=True,
                     cwd="./detection_code")
        door_camera = DoorCamera(host=host, port=port, motion_settings=motion_settings)
        time.sleep(3)
        door_camera.run_camera()

    finally:
        sp.run("fuser -k 8000/tcp ", shell=True)


if __name__ == "__main__":
    main()
