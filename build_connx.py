import shutil
import subprocess

from os import path

PATH = path.join(path.dirname(__file__), "connx/ports/linux")

subprocess.run("cmake . -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release", cwd="connx/ports/linux", shell=True)
subprocess.run("cmake --build build", cwd="connx/ports/linux", shell=True)

# Remove this part and use cmake install prefix
shutil.copy(path.join("connx/ports/linux", "build", "connx"), "onnx_connx/connx")
