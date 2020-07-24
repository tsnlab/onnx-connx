import subprocess
from setuptools import setup

__version__ = subprocess.run(['git', 'describe', '--tags', '--long'], stdout=subprocess.PIPE).stdout.decode('utf-8')[1:]

setup(name='onnx-connx',
      version=__version__,
      url='https://github.com/semihlab/onnx-connx',
      description='ONNX to CONNX converting tool',
      packages=['onnx_connx'],
      author='Semih Kim',
      author_email='semih.kim@gmail.com',
      install_requires=[
          'onnx>=1.7.0'
      ],
      entry_points={
          'console_scripts': [
              'onnx2connx=onnx_connx.cli:main',
          ],
      })
