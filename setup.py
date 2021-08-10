import os
import subprocess
from setuptools import setup


cwd = os.path.dirname(os.path.realpath(__file__))

__version_raw = subprocess.run(['git', 'describe', '--tags', '--long'],
                               stdout=subprocess.PIPE, cwd=cwd).stdout.decode('utf-8')[1:]
__versions = __version_raw.split('-')
__version__ = __versions[0] + '.' + __versions[1]

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
      tests_require=['pytest'],
      entry_points={
          'console_scripts': [
              'onnx2connx=onnx_connx.cli:main',
          ],
      })
