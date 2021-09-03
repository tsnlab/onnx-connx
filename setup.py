import os
import subprocess
import setuptools


# version
if os.path.isdir('.git'):
    cwd = os.path.dirname(os.path.realpath(__file__))

    __version_raw = subprocess.run(['git', 'describe', '--tags', '--long'],
                                   stdout=subprocess.PIPE, cwd=cwd).stdout.decode('utf-8')[1:]
    __versions = __version_raw.split('-')
    __version__ = __versions[0] + '.' + __versions[1]

    with open('VERSION', 'w') as f:
        f.write(__version__)
else:
    with open('VERSION', 'r') as f:
        __version__ = f.read()

# description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# setup
setuptools.setup(name='onnx-connx',
                 version=__version__,
                 author='Semih Kim',
                 author_email='semih.kim@gmail.com',
                 description='ONNX to CONNX converting tool',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 url='https://github.com/semihlab/onnx-connx',
                 project_urls={
                     'Bug Tracker': 'https://github.com/semihlab/onnx-connx/issues',
                 },
                 classifiers=[
                     'Environment :: Console',
                     'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
                     'Topic :: Software Development :: Build Tools',
                 ],
                 package_dir={'': '.'},
                 packages=setuptools.find_packages(where='.'),
                 install_requires=[
                     'onnx==1.9.0'
                 ],
                 tests_require=['pytest'],
                 entry_points={
                     'console_scripts': [
                         'onnx2connx=onnx_connx.cli:main',
                     ],
                 })
