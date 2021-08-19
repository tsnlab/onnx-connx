# onnx-connx contribution guidelines
Welcome to onnx-connx project and very welcome your contirbutions.

 1. Join CONNX community. [![Gitter](https://badges.gitter.im/c-onnx/community.svg)](https://gitter.im/c-onnx/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
 2. Don't hesitate to send a message to maintainer(@semihlab)
 3. If you are interested in contributing code, please follow contribution guilde lines below

# How to add new operator
 1. Add the code to opset.py directly if the implementations is few lines.
    Or Make a [Operator].py file in opset directory if the implementation is big.
 2. Import the function in head of opset.py.
 3. Add the function to opset dictionary.
 4. Add [min input count, max input count] array to argcount dictionary.
 5. Add default attributes to attrset dictionary.
 6. Run ONNX test cases(for particular operators).

```sh
onnx-connx$ pytest -k [test case]
```
 
    You can find ONNX test cases using find utility as below.

```sh
    ONNX_HOME=`python3 -c 'import onnx; print(onnx.__path__[0])'`
    find ${ONNX_HOME}/backend/test/ -name 'model.onnx'
```

    /home/semih/venv/lib/python3.x/site-packages/onnx/backend/test/data/node/test_add_uint8/model.onnx  
                                                This is the 'test case' name ^^^^^^^^^^^^^^
 7. Run full ONNX test cases. It will take about 30 mins under AMD Ryzen Threadripper 1900X.

```sh
onnx-connx$ make test
```

 8. Taking 30 mins is a BUG. I hope someone to fix it.

# How to contribute code
 1. Pass all the onnx test cases
 2. Check python lint (flake8 is required)

```sh
onnx-connx$ bin/lint
```

 3. Register lint to git commit hook (optional)

```sh
$ cp bin/lint .git/hooks/pre-commit                # Register Python lint
```

 4. If it's your first pull request, github will require to agree CLA, please agree it.
 5. Pull request to maintainer(@semihlab)

# Some tips
 * We don't care about training parameters. e.g. running_mean, running_var of BatchNormalization.
 * You can get the sample Numpy implementation from Operators.md document in ONNX project.
