from .opset import version as default_version
from .opset import opset as default_opset
from .opset import attrset as default_attrset

_versions = {
    '': default_version
}

_opsets = {
    '': default_opset
}

_attrsets = {
    '': default_attrset
}

# specs: [ { domain: str, version: int } ]
def _check(specs):
    for spec in specs:
        domain = spec['domain']
        version = spec['version']

        if domain not in _versions:
            raise Exception('There is no such opset domain: {}, version: {}'.format(domain, version))

        if _versions[domain] < version:
            raise Exception('The opset version is lower than the expected one: opset domain: {}, version: {}, expected: {}'.format(domain, _versions[domain], version))

def get_opset(specs):
    _check(specs)

    opset = { }

    for spec in specs:
        domain = spec['domain']
        version = spec['version']

        opsets = _opsets[domain]

        for key, value in opsets.items():
            opset[key] = value

    return opset

def get_attrset(specs):
    _check(specs)

    attrset = { }

    for spec in specs:
        domain = spec['domain']
        version = spec['version']

        attrsets = _attrsets[domain]

        for key, value in attrsets.items():
            attrset[key] = value

    return attrset


__all__ = [ get_opset, get_attrset ]
