import re
from importlib import resources


# domain: List[version]
_versions = {}

# domain_version: attrset
_attrsets = {}

# domain_version: version
_versets = {}

# find attrsets
opset_list = resources.contents('onnx_connx.opset')
for name in opset_list:
    match = re.match(r'opset_(.*)_(\d+)\.py', name)
    if match is not None:
        domain = match[1]
        version = int(match[2])

        if domain not in _versions:
            _versions[domain] = [version]
        else:
            _versions[domain].append(version)

        opset = __import__(f'onnx_connx.opset.opset_{domain}_{version}', fromlist=['onnx_connx.opset'])
        attrset = opset.attrset
        _attrsets[f'{domain}_{version}'] = attrset

        verset = {}
        for op_type in attrset:
            verset[op_type] = version
        _versets[f'{domain}_{version}'] = verset


# sort versions
for domain in _versions:
    _versions[domain].sort()

# merge attrsets and versets
for domain in _versions:
    old_verset = {}
    old_attrset = {}

    for version in _versions[domain]:
        attrset = _attrsets[f'{domain}_{version}']
        verset = _versets[f'{domain}_{version}']

        for op_type in old_attrset:
            if op_type not in attrset:
                attrset[op_type] = old_attrset[op_type]
                verset[op_type] = old_verset[op_type]

        old_attrset = attrset
        old_verset = verset


# specs: [ { domain: str, version: int } ]
def _check(specs):
    for spec in specs:
        domain = spec['domain']
        version = spec['version']

        if domain not in _versions:
            raise Exception(f'Domain {domain} is not supported yet')

        if _versions[domain][-1] < version:
            raise Exception('The opset version is lower than the expected one: opset domain: {}, version: {}, '
                            'expected: {}'.format(domain, _versions[domain][-1], version))


def find_version(domain, version):
    for i in range(len(_versions[domain])):
        v = _versions[domain][i]

        if v < version:
            continue
        else:
            return v


def get_attrset(specs):
    _check(specs)

    attrset = {}
    verset = {}

    for spec in specs:
        domain = spec['domain']
        version = spec['version']

        version = find_version(domain, version)

        attrsets = _attrsets[f'{domain}_{version}']
        versets = _versets[f'{domain}_{version}']

        for key, value in attrsets.items():
            attrset[key] = value

        for key, value in versets.items():
            verset[key] = value

    return attrset, verset


__all__ = [get_attrset]
