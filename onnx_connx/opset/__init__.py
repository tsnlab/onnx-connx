from .opset_6 import opset as opset_6
from .opset_6 import attribute as attribute_6
from .opset_18 import opset as opset_18
from .opset_18 import attribute as attribute_18

_default_opsets = {
    'min': 1,
    'max': 18,
    6: opset_6,
    18: opset_18
}

_default_attributes = {
    'min': 1,
    'max': 18,
    6: attribute_6,
    18: attribute_18
}

_opsets = {
    '': _default_opsets
}

_attributes = {
    '': _default_attributes
}

# specs: [ { domain: str, version: int } ]
def get_opset(specs):
    opset = { }

    for spec in specs:
        opsets = _opsets[spec['domain']]

        for i in range(opsets['min'], opsets['max'] + 1):
            if i in opsets:
                for key, value in opsets[i].items():
                    opset[key] = value

    return opset

def get_attribute(specs):
    attribute = { }

    for spec in specs:
        attributes = _attributes[spec['domain']]

        for i in range(attributes['min'], attributes['max'] + 1):
            if i in attributes:
                for key, value in attributes[i].items():
                    attribute[key] = value

    return attribute


__all__ = [ get_opset, get_attribute ]
