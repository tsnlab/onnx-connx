import configparser
import os

class Config:
    """
    CLI config
    :param str output_path: output directory path
    :param bool is_output_comment: To output comment to cnx file or not
    :param str endian: 'little' or 'big' endian
    :param str align: 'self_align' or 'static_align'
    :param int align_size: align size when 'static_align' is selected
    :param str padding: 'self_padding' or 'static_padding'
    :param int padding_size: padding size when 'static_padding' is selected
    """

    def __init__(self):
        self.output_path = 'out'
        self.output_comment = False
        self.endian = 'little'
        self.align = 'self_align'
        self.align_size = 4
        self.padding = 'static_padding'
        self.padding_size = 4

    def read(self, path):
        with open(path, 'r') as f:
            config_string = '[settings]\n' + f.read()
        config = configparser.ConfigParser()
        config.read(config_string)
        c = config['settings']

        if 'output_path' in c:
            value = c['output_path']
            self.set_output_path(value)

        if 'output_comment' in c:
            value = c['output_comment']
            self.set_output_comment(value)

        if 'endian' in c:
            value = c['endian']
            self.set_endian(value)

        if 'align' in c:
            value = c['align']
            self.set_align(value)

        if self.align == 'static_align' and 'align_size' in c:
            value = c['align_size']
            self.set_align_size(value)

        if 'padding' in c:
            value = c['padding']
            self.set_padding(value)

        if self.padding == 'static_padding' and 'padding_size' in c:
            value = c['padding_size']
            self.set_padding_size(value)

    def set_output_path(self, value):
        self.output_path = value

    def make_output_path(self):
        try:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
        except OSError:
            print('Cannot make output directory:', self.output_path)

    def set_output_comment(self, value):
        self.is_output_comment = value == 'true' or value == 'True'

    def set_endian(self, value):
        if value == 'little':
            self.endian = 'little'
        elif value == 'big':
            self.endian = 'big'

    def set_align(self, value):
        if value == 'self_align':
            self.align = 'self_align'
        elif value == 'size_align':
            self.align = 'static_align'

    def set_align_size(self, value):
        self.align_size = int(value)

    def set_padding(self, value):
        if value == 'self_padding':
            self.padding = 'self_padding'
        elif value == 'static_padding':
            self.padding = 'static_padding'

    def set_padding_size(self, value):
        self.padding_size = int(value)

    def get_output_dir(path):

        return path

    def alignof(self, offset, size):
        if self.align == 'self_align':
            if size == 0:
                size = 4
        elif self.align == 'static_align':
            size = self.align_size

        return (size - (offset % size)) % size

    def padof(self, offset, size):
        if self.padding == 'self_padding':
            if size == 0:
                size = 4
        elif self.padding == 'static_padding':
            size = self.padding_size

        if size == 0:
            size = 4

        size = 4
        return (size - (offset % size)) % size

