from argparse import ArgumentTypeError
import logging
import os

import numpy as np
import tensorflow as tf

def fid_to_image(fid, gender_lbl, age_lbl, image_root, image_size):
    """ Loads and resizes an image given by FID. Pass-through the PID. """
    # Since there is no symbolic path.join, we just add a '/' to be sure.
    image_encoded = tf.read_file(tf.reduce_join([image_root, '/', fid]))

    # tf.image.decode_image doesn't set the shape, not even the dimensionality,
    # because it potentially loads animated .gif files. Instead, we use either
    # decode_jpeg or decode_png, each of which can decode both.
    # Sounds ridiculous, but is true:
    # https://github.com/tensorflow/tensorflow/issues/9356#issuecomment-309144064
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
    image_resized = tf.image.resize_images(image_decoded, image_size)
    image_resized = (tf.to_float(image_resized) - 127.5) / 128
    return image_resized, fid, gender_lbl, age_lbl

def load_dataset(csv_file, image_root, fail_on_missing=True):
    """ Loads a dataset .csv file, returning PIDs and FIDs.

    PIDs are the "person IDs", i.e. class names/labels.
    FIDs are the "file IDs", which are individual relative filenames.

    Args:
        csv_file (string, file-like object): The csv data file to load.
        image_root (string): The path to which the image files as stored in the
            csv file are relative to. Used for verification purposes.
            If this is `None`, no verification at all is made.
        fail_on_missing (bool or None): If one or more files from the dataset
            are not present in the `image_root`, either raise an IOError (if
            True) or remove it from the returned dataset (if False).

    Returns:
        (pids, fids) a tuple of numpy string arrays corresponding to the PIDs,
        i.e. the identities/classes/labels and the FIDs, i.e. the filenames.

    Raises:
        IOError if any one file is missing and `fail_on_missing` is True.
    """
    dataset = np.genfromtxt(csv_file, delimiter=',', dtype='|U')
    gender_lbls, age_lbls, fids = dataset.T
    gender_lbls = np.int32(gender_lbls)
    age_lbls = np.int32(age_lbls)

    # Possibly check if all files exist
    if image_root is not None:
        missing = np.full(len(fids), False, dtype=bool)
        for i, fid in enumerate(fids):
            missing[i] = not os.path.isfile(os.path.join(image_root, fid))

        missing_count = np.sum(missing)
        if missing_count > 0:
            if fail_on_missing:
                raise IOError('Using the `{}` file and `{}` as an image root {}/'
                            '{} images are missing'.format(
                                csv_file, image_root, missing_count, len(fids)))
            else:
                print('[Warning] removing {} missing file(s) from the'
                    ' dataset.'.format(missing_count))
                # We simply remove the missing files.
                fids = fids[np.logical_not(missing)]
                gender_lbls = gender_lbls[np.logical_not(missing)]
                age_lbls = age_lbls[np.logical_not(missing)]

    return gender_lbls, age_lbls, fids


def get_logging_dict(name):
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'stderr': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'common.ColorStreamHandler',
                'stream': 'ext://sys.stderr',
            },
            'logfile': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': name + '.log',
                'mode': 'a',
            }
        },
        'loggers': {
            '': {
                'handlers': ['stderr', 'logfile'],
                'level': 'DEBUG',
                'propagate': True
            },

            # extra ones to shut up.
            'tensorflow': {
                'handlers': ['stderr', 'logfile'],
                'level': 'INFO',
            },
        }
    }

class _AnsiColorStreamHandler(logging.StreamHandler):
    DEFAULT = '\x1b[0m'
    RED     = '\x1b[31m'
    GREEN   = '\x1b[32m'
    YELLOW  = '\x1b[33m'
    CYAN    = '\x1b[36m'

    CRITICAL = RED
    ERROR    = RED
    WARNING  = YELLOW
    INFO     = DEFAULT  # GREEN
    DEBUG    = CYAN

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:  return cls.CRITICAL
        elif level >= logging.ERROR:   return cls.ERROR
        elif level >= logging.WARNING: return cls.WARNING
        elif level >= logging.INFO:    return cls.INFO
        elif level >= logging.DEBUG:   return cls.DEBUG
        else:                          return cls.DEFAULT

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)

    def format(self, record):
        text = logging.StreamHandler.format(self, record)
        color = self._get_color(record.levelno)
        return (color + text + self.DEFAULT) if self.is_tty() else text

    def is_tty(self):
        isatty = getattr(self.stream, 'isatty', None)
        return isatty and isatty()


class _WinColorStreamHandler(logging.StreamHandler):
    # wincon.h
    FOREGROUND_BLACK     = 0x0000
    FOREGROUND_BLUE      = 0x0001
    FOREGROUND_GREEN     = 0x0002
    FOREGROUND_CYAN      = 0x0003
    FOREGROUND_RED       = 0x0004
    FOREGROUND_MAGENTA   = 0x0005
    FOREGROUND_YELLOW    = 0x0006
    FOREGROUND_GREY      = 0x0007
    FOREGROUND_INTENSITY = 0x0008 # foreground color is intensified.
    FOREGROUND_WHITE     = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED

    BACKGROUND_BLACK     = 0x0000
    BACKGROUND_BLUE      = 0x0010
    BACKGROUND_GREEN     = 0x0020
    BACKGROUND_CYAN      = 0x0030
    BACKGROUND_RED       = 0x0040
    BACKGROUND_MAGENTA   = 0x0050
    BACKGROUND_YELLOW    = 0x0060
    BACKGROUND_GREY      = 0x0070
    BACKGROUND_INTENSITY = 0x0080 # background color is intensified.

    DEFAULT  = FOREGROUND_WHITE
    CRITICAL = BACKGROUND_YELLOW | FOREGROUND_RED | FOREGROUND_INTENSITY | BACKGROUND_INTENSITY
    ERROR    = FOREGROUND_RED | FOREGROUND_INTENSITY
    WARNING  = FOREGROUND_YELLOW | FOREGROUND_INTENSITY
    INFO     = FOREGROUND_GREEN
    DEBUG    = FOREGROUND_CYAN

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:  return cls.CRITICAL
        elif level >= logging.ERROR:   return cls.ERROR
        elif level >= logging.WARNING: return cls.WARNING
        elif level >= logging.INFO:    return cls.INFO
        elif level >= logging.DEBUG:   return cls.DEBUG
        else:                          return cls.DEFAULT

    def _set_color(self, code):
        import ctypes
        ctypes.windll.kernel32.SetConsoleTextAttribute(self._outhdl, code)

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)
        # get file handle for the stream
        import ctypes, ctypes.util
        # for some reason find_msvcrt() sometimes doesn't find msvcrt.dll on my system?
        crtname = ctypes.util.find_msvcrt()
        if not crtname:
            crtname = ctypes.util.find_library("msvcrt")
        crtlib = ctypes.cdll.LoadLibrary(crtname)
        self._outhdl = crtlib._get_osfhandle(self.stream.fileno())

    def emit(self, record):
        color = self._get_color(record.levelno)
        self._set_color(color)
        logging.StreamHandler.emit(self, record)
        self._set_color(self.FOREGROUND_WHITE)

# select ColorStreamHandler based on platform
import platform
if platform.system() == 'Windows':
    ColorStreamHandler = _WinColorStreamHandler
else:
    ColorStreamHandler = _AnsiColorStreamHandler
