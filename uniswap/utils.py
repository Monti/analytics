import logging
import time


def topic_from_address(address: str) -> str:
    return '0x' + '0' * 24 + address[2:]


def bytes_to_str(x):
    return x.decode().strip('\x00')


def int_to_str(x):
    return bytes_to_str(x.to_bytes((x.bit_length() + 7) // 8, 'big'))


def timeit(foo):
    def wrapper_foo(*args, **kwargs):
        t = time.time()
        ret = foo(*args, **kwargs)
        t = time.time() - t
        logging.info('{} finished in {:.2f}s'.format(foo.__name__, t))
        return ret

    return wrapper_foo
