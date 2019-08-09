import json
import os
from multiprocessing.pool import ThreadPool
from web3 import Web3

from custom_http_provider import CustomHTTPProvider

NODE_ADDRESS = 'http://127.0.0.1:8545'
DEFAULT_REQUEST_TIMEOUT = 30

web3 = Web3(CustomHTTPProvider(endpoint_uri=NODE_ADDRESS, request_kwargs={'timeout': DEFAULT_REQUEST_TIMEOUT}))
for each in ['request_param_normalizer', 'name_to_address', 'validation', 'abi']:
    web3.middleware_stack.remove(each)
web3_infura = web3

ETH = 10 ** 18

MAX_FEE = 10000

UNISWAP_BEGIN_BLOCK = 1775445

HISTORY_BEGIN_BLOCK = 1775000

HISTORY_CHUNK_SIZE = 5000

REORG_PROTECTION_BLOCKS_COUNT = 50

CURRENT_BLOCK = web3.eth.blockNumber - REORG_PROTECTION_BLOCKS_COUNT

LOGS_BLOCKS_CHUNK = 5000

THREADS = 8

pool = ThreadPool(THREADS)

with open('abi/uniswap_factory.abi') as in_f:
    UNISWAP_FACTORY_ABI = json.load(in_f)

with open('abi/uniswap_exchange.abi') as in_f:
    UNISWAP_EXCHANGE_ABI = json.load(in_f)

with open('abi/erc_20.abi') as in_f:
    ERC_20_ABI = json.load(in_f)

with open('abi/str_erc_20.abi') as in_f:
    STR_ERC_20_ABI = json.load(in_f)

with open('abi/str_caps_erc_20.abi') as in_f:
    STR_CAPS_ERC_20_ABI = json.load(in_f)

UNISWAP_FACTORY_ADDRESS = '0x6A662F91E14312a11a2E35b359427AEf798fD928'

uniswap_factory = web3.eth.contract(abi=UNISWAP_FACTORY_ABI, address=UNISWAP_FACTORY_ADDRESS)

HARDCODED_INFO = {
}

DIST_DIR = '../dist/uniswap/'

LIQUIDITY_DATA = os.path.join(DIST_DIR, 'data/liquidity.csv')

PROVIDERS_DATA = os.path.join(DIST_DIR, 'data/providers/{}.csv')

ROI_DATA = os.path.join(DIST_DIR, 'data/roi/{}.csv')

VOLUME_DATA = os.path.join(DIST_DIR, 'data/volume/{}.csv')

TOTAL_VOLUME_DATA = os.path.join(DIST_DIR, 'data/total_volume.csv')

TOKENS_DATA = os.path.join(DIST_DIR, 'data/tokens.json')

EVENT_TRANSFER = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'

EVENT_TOKEN_PURCHASE = '0xcd60aa75dea3072fbc07ae6d7d856b5dc5f4eee88854f5b4abf7b680ef8bc50f'

EVENT_ETH_PURCHASE = '0x7f4091b46c33e918a0f3aa42307641d17bb67029427a5369e54b353984238705'

EVENT_ADD_LIQUIDITY = '0x06239653922ac7bea6aa2b19dc486b9361821d37712eb796adfd38d81de278ca'

EVENT_REMOVE_LIQUIDITY = '0x0fbf06c058b90cb038a618f8c2acbf6145f8b3570fd1fa56abb8f0f3f05b36e8'

ALL_EVENTS = [EVENT_TRANSFER, EVENT_TOKEN_PURCHASE, EVENT_ETH_PURCHASE, EVENT_ADD_LIQUIDITY, EVENT_REMOVE_LIQUIDITY]

INFOS_DUMP = 'infos.dump'

LAST_BLOCK_DUMP = 'last_block.dump'
