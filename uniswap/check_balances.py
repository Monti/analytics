import logging
import traceback

from analyse import load_raw_data, get_chart_range, load_ts_data
from config import web3, ERC_20_ABI, VTHO_ADDRESS
from exchange_info import ExchangeInfo
from utils import timeit

logging.basicConfig(level=logging.INFO, format='%(message)s')


def calc_divs(timestamp_diff, eth_balance):
    return timestamp_diff * eth_balance * 5 // 10 ** 9


def compare_with_divs(stored_tokens, actual_tokens, divs):
    return actual_tokens - stored_tokens - divs


def find_last_log_blocknumber(logs, block_number):
    last_log = block_number
    for log in logs:
        if log['blockNumber'] <= block_number:
            last_log = log['blockNumber']
        else:
            break
    return last_log


@timeit
def check_balances(info):
    try:
        token_contract = web3.eth.contract(abi=ERC_20_ABI, address=info.token_address)
        passed = True

        for i, b in enumerate(get_chart_range()):
            stored_eth = info.roi[i].eth_balance
            if stored_eth == 0:
                continue
            actual_eth = web3.eth.getBalance(info.exchange_address, block_identifier=b)
            stored_tokens = info.roi[i].token_balance
            actual_tokens = token_contract.functions.balanceOf(info.exchange_address).call(block_identifier=b)
            if stored_eth != actual_eth:
                logging.warning('ETH balance differs at {} {}. Actual {}, but stored {}. Diff {}'.format(
                    i, b, actual_eth, stored_eth, actual_eth - stored_eth))
                passed = False

            last_block_number = find_last_log_blocknumber(info.logs, b)
            last_ts = ts_dict.get(last_block_number) or web3.eth.getBlock(last_block_number)['timestamp']
            cur_ts = ts_dict.get(b) or web3.eth.getBlock(b)['timestamp']
            divs = calc_divs(cur_ts - last_ts, stored_eth) if info.token_address == VTHO_ADDRESS else 0
            if abs(actual_tokens - stored_tokens - divs) > 10:
                logging.warning(
                    'Token balance differs at {} {}. Actual {}, but stored {}. Diff {}. Diff with divs {}'.format(
                        i, b, actual_tokens, stored_tokens, actual_tokens - stored_tokens,
                                                            actual_tokens - stored_tokens - divs))
                passed = False

        if passed:
            logging.info('{} balances are good'.format(info.token_symbol))
        else:
            logging.warning('{} balances differ'.format(info.token_symbol))

        return passed
    except:
        traceback.print_exc()
        return False


def get_price(info: ExchangeInfo, block_number: int) -> (int, int):
    token_contract = web3.eth.contract(abi=ERC_20_ABI, address=info.token_address)
    actual_eth = web3.eth.getBalance(info.exchange_address, block_identifier=block_number)
    actual_tokens = token_contract.functions.balanceOf(info.exchange_address).call(block_identifier=block_number)
    return actual_tokens, actual_eth


infos = load_raw_data()
ts_dict = load_ts_data()
results = [check_balances(i) for i in infos]

if all(results):
    logging.info('All balances are good')
else:
    logging.warning('Some balances are different')
