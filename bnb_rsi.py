import traceback

import okx.Trade as Trade
import okx.MarketData as MarketData
import okx.Account as Account
from loguru import logger
import pandas as pd
from pandas import DataFrame
import time
import datetime
import numpy as np
import string
import random
import sys
import requests
from decimal import Decimal, getcontext
import json
from binance.um_futures import UMFutures


class okex_rsi:
    ## K线时间
    def k_line_date(self, data1=[]):
        del_list = []
        for it in data1:
            time1 = int(it[0])
            timeArray = time.localtime(time1 / 1000)
            d_t = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)

            it[4] = float(it[4])
            it.append(d_t)

            ##要删掉最后一个没有结束的
            confirm = int(it[8])
            if confirm == 0:
                del_list.append(it)

        for it1 in del_list:
            # data1.remove(it1)
            pass

        return data1

    ## 辨认 rsi
    def rsi_to(self, rsi_, rsi_s, s_1, s_2, close_1):
        rs_ = (-100 / (rsi_ - 100)) - 1
        # 假设上次的 rsi大于 30 则本次价格要低
        rsi30 = 0
        if rsi_s > rsi_:
            up_ = 0
            down_1 = ((s_1 + up_) / rs_) - s_2
            rsi30 = close_1 - down_1
        else:
            down = 0
            ## ( s_1 + up) /  (s_2 + down)  = rs_
            #  ( s_1 + up) = rs_ * (s_2 + down)
            # up = close_ - close_1
            up = rs_ * (s_2 + down) - s_1
            rsi30 = close_1 + up

        return rsi30

    ## 计算rsi
    def RSI2(self, df, periods=4):
        ii = 0
        close_s = 0
        up_avg = 0
        down_avg = 0
        rsi = 0
        rsi_s = 0

        rsi_list = self.rsi_list

        for index, row in df.iterrows():
            ii += 1
            ## 当前的价格
            close_ = df.loc[index, 'close']
            close_ = float(close_)
            # 上一次的价格
            close_1 = close_s

            if ii <= periods + 1:
                df.loc[index, 'RSI'] = np.nan
                # logger.info(f" {ii}  {down_avg}")
                if ii >= 2:
                    if close_ >= close_1:
                        up_avg += close_ - close_1
                    else:
                        down_avg += close_1 - close_
                    up_avg_ = up_avg / periods
                    down_avg_ = down_avg / periods

                    if down_avg_ == 0:
                        rsi = 0
                    else:
                        rs = up_avg_ / down_avg_
                        rsi = 100 - 100 / (1 + rs)
                if ii == periods + 1:
                    df.loc[index, 'RSI'] = rsi
            else:
                if close_ >= close_1:
                    up = close_ - close_1
                    down = 0
                else:
                    up = 0
                    down = close_1 - close_

                # 类似移动平均的计算公式;
                s_1 = up_avg * (periods - 1)
                s_2 = down_avg * (periods - 1)
                up_avg = (s_1 + up) / periods
                down_avg = (s_2 + down) / periods
                if down_avg == 0:
                    rsi = 0
                else:
                    rs = up_avg / down_avg
                    rsi = 100 - 100 / (1 + rs)

                df.loc[index, 'RSI'] = rsi

                for rr1 in rsi_list:
                    rsi_ = float(rr1)
                    rsi30 = self.rsi_to(rsi_, rsi_s, s_1, s_2, close_1)
                    df.loc[index, 'RSI_' + str(rr1)] = rsi30

            close_s = close_
            rsi_s = rsi
        return df

    ##业务综合
    def get_k_line(self, coin='link', bar='5m'):
        instId = (coin + 'USDT').upper()
        limit1 = 30
        url = f'{self.okex_path}/fapi/v1/klines?symbol={instId}&interval=5m&limit={limit1}'

        s = requests.session()
        s.keep_alive = False
        re1 = s.get(url)
        re = re1.json()

        time.sleep(0.5)

        data1 = re
        data1 = self.k_line_date(data1)

        len1 = len(data1)
        if len1 < limit1 - 1:
            logger.info(f" {coin} {bar}  {len1}  {instId} 不够数量  {re}")
            return [], []

        df = DataFrame(data=data1,
                       columns=['time', 'open', 'high', 'low', 'close', 'a', 'b', 'c', 'confirm', 'd_t2', 'dd', 'tt',
                                'd_t'])
        df = df[['time', 'open', 'high', 'low', 'close', 'd_t']]
        df = df.sort_values('time', ascending=True)

        df = df.set_index('d_t')
        df = self.RSI2(df)

        return df.iloc[limit1 - 1], df.iloc[limit1 - 2]

    pos_info = {}

    def okex_can(self):
        api_key = self.api_key
        secret_key = self.secret_key

        tradeAPI = UMFutures(api_key, secret_key)
        ret = tradeAPI.get_orders()

        logger.info(f"  未完成订单 ： {ret} {len(ret)} ")

        re3 = tradeAPI.get_position_risk()
        for pos1 in re3:
            if pos1['positionAmt'] == '':
                continue
            symbol = pos1['symbol']
            positionAmt = float(pos1['positionAmt'])
            if positionAmt == 0:
                continue
            sy = symbol.replace('USDT', '').lower()
            sy = sy.replace('USDC', '').lower()

            self.pos_info[sy] = abs(positionAmt)

        logger.debug(f"  仓位信息 ： {self.pos_info} ")
        # logger.info(ret)
        dic = []
        for ord in ret:
            cid = ord['orderId']
            symbol = ord['symbol']
            timeInForce = ord['timeInForce']
            clientOrderId = ord['clientOrderId']
            ## 挂单
            if timeInForce != 'GTX':
                continue
            ## rrr1aa
            if 'rrr1aa' not in clientOrderId:
                continue

            re55 = tradeAPI.cancel_order(symbol, cid)
            logger.info(f"  取消订单 {cid}： {re55}")

    def bnb_trade_par(self, dic=[], coin='bnb', side='buy', price='-0.001', num=1, rsi_=10, c_price=0):
        api_key = self.api_key
        secret_key = self.secret_key
        tradeAPI = UMFutures(api_key, secret_key)

        logger.error(len(dic))

        ### 分段 集中下单
        if len(dic) >= 5 or (coin == '' and len(dic) >= 1) :
            try:
                logger.info(f'批量 下单参数 {coin}   {dic}')
                logger.info(f'   批量  下单   {len(dic)}  ')
                re = tradeAPI.new_batch_order(batchOrders = dic)
                logger.info(f'批量批量 下单返回 {coin}   {re}')
                time.sleep(0.3)
                del dic[:]
            except Exception as e:
                traceback.print_exc()
                logger.error(f"批量下单异常 {coin}   {e}")
                time.sleep(0.2)
                del dic[:]

        if coin == '':
            return

        if coin.upper() == "DEFI":
            instId = (coin + 'USDT').upper()
        else:
            instId = (coin + 'USDC').upper()
        # num = 0.01

        posSide = 'long'
        tdMode = 'isolated'
        cid1 = ''.join(random.choices(string.ascii_lowercase, k=10))
        clOrdId = 'rrr1aa' + (coin.upper()) + cid1 + str(rsi_).upper()
        clOrdId = clOrdId.replace('.', '')

        price = round(float(price), c_price)
        price = str(price)
        num = str(num)

        par = {
            'newClientOrderId': clOrdId,
            'symbol': instId,
            # 'positionSide': tdMode,
            'side': side.upper(),
            'type': 'LIMIT',
            'price': price,
            'quantity': num,
            'positionSide': posSide,
            'timeInForce': 'GTX',
        }
        dic.append(json.dumps(par))
        # dic.append(par)

        try:
            logger.info(f'下单参数 {coin}   {par}')
            re4 = tradeAPI.new_order(**par)
            logger.info(f'下单返回 {coin}   {re4}')
            del dic[:]
        except Exception as e:
            logger.error(f"下单异常 {coin}   {e}")
            time.sleep(0.2)

    def gg1(self, coin):
        return
        api_key = self.api_key
        secret_key = self.secret_key
        passphrase = self.passphrase

        flag = "0"  # live trading: 0, demo trading: 1
        tradeAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag, debug=False, domain=self.okex_path)
        if coin in ['eth', 'btc']:
            lever = 7
            instId = (coin + '-USD-SWAP').upper()
        else:
            lever = 5
            instId = (coin + '-USDT-SWAP').upper()

        re = tradeAPI.set_leverage(instId=instId, lever=lever, mgnMode='isolated', posSide='long')
        logger.info(f"  杠杆设置 ： {re}")
        time.sleep(0.2)

    okex_path = ' https://fapi.binance.com'

    api_key = None
    secret_key = None
    passphrase = None
    coin_list = None
    rsi_list = None
    nn = None

    def __init__(self):
        ## defi
        config_fil = './config.json'
        with open(config_fil, 'r') as f:
            data = json.load(f)
            config = data['binance']

            self.api_key = config['api_key']
            self.secret_key = config['secret_key']
            self.passphrase = config['passphrase']

            self.coin_list = config['coin_list']

            self.rsi_list = data['rsi_list']
            self.nn = data['nn']
            self.max_pos = data['max_pos']

    ff = True

    def loop(self):
        logger.info(" ")
        logger.info(" ")
        logger.info(" ########  ########  ########  ########  ########  ######## ")
        logger.info(" ########  ########  ########  ########  ########  ######## ")

        rsi_list = self.rsi_list

        self.okex_can()

        dic = []
        for coin, init_num in self.coin_list.items():
            ##每次启动的时候  设置仓位杠杆
            if self.ff:
                self.gg1(coin)

            ##返回最后两个单元  re1是最后一个单元  re2是倒数第二个单元
            ## re1是没有完成的单元  结束时间是大于当前时间
            re1, re2 = self.get_k_line(coin, '5m')

            if type(re1) == type([]):
                continue
            last_RSI = re2['RSI']
            open_price = re1['open']

            c_price = init_num['price']
            c_num = init_num['num']
            c_value = init_num['value'] if 'value' in init_num else 0
            amt_p = init_num['amt_p'] if 'amt_p' in init_num else 0

            if c_value > 0:
                c_num = round(c_value / float(open_price), amt_p)

            nn = self.nn
            max_pos = self.max_pos
            pos_num = self.pos_info[coin] if coin in self.pos_info else 0
            if pos_num < c_num * max_pos:
                nn = 0
                logger.info(f" {coin} pos_num{pos_num}  max_pos{max_pos} init_num{init_num}仓位数量太少-尽量买入")
            logger.info(f" {coin} nn:{nn} 单次下单：{c_num} 持仓：{pos_num} dict: {self.pos_info}")


            for rsi_, num1 in rsi_list.items():
                rsi_1 = rsi_
                rsi_ = float(rsi_)
                if rsi_ <= 30:
                    a = Decimal(str(num1 - nn))
                else:
                    a = Decimal(str(num1))
                b = Decimal(str(c_num))
                num = str(a * b)
                if float(num) <= 0:
                    logger.info(f" {coin} {rsi_} {num1} {a} {b} {num}数量为0 跳过{open_price}")
                    continue

                # rsi_price = re1['RSI_' + str(rsi_)]
                try:
                    rsi_price = re1['RSI_' + str(rsi_1)]
                except Exception as e:
                    logger.error(e)
                    continue

                ## 当前的rsi对应的价格 和开盘价格小于千n  就跳过这个价格
                diff_p = abs(float(rsi_price) - float(open_price)) / float(open_price)
                if diff_p < 0.0015 or diff_p > 0.2:
                    logger.info(
                        f"当前价格和开盘价格相差太小或者太小-跳过{coin} {rsi_} diff_p{diff_p}  open_price {open_price}  rsi_price {rsi_price}")
                    continue

                ## 买单
                if rsi_ <= 30:
                    ## 如果上次的rsi 小于 当前的rsi就跳过
                    if rsi_ > last_RSI:
                        logger.info(f" 当次的rsi 要比上次低 跳过 {coin}  {rsi_}  {last_RSI} ")
                    ## 买单
                    side = 'buy'
                if rsi_ >= 70:
                    ##卖单
                    side = 'sell'

                self.bnb_trade_par(dic, coin, side, rsi_price, num, rsi_, c_price)

        self.bnb_trade_par(dic, '')

        self.ff = False

    def zhisun(self):
        api_key = self.api_key
        secret_key = self.secret_key
        passphrase = self.passphrase

        flag = "0"  # live trading: 0, demo trading: 1
        Account11 = Account.AccountAPI(api_key, secret_key, passphrase, False, flag, debug=False, domain=self.okex_path)

        ##持仓
        pos_list = Account11.get_positions()
        tradeAPI1 = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag, debug=False, domain=self.okex_path)

        ##获取止损订单
        algos = tradeAPI1.order_algos_list(instType='SWAP', ordType='conditional')
        # logger.info(algos)

        for alg in algos['data']:
            instId = alg['instId']
            algoId = alg['algoId']
            tag = alg['tag']

            if 'rrr1aa' not in tag:
                # break
                continue

            par1 = {
                'instId': instId,
                'algoId': algoId,
            }
            params = [
                par1
            ]
            # re3 = tradeAPI1.cancel_advance_algos(params=params)
            re3 = tradeAPI1.cancel_algo_order(params=params)
            logger.info(f"止损 撤单订单 {params} {re3}")
            time.sleep(0.2)

        logger.info(" ")
        ##添加止损
        for pos in pos_list['data']:
            instId = pos['instId']
            liqPx = pos['liqPx']
            mgnMode = pos['mgnMode']

            if mgnMode == 'cross' or liqPx == '':
                logger.info(f" {instId}  {mgnMode} {liqPx} 跳过 不添加止损")
                continue

            cid1 = ''.join(random.choices(string.ascii_lowercase, k=9))
            tag = 'rrr1aa' + cid1
            price = float(liqPx) * (1 + 0.01)

            re1 = tradeAPI1.place_algo_order(instId=instId, tdMode='isolated', side='sell', posSide='long',
                                             ordType='conditional', tag=tag,
                                             closeFraction='1', slTriggerPxType='mark', slOrdPx='-1', slTriggerPx=price
                                             )
            logger.info(f"止损 下单 {instId}  {liqPx}  {re1}")
            time.sleep(0.2)

    def whil(self):
        time2 = time.time()

        while True:
            time1 = int(time.time())
            yu_time = time1 % 300
            ##logger.info(yu_time)

            time.sleep(1)
            if yu_time in [10, 3, 4, 5, 6, 7]:
                try:
                    self.loop()
                except Exception as e:
                    logger.error(f"loop 异常 {e}")

                time.sleep(5)
                continue

            # 18分钟
            if time1 - time2 > 18 * 60:
                time2 = time1

                try:
                    # self.zhisun()
                    pass
                except Exception as e:
                    logger.error(f"loop 异常 {e}")


## nohup  python3 bnb_rsi_test.py  whil  >> bnb_rsi_test.log   2>&1 &

len_ = len(sys.argv)
if len_ >= 2:
    func = sys.argv[1]

    class_ = okex_rsi()
    ff = hasattr(class_, func)
    if ff:
        getattr(class_, func)()
    else:
        logger.info("no fun")

