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
                    rsi_ = rr1
                    rsi30 = self.rsi_to(rsi_, rsi_s, s_1, s_2, close_1)
                    df.loc[index, 'RSI_' + str(rr1)] = rsi30

            close_s = close_
            rsi_s = rsi
        return df

    ##业务综合
    def get_k_line(self, coin='link', bar='5m'):
        # client = MarketData.MarketAPI(debug=True)
        if coin in ['eth1', 'btc1']:
            instId = (coin + '-USD-SWAP').upper()
        else:
            instId = (coin + '-USDT-SWAP').upper()
        # re = client.get_candlesticks(instId=instId, bar=bar, limit=300)
        limit_num = 80
        url = f'{self.okex_path}/api/v5/market/candles?instId={instId}&bar=5m&limit={limit_num}'
        # re1 = requests.get(url)
        # re = re1.json()

        s = requests.session()
        s.keep_alive = False
        re1 = s.get(url)
        re = re1.json()

        time.sleep(1)

        data1 = re['data']
        data1 = self.k_line_date(data1)

        len1 = len(data1)
        if len1 < limit_num - 1:
            logger.info(f" {coin} {bar}  {len1}  {instId} 不够数量  {re}")
            return [], []

        df = DataFrame(data=data1, columns=['time', 'open', 'high', 'low', 'close', 'a', 'b', 'c', 'confirm', 'd_t'])
        df = df[['time', 'open', 'high', 'low', 'close', 'confirm', 'd_t']]
        df = df.sort_values('time', ascending=True)

        df = df.set_index('d_t')
        # logger.info(df)

        df = self.RSI2(df)
        # logger.info(df)
        # logger.info(df.iloc[298])
        return df.iloc[limit_num -1 ], df.iloc[limit_num-2]

    def okex_can(self):
        api_key = self.api_key
        secret_key = self.secret_key
        passphrase = self.passphrase

        flag = "0"  # live trading: 0, demo trading: 1
        tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag, debug=True, domain=self.okex_path)
        ret = tradeAPI.get_order_list()
        logger.info(ret)
        dic = []
        for ord in ret['data']:
            cid = ord['clOrdId']
            if 'rrr1aa' in cid:
                dic.append(
                    {
                        "instId": ord['instId'],
                        "ordId": ord['ordId']
                    },
                )
                if len(dic) >= 20:
                    re1 = tradeAPI.cancel_multiple_orders(dic)
                    logger.info(f" 撤单：  {re1}")
                    time.sleep(0.3)
                    dic = []

            # re1 = tradeAPI.cancel_order(ord['instId'], ordId=ord['ordId'])
            # logger.info(f" 撤单：  {re1}")
        if len(dic) >= 1:
            re1 = tradeAPI.cancel_multiple_orders(dic)
            logger.info(f" 撤单：  {re1}")
            time.sleep(0.3)

    def okex_trade_par(self, dic=[], coin='bnb', side='buy', price='-0.001', num=1, rsi_=10, ):
        api_key = self.api_key
        secret_key = self.secret_key
        passphrase = self.passphrase
        flag = "0"  # live trading: 0, demo trading: 1
        tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag, debug=False, domain=self.okex_path)

        ### 分段 集中下单
        if len(dic) >= 20 or (coin == '' and len(dic) >= 1) or (coin == 'btc' and rsi_ == 97):
            logger.info(f'下单参数 {coin}   {dic}')
            logger.info(f'     下单     ')
            re = tradeAPI.place_multiple_orders(dic)
            # logger.info(f"参数 {dic}")
            logger.info(f'下单返回 {coin}   {re}')
            time.sleep(0.3)
            del dic[:]

        if coin == '':
            return

        if coin in ['eth1', 'btc1']:
            instId = (coin + '-USD-SWAP').upper()
        else:
            instId = (coin + '-USDT-SWAP').upper()
        if coin in ['btc1', 'doge1']:
            num = 1

        posSide = 'long'
        tdMode = 'isolated'
        cid1 = ''.join(random.choices(string.ascii_lowercase, k=10))
        clOrdId = 'rrr1aa' + (coin.upper()) + cid1 + str(rsi_).upper()

        par = {
            'clOrdId': clOrdId,
            'instId': instId,
            'tdMode': tdMode,
            'side': side,
            'ordType': 'post_only',
            'px': price,
            'sz': num,
            'posSide': posSide,
        }
        dic.append(par)

    def gg1(self, coin):
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

    okex_path = 'https://www.okx.com'
    api_key = None
    secret_key = None
    passphrase = None
    coin_list = None
    rsi_list = None

    def __init__(self):
        config_fil = './config.json'
        with open(config_fil, 'r') as f:
            config = json.load(f)
            self.api_key = config['api_key']
            self.secret_key = config['secret_key']
            self.passphrase = config['passphrase']

            self.coin_list = config['coin_list']
            self.rsi_list = config['rsi_list']

    ff = True
    def loop(self):
        logger.info(" ")
        logger.info(" ")
        logger.info(" ########  ########  ########  ########  ########  ######## ")
        logger.info(" ########  ########  ########  ########  ########  ######## ")

        coin_list = self.coin_list
        rsi_list = self.rsi_list

        self.okex_can()

        dic = []
        nn = 1
        for coin, init_num in coin_list.items():
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

            for rsi_, num1 in rsi_list.items():
                a = Decimal(str(num1 - nn ))
                b = Decimal(str(init_num))
                num = str(a * b)

                if float(num) <= 0:
                    continue


                rsi_price = re1['RSI_' + str(rsi_)]

                ## 当前的rsi对应的价格 和开盘价格小于千n  就跳过这个价格
                diff_p = abs(float(rsi_price) - float(open_price)) / float(open_price)
                if diff_p < 0.0015:
                    logger.info(f"当前价格和开盘价格相差太小-跳过{coin} {rsi_} diff_p{diff_p}  open_price {open_price}  rsi_price {rsi_price}")
                    continue

                ## 买单
                if rsi_ <= 30:
                    ## 如果上次的rsi 小于 当前的rsi就跳过
                    if rsi_ > last_RSI:
                        logger.info(f" 当次的rsi 要比上次低 跳过 {coin}  {rsi_}  {last_RSI} ")
                    ## 买单
                    self.okex_trade_par(dic, coin, 'buy', rsi_price, num, rsi_)
                    logger.error(len(dic))
                if rsi_ >= 70:
                    ##卖单
                    self.okex_trade_par(dic, coin, 'sell', rsi_price, num, rsi_)
                    logger.error(len(dic))

        self.okex_trade_par(dic, '')
        logger.error(len(dic))

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
                    time.sleep(5)
                    continue
                except Exception as e:
                    pass

            # 18分钟
            if time1 - time2 > 18 * 60:
                time2 = time1
                try:
                    self.zhisun()
                except Exception as e:
                    pass


len_ = len(sys.argv)
if len_ >= 2:
    func = sys.argv[1]

    class_ = okex_rsi()
    ff = hasattr(class_, func)
    if ff:
        getattr(class_, func)()
    else:
        logger.info("no fun")

