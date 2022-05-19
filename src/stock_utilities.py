from datetime import datetime as dt, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, pacf, acf
from statsmodels.tsa.arima.model import ARIMA


class StockForecastModel:
    def __init__(self, ticker_name, start_dt, end_dt=None):
        if end_dt is None:
            end_dt = dt.today()
        self.df_daily, self.df_weekly = StockForecastModel.load_data(
            ticker_name, start_dt, end_dt
            )

    @staticmethod
    def load_data(ticker_name, start_dt, end_dt):
        if isinstance(end_dt, str):
            end_dt = dt.strptime(end_dt, "%Y-%m-%d").date() + timedelta(days=1)
        else:
            end_dt = end_dt + timedelta(days=1)
            
        daily_data = yf.download(
            tickers=ticker_name, start=start_dt, end=end_dt, interval='1d'
            ).reset_index().rename(columns={'Date': 'date'})
        
        daily_data['year'] = daily_data.date.apply(lambda x: x.year)
        daily_data['week'] = daily_data.date.apply(lambda x: x.week)

        weekly_data = daily_data.groupby(['year', 'week']).agg(
            {'Open': 'mean', 'High': 'mean', 'Low': 'mean', 'Close': 'mean'}
            ).reset_index()
        return daily_data, weekly_data

    @staticmethod
    def _get_d(x):
        pval = 1
        adf_stat = 1
        d = 0
        while pval >= 0.05 or adf_stat >= 0:
            res = adfuller(x)
            adf_stat, pval = res[0], res[1]
            x = x.diff().dropna()
            d += 1
        return d - 1

    @staticmethod
    def _check_conf(x, confint):
        return (np.abs(x) > np.abs(confint[1] - confint[0])/2) and (np.abs(x) != 1)

    @staticmethod
    def _get_first_q_cross(crosses):
        m = min(crosses)
        while m in crosses:
            m += 1
        return m - 1

    @staticmethod
    def _get_pq(x, d, p_cap=None, q_cap=None):
        for _ in range(d):
            x = x.diff().dropna()

        pacf_v, confint = pacf(x, method='ywm', alpha=0.05)
        p_lst = np.argwhere(
            np.array(list(map(StockForecastModel._check_conf, pacf_v, confint)))==True
            ).reshape(-1)
        if len(p_lst) > 0:
            p = min(p_lst)
            if p_cap is not None and p_cap<p:
                p = min(p, p_cap)
                print("p capped!")
        else:
            print("None of the pacf values is crossing the confidence interval")
            print("Setting p to 1")
            p = 1
            
        acf_v, confint = acf(x, alpha=0.05)
        q_lst = np.argwhere(
            np.array(list(map(StockForecastModel._check_conf, acf_v, confint)))==True
            ).reshape(-1)
        if len(q_lst) > 0:
            q = StockForecastModel._get_first_q_cross(q_lst)
            if q_cap is not None and q_cap<q:
                print("q capped!")
                q = min(q, q_cap)
        else:
            print("None of the acf values is crossing the confidence interval")
            print("Setting q to 0")
            q = 0
        return p, q

    @staticmethod
    def get_pdq(x, p_cap=None, q_cap=None):
        d = StockForecastModel._get_d(x)
        p, q = StockForecastModel._get_pq(x, d, p_cap=p_cap, q_cap=q_cap)
        return p, d, q
    
    def generate_graphs(self, data_type='daily', col='Close', diff_level=0):
        if data_type=='daily':
            data = self.df_daily[col]
        elif data_type=='weekly':
            data = self.df_weekly[col]
        else:
            raise(ValueError)

        s = data.copy()
        for _ in range(diff_level):
            s = s.diff().dropna()

        _, axes = plt.subplots(1, 2, sharex=True, figsize=(15,5))
        axes[0].plot(data)
        axes[1].plot(s)
        
        result = adfuller(s)
        print("Fuller stat: ", result[0])
        print("P-value", result[1])

        if result[1] < 0.05:
            _, axes = plt.subplots(1, 2, sharex=True, figsize=(15,5))
            _ = plot_acf(s)
            _ = plot_pacf(s, method='ywm', ax=axes[1])


    def fit_arima_model(
        self, data_type='daily', col='Close', 
        p_cap=None, q_cap=None, verbose=False
        ):
        if data_type=='daily':
            data = self.df_daily
            interval_str = str(self.df_daily['date'].iloc[-1] + timedelta(days=1))
        elif data_type=='weekly':
            data = self.df_weekly
            nxt_week = self.df_weekly['week'].iloc[-1] + 1
            year = self.df_weekly['year'].iloc[-1]
            if nxt_week > 52:
                nxt_week = 1
                year = year + 1
            interval_str = f"week {nxt_week} of year {year}"
        else:
            raise(ValueError)

        arima_para = StockForecastModel.get_pdq(data[col], p_cap=p_cap, q_cap=q_cap)

        model = ARIMA(endog=data[col], order=arima_para)
        model = model.fit()

        data[f'pred_{col}'] = model.predict().to_list()
        trn_r2 = metrics.r2_score(data[col], data[f'pred_{col}'])
        print("Train R-squared: ", trn_r2)
        print(
            f"Estimated {col} price for {interval_str}: {model.forecast(1).iloc[0]}"
        ) 
        if verbose:
            residuals = pd.DataFrame(model.resid)
            print(residuals.describe())
            residuals.plot(kind='kde')
            print(model.summary())
            plt.figure(figsize=(12,5), dpi=100)
            plt.plot(data[col][1:], color='c', marker='o')
            plt.plot(data[f'pred_{col}'][1:], color='y', marker='o')
            plt.legend(['true_price', 'predictions'])
        return model.forecast(1).iloc[0], arima_para

    def generate_predictions(self, p_cap=None, q_cap=2):
        pred_dict = {}
        para_dict = {}
        col_lst = ['Open', 'High', 'Low', 'Close']
        for c in col_lst:
            print(f"********************** daily {c} **********************")
            pred, para = self.fit_arima_model(
                data_type='daily', col=c, p_cap=p_cap, q_cap=q_cap
                )
            pred_dict[f"pred_{c}"] = [pred]
            para_dict[f"daily_{c}"] = para

            print(f"********************** weekly {c} **********************")
            pred, para = self.fit_arima_model(
                data_type='weekly', col=c, p_cap=p_cap, q_cap=q_cap
                )
            pred_dict[f"weekly_pred_{c}"] = [pred]
            para_dict[f"weekly_{c}"] = para
            
        pred_df = pd.DataFrame(pred_dict)
        pred_df['min_pred'] = pred_df.min(axis=1)
        pred_df['max_pred'] = pred_df.max(axis=1)
        return pred_df, para_dict

    def generate_recommendation(self, pred_df, buy_cutoff, sell_cutoff, buying_price):
        closing_price = self.df_daily['Close'].iloc[-1]
        pred_closing_price = self.df_daily['pred_Close'].iloc[-1]

        print("\n************************")
        today = self.df_daily['date'].iloc[-1]
        tomorrow = str(today.date() + timedelta(days=1))
        print(
            "Trade signal is generated using price data "
            f"from {self.df_daily['date'].iloc[-1]}"
            )

        cond1 = closing_price - pred_closing_price >  sell_cutoff * pred_closing_price
        cond2 = pred_df['max_pred'].iloc[0] > buying_price

        if cond1 and cond2:
            msg = (
                "Sell as soon as price crosses"
                f" {pred_df['max_pred'].iloc[0]} on {tomorrow}"
                )
            print(msg)
        elif pred_closing_price - closing_price > buy_cutoff * pred_closing_price:
            msg = (
                "Buy as soon as price falls below"
                f" {pred_df['min_pred'].iloc[0]} on {tomorrow}"
                )
            print(msg)
        else:
            msg = f"No action required for {tomorrow}"
            print(msg)
        print("************************")
        return msg


class TradeStrategy:
    def __init__(self, forecast_object):
        df_daily = forecast_object.df_daily.copy()
        req_cols = [
            'date', 'year', 'week', 'Open', 'High', 'Low', 'Close',
            'pred_Open', 'pred_High', 'pred_Low', 'pred_Close'
            ]
        self._validate_data(df_daily, req_cols, 'df_daily')
        df_daily = df_daily[req_cols]

        df_weekly = forecast_object.df_weekly.copy()
        req_cols = ['year', 'week', 'pred_Open', 'pred_High', 'pred_Low', 'pred_Close']
        self._validate_data(df_weekly, req_cols, 'df_weekly')
        df_weekly = df_weekly[req_cols]

        self.df_estimates = self._generate_estimates_df(df_daily, df_weekly)

    def _validate_data(self, df, req_cols, name):
        missing_cols = set(req_cols) - set(df.columns)
        assert missing_cols == set(), f"{missing_cols} cols are missing in {name}"
    
    def _generate_estimates_df(self, df_daily, df_weekly):
        df_weekly = df_weekly.rename(columns={
            'pred_Open': 'weekly_pred_Open',
            'pred_High': 'weekly_pred_High',
            'pred_Low': 'weekly_pred_Low',
            'pred_Close': 'weekly_pred_Close'
        })
        return pd.merge(df_daily, df_weekly, on=['year', 'week'], how='left')

    def _generate_signals(self, df_estimates, buy_cutoff, sell_cutoff):
        df = df_estimates.copy()
        df['sell_next_flag'] = (
            (df['Close'] - df['pred_Close']) 
            >= (sell_cutoff * df['pred_Close'])
        ).astype(int)

        df['buy_next_flag'] = (
            (df['pred_Close'] - df['Close']) 
            >= (buy_cutoff * df['pred_Close'])
        ).astype(int)

        df['sell_flag'] = df[
            'sell_next_flag'
            ].shift(1).fillna(0).astype(int)
        df['buy_flag'] = df[
            'buy_next_flag'
            ].shift(1).fillna(1).astype(int)
        df = df.drop(
            columns=['sell_next_flag', 'buy_next_flag']
            )
        df.loc[1, 'sell_flag'] = 0

        df['max_pred'] = df.filter(like='pred').max(axis=1)
        df['min_pred'] = df.filter(like='pred').min(axis=1)
        return df

    def _sell_units(
        self, initial_investment, units, open,
        high, max_pred, last_buy_price
        ):
        profit = 0
        sell = True
        buy = False
        if open >= max_pred and open > last_buy_price:
            amount = units * open
            sell = False
            buy = True
            profit = amount - initial_investment
            units = 0
        elif high >= max_pred and max_pred > last_buy_price:
            amount = units * max_pred
            sell = False
            buy = True 
            profit = amount - initial_investment
            units = 0
        return buy, sell, profit, units

    def _buy_units(self, amount, open, low, min_pred, bypass_conditions=False):
        if not bypass_conditions:
            units = 0
            buy_price = None
            buy = True
            sell = False
            if open <= min_pred:
                units = amount/open
                buy_price = open
                buy = False
                sell = True
            elif low <= min_pred:
                units = amount/min_pred
                buy_price = min_pred
                buy = False
                sell = True
        else:
            units = amount/open
            buy = False
            sell = True
            buy_price = open
        return buy, sell, units, buy_price

    def run_simulation(
        self, buy_cutoff=0.03, sell_cutoff=0.03, initial_amount=10000, verbose=True
        ):
        df = self._generate_signals(self.df_estimates, buy_cutoff, sell_cutoff)
        
        buy = True
        sell = False
        force_initial_buy = True
        amount = initial_amount
        total_profit = 0
        col_idx = dict(zip(df.columns, range(df.shape[1])))

        for r in df.values:
            if buy and not sell and r[col_idx['buy_flag']]:
                buy, sell, units, last_buy_price = self._buy_units(
                    amount, r[col_idx['Open']], r[col_idx['Low']],
                    r[col_idx['min_pred']], bypass_conditions=force_initial_buy
                    )
                force_initial_buy = False
                if units != 0:
                    last_amount = amount
                    amount = 0
                    if verbose:
                        print(f"{units} units bought on {r[col_idx['date']]}")
                    
            elif sell and not buy and r[col_idx['sell_flag']]:
                buy, sell, profit, units = self._sell_units(
                    last_amount, units, r[col_idx['Open']], r[col_idx['High']],
                    r[col_idx['max_pred']], last_buy_price
                    )
                if profit != 0:
                    total_profit += profit
                    amount = last_amount + profit
                    if verbose:
                        print(f"All units sold on {r[col_idx['date']]}")
                        print(
                            f"Profit earned: {profit}; Updated total amount: {amount}\n"
                            )
        if verbose:
            print("**************************")
            print(f"Percentage returns: {(total_profit/initial_amount)*100}")
            print("**************************")
        return df, units, amount, total_profit

    def tune_buy_sell_cutoffs(self, min_cutoff, max_cutoff, step_size):
        grid = np.arange(min_cutoff, max_cutoff + step_size, step_size)
        results = {}
        b_lst = []
        s_lst = []
        profit_lst = []
        unit_lst = []
        for buy in grid:
            for sell in grid:
                b_lst.append(buy)
                s_lst.append(sell)
                _, stock_units, _, profit = self.run_simulation(
                    buy_cutoff=buy, sell_cutoff=sell,
                    initial_amount=10000, verbose=False
                    )
                profit_lst.append(profit)
                unit_lst.append(stock_units)
        results['buy_cutoff'] = b_lst
        results['sell_cutoff'] = s_lst
        results['profit'] = profit_lst
        results['final_units'] = unit_lst
        results = pd.DataFrame(results).sort_values('profit', ascending=False)
        tuned_cutoffs = {
            'max_profit': results['profit'].iloc[0]/100,
            'buy_cutoff': results['buy_cutoff'].iloc[0],
            'sell_cutoff': results['sell_cutoff'].iloc[0]
        }
        return tuned_cutoffs


class Pipeline:
    def __init__(self, ticker_name, start_date, end_date=None):
        self.tick = ticker_name
        self.start_dt = start_date
        self.end_dt = end_date
    
    def _tune_cutoffs(self, forecast_obj):
        strategy = TradeStrategy(forecast_obj)
        return strategy.tune_buy_sell_cutoffs(0.01, 0.1, 0.01)

    def refresh_recommendation(self, buying_price=None, pcap=None, qcap=2):
        forecast_obj = StockForecastModel(self.tick, self.start_dt, self.end_dt)
        pred_df, params = forecast_obj.generate_predictions(p_cap=pcap, q_cap=qcap)
        tuned_cutoffs = self._tune_cutoffs(forecast_obj)
        if buying_price is not None:
            msg1 = forecast_obj.generate_recommendation(
                pred_df, tuned_cutoffs['buy_cutoff'],
                tuned_cutoffs['sell_cutoff'], buying_price
                )
        else:
            msg1 = 'Please enter buying price to get recommendation'
            print(msg1)
        msg2 = (
            f"Current strategy would have earned {tuned_cutoffs['max_profit']}"
            f" % profit, if deployed since {self.start_dt}"
            )
        print(msg2)
        info_dict = {'recom': msg1, 'additional_info': msg2}
        return pred_df, params, tuned_cutoffs, info_dict