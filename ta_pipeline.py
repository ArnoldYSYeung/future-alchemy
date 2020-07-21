"""
Technical Analysis Pipeline functions

Author:     Arnold YS Yeung
Date:       2020-07-18

"""

import ta                     # primary library
import technical_feats as tf  # secondary library 

def get_pct_change(df, n):
    """
    Calculate the percentage change for series in a period of n.
    """
    assert 'Close' in df
    df_series = df['Close']         #   price of previous n days
    df_series = df_series.shift(n)
    
    df['pct_change'] = df['Close'] / df_series
    #df = df.drop(columns=['temp'])
    
    print(df.columns)
    
    return df
    

def get_bollinger_bands(df, n, ndev):
    """
    Calculate Bollinger Band values.
    Arguments:
        - df (pd.DataFrame) :                   input DataFrame containing 'Close' prices
        - n (int) :                             size of window (days)
        - ndev (int) :                          number of standard deviations
    Returns:
        - Updated df (pd.DataFrame)
    """
    indicator_bb = ta.volatility.BollingerBands(close=df['Close'], 
                                                     n=n, ndev=ndev)
    df['boll_mavg'] = indicator_bb.bollinger_mavg()
    df['boll_hband'] = indicator_bb.bollinger_hband()
    df['boll_lband'] = indicator_bb.bollinger_lband()
    df['boll_hband_ind'] = indicator_bb.bollinger_hband_indicator()
    df['boll_lband_ind'] = indicator_bb.bollinger_lband_indicator()
    df['boll_bandwidth'] = indicator_bb.bollinger_wband()
    
    return df

def get_RSI(df, n):
    """
    Calculate the RSI over a window period.
    Arguments:
        - df (pd.DataFrame) :                   input DataFrame containing 'Close' prices
        - n (int) :                             size of window (days)
    """
    
    indicator_rsi = ta.momentum.RSIIndicator(close=df['Close'], n=n)
    df['RSI'] = indicator_rsi.rsi()
    
    return df

def get_daily_return(df):
    """
    Calculate the daily return.
    Arguments:
        - df (pd.DataFrame) :                   input DataFrame containing 'Close' prices
    """
    indicator_dr = ta.others.DailyReturnIndicator(close=df['Close'])
    df['daily_return'] = indicator_dr.daily_return()
    
    return df
    
def get_sharpe_ratio(df, n, sf=252, risk_free_rate=0.01):
    """
    Calculate the rolling Sharpe Ratio within a window of size n.
    Arguments:
        - df (pd.DataFrame) :                   input DataFrame containing 'Close' prices
        - n (int) :                             period of window
        - sf (int) :                            sampling frequency (# of days per year)
        - risk_free_rate (float) :              risk-free rate of return per annum
    """
    if 'daily_return' not in df:
        df = get_daily_return(df)
    
    df['sharpe_ratio'] = tf.get_rolling_sharpe_ratio(df[['daily_return']], rolling_window=n)
    
    return df

def get_ROC(df, n):
    """
    Calculates the Rate of Change (ROC) (i.e., % change) within a period
    Arguments:
        - df (pd.DataFrame) :                   input DataFrame containing 'Close' prices
        - n (int) :                             period of window 
    """
    indicator_roc = ta.momentum.ROCIndicator(close=df['Close'])
    df['ROC'] = indicator_roc.roc()
    
    return df

def get_ADI(df):
    
    indicator_adi = ta.volume.AccDistIndexIndicator(high=df['High'], low=['Low'], 
                                                    close=df['Close'], volume=df['Volume'])
    df['ADI'] = indicator_adi.acc_dist_index()
    return df

def get_MACD(df, n_short_term, n_long_term, n_signal):
    
    indicator_macd = ta.trend.MACD(close=df['Close'], n_fast=n_short_term, n_slow=n_long_term,
                                   n_sign=n_signal)
    df['MACD'] = indicator_macd.macd()
    return df

def get_on_balance_volume(df):
    
    indicator_on_balance_volume = ta.volume.OnBalanceVolumeIndicator(close=df['Close'],
                                                                     volume=df['Volume'])
    df['on_balance_volume'] = indicator_on_balance_volume.on_balance_volume()
    return df
    

def calculate_technical_indicators(df, include_indicators, params, verbose=False):
    """
    Calculates and include the technical indicators into inputted dataframe
    Arguments:
        - df (pd.DataFrame) :                       DataFrame containing 'Close' prices per day
        - include_indicators (list[str,]):          List of indicators to include
        - params (dict{str: ,}):                    Parameters for indicators
        - verbose (bool):                           Print progress  
    """

    if include_indicators == []:
        print("No indicators included.")
        return df
    
    if 'bollinger_bands' in include_indicators:
        if verbose:
            print("Bollinger bands...")
        boll_n_days = params['boll_n_days']
        boll_n_std = params['boll_n_std']
        df = get_bollinger_bands(df, n=boll_n_days, ndev=boll_n_std)
    
    if 'RSI' in include_indicators:
        if verbose:
            print("RSI...")
        rsi_n_days = params['rsi_n_days']
        df = get_RSI(df, rsi_n_days)
    
    if 'daily_return' in include_indicators:
        if verbose:
            print("Daily return...")
        df = get_daily_return(df)
    
    if 'sharpe_ratio' in include_indicators:
        if verbose:
            print("Sharpe ratio...")
        sharpe_n_days = params['sharpe_n_days']
        sharpe_sf = params['annual_sf']
        rfr = params['annual_rfr']
        df = get_sharpe_ratio(df, n=sharpe_n_days, sf=sharpe_sf, risk_free_rate=rfr)
    
    if 'ROC' in include_indicators:
        if verbose:
            print("ROC...")
        roc_n_days = params['roc_n_days']
        df = get_ROC(df, roc_n_days)
        
    if 'ADI' in include_indicators:
        if verbose:
            print("ADI...")
        df = get_ADI(df)
    
    if 'MACD' in include_indicators:
        if verbose:
            print("Moving Average Convergence Divergence...")
        n_short_term = params['macd_n_short']
        n_long_term = params['macd_n_long']
        n_signal = params['macd_n_signal']
        df = get_MACD(df, n_short_term=n_short_term, n_long_term=n_long_term,
                      n_signal=n_signal)
    
    if 'on_balance_volume' in include_indicators:
        if verbose:
            print("On-Balance Volume...")
        df = get_on_balance_volume(df)
    
    return df

def calculate_technical_features(df, include_features, window_size, verbose=False):
    
    if 'avg_price' in include_features:
        if verbose:
            print("Average Price...")
        assert 'Close' in df
        df['avg_close'] = df['Close'].rolling(window=window_size).mean()
    
    if 'max_price' in include_features:
        if verbose:
            print("Max Price...")
        assert 'Close' in df
        df['max_close'] = df['Close'].rolling(window=window_size).max()
    
    if 'min_price' in include_features:
        if verbose:
            print("Min Price...")
        assert 'Close' in df
        df['min_close'] = df['Close'].rolling(window=window_size).min()
    
    if 'max_price_pct' in include_features:
        if verbose:
            print("Max Price Pct...")
        assert 'Close' in df
        df['max_close_pct'] = df['Close'].rolling(window=window_size).max() / df['Close'].rolling(window=window_size).mean()

    if 'min_price_pct' in include_features:
        if verbose:
            print("Min Price Pct...")
        assert 'Close' in df
        df['min_close_pct'] = df['Close'].rolling(window=window_size).min() / df['Close'].rolling(window=window_size).mean()
    
    if 'bollinger_num_hband' in include_features:
        if verbose:
            print("Bollinger Num HBands...")
        assert 'boll_hband_ind' in df
        df['boll_hband_num'] = df['boll_hband_ind'].rolling(window=window_size).sum()

    if 'bollinger_num_lband' in include_features:
        if verbose:
            print("Bollinger Num LBands...")
        assert 'boll_lband_ind' in df
        df['boll_lband_num'] = df['boll_lband_ind'].rolling(window=window_size).sum()

    return df
    
    


        