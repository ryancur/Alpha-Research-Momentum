import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas._testing import assert_frame_equal

def resample_prices(prices, freq='M'):
    """
    Resample close prices for each ticker at a specified frequency.

    Parameters
    ----------
    close_prices : DataFrame
        Close prices for each ticker and date
    freq : str
        Frequency to sample at
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

    Returns
    -------
    prices_resampled : DataFrame
        Resampled prices for each ticker and date
    """
    return prices.resample(freq).last()

def compute_log_returns(prices):
    """
    Compute log returns for each ticker.

    Parameters
    ----------
    prices : DataFrame
        Prices for each ticker and date

    Returns
    -------
    log_returns : DataFrame
        Log returns for each ticker and date
    """
    return np.log(prices) - np.log(prices.shift(1))

def shift_returns(returns, shift_n):
    """
    Generate shifted returns

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    shift_n : int
        Number of periods to move, can be positive or negative

    Returns
    -------
    shifted_returns : DataFrame
        Shifted returns for each ticker and date
    """
    return returns.shift(shift_n)

def get_top_n(prev_returns, top_n):
    """
    Select the top performing stocks

    Parameters
    ----------
    prev_returns : DataFrame
        Previous shifted returns for each ticker and date
    top_n : int
        The number of top performing stocks to get

    Returns
    -------
    top_stocks : DataFrame
        Top stocks for each ticker and date marked with a 1
    """
    new_df = pd.DataFrame(0, index=prev_returns.index.values,
                            columns=prev_returns.columns)
    for idx, row in prev_returns.iterrows():
        top = row.nlargest(top_n).index.values.tolist()
        for ticker in new_df.columns:
            if ticker in top:
                new_df.loc[idx, ticker] = 1
    return new_df

def portfolio_returns(df_long, df_short, lookahead_returns, n_stocks):
    """
    Compute expected returns for the portfolio, assuming equal investment in
    each long/short stock.

    Parameters
    ----------
    df_long : DataFrame
        Top stocks for each ticker and date marked with a 1
    df_short : DataFrame
        Bottom stocks for each ticker and date marked with a 1
    lookahead_returns : DataFrame
        Lookahead returns for each ticker and date
    n_stocks: int
        The number of stocks chosen for each month

    Returns
    -------
    portfolio_returns : DataFrame
        Expected portfolio returns for each ticker and date
    """
    pos_long = lookahead_returns * df_long
    pos_short = lookahead_returns * df_short
    total_returns = (pos_long - pos_short)/n_stocks
    return total_returns

def analyze_alpha(expected_portfolio_returns_by_date, null_hypothesis=0.0):
    """
    Perform a t-test with the null hypothesis being that the expected mean
    return is zero.

    Parameters
    ----------
    expected_portfolio_returns_by_date : Pandas Series
        Expected portfolio returns for each date

    Returns
    -------
    t_value
        T-statistic from t-test
    p_value
        Corresponding p-value
    """
    t, p = stats.ttest_1samp(expected_portfolio_returns_by_date,
                                null_hypothesis)
    p = p/2
    return t, p

def preprocess_data(ticker_list, values='Adj Close', relative_path=''):
    """
    Aggregate multiple csv files of OHLC data into a table where each ticker is
    the column header and the values are the sort_by input.
    Assumes:
        Historical data has the columns: Date, Open, High, Low, Close,
            Adj Close, Volume.
        Data is stored as ticker.csv --> AAPL.csv

    Parameters
    ----------
    ticker_list : List
        List of ticker symbols. ['AAPL', 'FB', 'GOOG']
    values : String
        Column name of the values in the returned dataframe. Default is
        'Adj Close'.
    relative_path : String
        Relative path to where the data is stored. Default is local folder.
        If the data is in another folder, format example: '../data/'

    Returns
    -------
    prices: DataFrame
        dataframe with ticker symbols as column names, date as index and
        values as row values.

    """
    column_names = {'Date': 'date', 'Open': 'open', 'High': 'high',
                    'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close',
                    'Volume': 'volume'}
    df = pd.DataFrame([])
    values_column = '_'.join(values.lower().split())
    for ticker in ticker_list:
        try:
            df_ticker = pd.read_csv(f"{relative_path}{ticker}.csv",
                                    parse_dates=['Date'], index_col=False)
            df_ticker.rename(columns=column_names, inplace=True)
            df_ticker['ticker'] = ticker
            df = df.append(df_ticker, ignore_index=True)
        except:
            print(f"{ticker} not in files.")
            continue
    try:
        prices = df.reset_index().pivot(index='date', columns='ticker',
                                    values=values_column)
    except:
        return None
    return prices

def test_preprocess_data():
    """
    Test function for preprocess_data function
    """
    test_status_list = []
    test_counter = 0
    total_tests = 3
    df_1 = preprocess_data(['AAPL'], values='Adj Close',
                            relative_path='../data/')
    df_2 = preprocess_data(['AAPL'], values='Adj Close',
                            relative_path='../data/')
    df_3 = preprocess_data(['GOOGL'], values='Adj Close',
                            relative_path='../data/')

    # test 1 - dataframes equal
    try:
        assert_frame_equal(df_1, df_2)
        test_status_list.append('Test 1 Passed.')
    except AssertionError:
        test_status_list.append('Test 1 Failed.')
    test_counter += 1

    # test 2 - dataframes not equal
    try:
        assert_frame_equal(df_1, df_3)
        test_status_list.append('Test 2 Failed.')
    except AssertionError:
        test_status_list.append('Test 2 Passed.')
    test_counter += 1

    # test 3 - ticker not in files
    try:
        assert (preprocess_data(['ABC'], values='Adj Close',
                                relative_path='../data/') == None)
        test_status_list.append('Test 3 Passed.')
    except AssertionError:
        test_status_list.append('Test 3 Failed.')
    test_counter += 1

    print(f"{test_counter}/{total_tests} Tests Passed.")
    if test_counter != total_tests:
        for test in test_status_list:
            print(test)


if __name__ == '__main__':
    pd.set_option('max_columns', 100)

    ## Before Research - Test preprocess
    test_preprocess_data()

    ###########################################################################
    ### START HERE ###
    ###########################################################################

    ## Research Setup - TODO: Change per hypothesis and datasets used
    freq_dict = {'D': ['Daily', 252], 'W': ['Weekly', 52], 'M': ['Monthly', 12],
                'Q': ['Quarterly', 4], 'A': ['Annually', 1]}

    dataset_base_freq = 'D'
    frequency = 'M'
    plot_ticker_long = 'AAPL'
    plot_ticker_short = 'XOM'
    risk_free_rate = 0.001 # As of 11/6/20

    top_bottom_n = 2 # Used in STEP 5 - getting the top and bottom performing stock(s)

    ticker_symbols_long = ['FB', 'AMZN', 'AAPL', 'MSFT', 'GOOGL']
    ticker_symbols_short = ['CVX', 'EQT', 'MRO', 'RRC', 'XOM']

    null_hyp = 0.0 #Default = 0.0
    alpha_lvl = 0.05 #Default = 0.05

    base_freq = freq_dict[dataset_base_freq][0]
    plt_time_interval = freq_dict[frequency][0]
    ar_multiple = freq_dict[frequency][1]

    ## STEP 0: Import datasets from CSV files and aggregate into a dataframe
    prices_long = preprocess_data(ticker_symbols_long, values='Adj Close',
                            relative_path='../data/')

    prices_short = preprocess_data(ticker_symbols_short, values='Adj Close',
                            relative_path='../data/')

    start_date = prices_long.index.min()
    end_date = prices_long.index.max()

    ## STEP 1: visualize stock data
    plt.figure()
    plt.plot(prices_long[plot_ticker_long])
    plt.title(f"{plot_ticker_long} Stock Price - Long Group")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig(f"../images/sep_stock_price_lng_{plot_ticker_long}.png")
    plt.show()

    plt.figure()
    plt.plot(prices_short[plot_ticker_short])
    plt.title(f"{plot_ticker_short} Stock Price - Short Group")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig(f"../images/sep_stock_price_sh_{plot_ticker_short}.png")
    plt.show()

    ## STEP 2: resample data
    if frequency == dataset_base_freq:
        resamp_prices_long = prices_long
    else:
        resamp_prices_long = resample_prices(prices_long, frequency)

    if frequency == dataset_base_freq:
        resamp_prices_short = prices_short
    else:
        resamp_prices_short = resample_prices(prices_short, frequency)

    plt.figure()
    plt.plot(prices_long[plot_ticker_long], color='blue', alpha=0.5, label='Close')
    plt.plot(resamp_prices_long[plot_ticker_long], color='navy', alpha=0.6,
                label=f'{plt_time_interval} Close')
    plt.title(f"{plot_ticker_long} Stock - Close Vs {plt_time_interval} Close")
    plt.xlabel("Date")
    plt.ylabel(f"Price ($)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"../images/sep_resample_lng_{plot_ticker_long}.png")
    plt.show()

    plt.figure()
    plt.plot(prices_short[plot_ticker_short], color='blue', alpha=0.5, label='Close')
    plt.plot(resamp_prices_short[plot_ticker_short], color='navy', alpha=0.6,
                label=f'{plt_time_interval} Close')
    plt.title(f"{plot_ticker_short} Stock - Close Vs {plt_time_interval} Close")
    plt.xlabel("Date")
    plt.ylabel(f"Price ($)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"../images/sep_resample_sh_{plot_ticker_short}.png")
    plt.show()

    ## STEP 3: generate log returns
    returns_long = compute_log_returns(resamp_prices_long)

    plt.figure()
    plt.plot(returns_long[plot_ticker_long])
    plt.hlines(y=0, xmin=returns_long.index.min(), xmax=returns_long.index.max(),
                color='black', linestyles='--', lw=1)
    plt.title(f"{plot_ticker_long} {plt_time_interval} Log Returns")
    plt.xlabel("Date")
    plt.ylabel(f"{plt_time_interval} Returns")
    plt.tight_layout()
    plt.savefig(f"../images/sep_log_returns_lng_{plot_ticker_long}.png")
    plt.show()

    returns_short = compute_log_returns(resamp_prices_short)

    plt.figure()
    plt.plot(returns_short[plot_ticker_short])
    plt.hlines(y=0, xmin=returns_short.index.min(), xmax=returns_short.index.max(),
                color='black', linestyles='--', lw=1)
    plt.title(f"{plot_ticker_short} {plt_time_interval} Log Returns")
    plt.xlabel("Date")
    plt.ylabel(f"{plt_time_interval} Returns")
    plt.tight_layout()
    plt.savefig(f"../images/sep_log_returns_sh_{plot_ticker_short}.png")
    plt.show()

    ## STEP 4: view previous timestep and next timestep returns
    # long stock universe
    prev_returns_long = shift_returns(returns_long, 1)
    lookahead_returns_long = shift_returns(returns_long, -1)

    plt.figure()
    plt.plot(prev_returns_long.loc[:, plot_ticker_long], color='blue', alpha=0.5,
                label='Shifted Returns')
    plt.plot(returns_long.loc[:, plot_ticker_long], color='gray', alpha=0.5,
                label='Returns')
    plt.hlines(y=0, xmin=returns_long.index.min(), xmax=returns_long.index.max(),
                color='black', linestyles='--', lw=1)
    plt.title(f"Previous Returns of {plot_ticker_long} Stock")
    plt.xlabel("Date")
    plt.ylabel(f"{plt_time_interval} Returns")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"../images/sep_prev_returns_lng_{plot_ticker_long}.png")
    plt.show()

    plt.figure()
    plt.plot(lookahead_returns_long.loc[:, plot_ticker_long], color='blue', alpha=0.5,
                label='Shifted Returns')
    plt.plot(returns_long.loc[:, plot_ticker_long], color='gray', alpha=0.5,
                label='Returns')
    plt.hlines(y=0, xmin=returns_long.index.min(), xmax=returns_long.index.max(),
                color='black', linestyles='--', lw=1)
    plt.title(f"Lookahead Returns of {plot_ticker_long} Stock")
    plt.xlabel("Date")
    plt.ylabel(f"{plt_time_interval} Returns")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"../images/sep_lookahead_returns_lng_{plot_ticker_long}.png")
    plt.show()

    #short stock universe
    prev_returns_short = shift_returns(returns_short, 1)
    lookahead_returns_short = shift_returns(returns_short, -1)

    plt.figure()
    plt.plot(prev_returns_short.loc[:, plot_ticker_short], color='blue', alpha=0.5,
                label='Shifted Returns')
    plt.plot(returns_short.loc[:, plot_ticker_short], color='gray', alpha=0.5,
                label='Returns')
    plt.hlines(y=0, xmin=returns_short.index.min(), xmax=returns_short.index.max(),
                color='black', linestyles='--', lw=1)
    plt.title(f"Previous Returns of {plot_ticker_short} Stock")
    plt.xlabel("Date")
    plt.ylabel(f"{plt_time_interval} Returns")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"../images/sep_prev_returns_sh_{plot_ticker_short}.png")
    plt.show()

    plt.figure()
    plt.plot(lookahead_returns_short.loc[:, plot_ticker_short], color='blue', alpha=0.5,
                label='Shifted Returns')
    plt.plot(returns_short.loc[:, plot_ticker_short], color='gray', alpha=0.5,
                label='Returns')
    plt.hlines(y=0, xmin=returns_short.index.min(), xmax=returns_short.index.max(),
                color='black', linestyles='--', lw=1)
    plt.title(f"Lookahead Returns of {plot_ticker_short} Stock")
    plt.xlabel("Date")
    plt.ylabel(f"{plt_time_interval} Returns")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"../images/sep_lookahead_returns_sh_{plot_ticker_short}.png")
    plt.show()

    ## STEP 5: get the top n stocks and visualize
    df_long = get_top_n(prev_returns_long, top_bottom_n)
    df_short = get_top_n(-1*prev_returns_short, top_bottom_n)
    print('Longed Stocks\n', df_long)
    print('Shorted Stocks\n', df_short)

    ## STEP 6: get portfolio returns and visualize
    lookahead_returns = lookahead_returns_long.join(lookahead_returns_short)
    df_long = df_long.join(prev_returns_short)
    df_short = df_short.join(prev_returns_long)

    for col in df_long.columns:
        df_long[col] = np.where(df_long[col] == 1, 1, 0)

    for col in df_short.columns:
        df_short[col] = np.where(df_short[col] == 1, 1, 0)

    expected_portfolio_returns = portfolio_returns(df_long, df_short,
                                lookahead_returns, 2*top_bottom_n)

    plt.figure()
    plt.plot(expected_portfolio_returns.T.sum())
    plt.hlines(y=0, xmin=returns_long.index.min(), xmax=returns_long.index.max(),
                color='black', linestyles='--', lw=1)
    plt.title(f"Expected Portfolio Returns")
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.tight_layout()
    plt.savefig("../images/sep_portfolio_returns.png")
    plt.show()

    ## STEP 7: annualized rate of return
    expected_portfolio_returns_by_date = expected_portfolio_returns.T.sum().dropna()
    portfolio_ret_mean = expected_portfolio_returns_by_date.mean()
    portfolio_ret_ste = expected_portfolio_returns_by_date.sem()
    portfolio_ret_std = expected_portfolio_returns_by_date.std()
    portfolio_ret_annual_rate = (np.exp(portfolio_ret_mean * ar_multiple) - 1) * 100


    print(f"""
    Research Information:
        Dataset Start Date:             {start_date}
        Dataset End Date:               {end_date}
        Dataset Base Time Interval:     {base_freq}
        Resampled Time Interval:        {plt_time_interval}
        Asset Sample:
            {ticker_symbols_long + ticker_symbols_short}
    """)

    print(f"""
    Expected Portfolio Returns by Date:
        Mean:                           {portfolio_ret_mean:.6f}
        Standard Error:                 {portfolio_ret_ste:.6f}
        Standard Deviation:             {portfolio_ret_std:.6f}
        Annualized Rate of Return:      {portfolio_ret_annual_rate:.2f}%
    """)

    # t-value and p-value for alpha
    t_value, p_value = analyze_alpha(expected_portfolio_returns_by_date, null_hyp)
    print(f"""
    Alpha analysis:
        null hypothesis:                {null_hyp}
        alpha:                          {alpha_lvl}
        t-value:                        {t_value:.3f}
        p-value:                        {p_value:.6f}
    """)

    if p_value <= alpha_lvl:
        reject_null = True
    else:
        reject_null = False

    print("Reject the Null Hypothesis: ", reject_null)
