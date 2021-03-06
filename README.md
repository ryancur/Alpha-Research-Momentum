# Momentum Strategy Research

I wanted to work on this project to dive deeper into financial trading using algorithms and code to handle the research, backtesting, and buy/sell signals used in financial firms. This project focuses on the research aspect of the algorithmic trading process. With the rise in using computers, machine learning and big data to find tradable signals to beat the market (alpha signals), I wanted to understand how existing firms perform research in order to build a better intuition for applying machine learning to markets. This project is a personal project to build understanding through research and application of the skills used in practice and is not intended as financial or investment advice.

## Project Background & Observation
To combat the 2020 novel coronavirus (Covid-19) outbreak within the United States, many city and state officials issued 4-6 week stay-at-home orders that started in March and ended between the end of April to the beginning of May. These stay-at-home orders caused an abrupt fall in demand for oil and an increase in the use of online platforms to conduct work remotely. The drop in demand for oil created a rise in the supply, which quickly lead to a shortage of space to store the excess oil causing prices to decrease into negative territory. This was preceded by lock-downs around the world where the virus was present and rapidly spreading. China's lockdown and subsequent drop in demand, in particular, triggered a price war between Saudi Arabia and Russia in early March due to both countries' refusal to cut crude oil production. On the other hand, large tech stocks continued to rise to historic all time highs as people shifted to remote work, utilizing web-conferencing and cloud computing technologies. For example, Apple's valuation was around $1 Trillion before the stay-at-home orders were put in place in March. After a sharp and rapid decline with the rest of the market, Apple's stock price quickly bounced back and increased to a new all-time-high and continued into new price discovery territory for months. Apple eventually stabilized around a $2 Trillion valuation, doubling its value in less than a year. Many of the other large technology companies saw similar patterns during this time period. The novel coronavirus has been a true "black swan" event with its abrupt effects causing major shifts in the demand and supply curves for many industries in 2020.

I was curious about whether these extreme shifts and following trends in demand and supply between the technology stocks and the oil and gas stocks would provide a statistically significant alpha signal. I used this project to test this curiosity using a cross-sectional momentum strategy. For this particular project, I changed a few things from a typical cross-sectional momentum strategy, which I detail below.

### Financial Research Process

The financial research process is a multi-step process that seeks to find a signal that is statistically significant to trade. This project focuses on the first three steps in the image below: observe & research, form hypothesis, validate hypothesis.

![Research Process](images/a_alpha_research_process.png)
Alpha Research Process *(Source: Udacity, AI for Trading)*

Steps:
1. Observe & Research
2. Form Hypothesis
3. Validate Hypothesis

The first step is where an analyst or trader observes some kind of phenomenon or "effect" in the markets that they think might provide a profitable edge in trading. This is typically an observation of some kind of cause-and-effect consequence related to an event in the world. The effect is observed, and then the analyst starts to perform research around the phenomenon to understand what might have caused it or what it is correlated to.

The next step is to form a hypothesis in order to test whether the effect observed and researched is statistically significant or whether it was caused by chance. This is performed using hypothesis testing, which requires an analyst to state an hypothesis and a null hypothesis. The null hypothesis states that the effect was created by random chance and is not a real phenomenon representative of the population, meaning there is no effect. In other words, there is not sufficient evidence present in the sample to reject the null hypothesis. For this project I use a null hypothesis value of 0.0 to represent no effect. The alternative hypothesis states that the effect was not created by chance and there is a higher chance that it is a real effect. In other words, there is sufficient evidence present in the sample to reject the null hypothesis, meaning that the effect is more likely to exist in the population. However, it is important to note that this does not mean that this is a real phenomenon, further validation is needed after getting a statistically significant p-value. With that said, financial trading is a business and businesses need to move quickly, so finding a statistically significant p-value at a pre-determined alpha level is usually good enough to continue with the strategy in practice. Alpha, in respect to statistics, is the significance level of the hypothesis test and is typically set at the .05 threshold.

The final step in this three step process is to validate the hypothesis. This is where we use statistical analysis to determine whether we should reject or not reject the null hypothesis. To do this we start by calculating the mean return of our portfolio with respect to our resampled interval (more about this later). Then to test whether the mean return was caused by random chance or not, we perform a statistical test called a t-test. T-tests are used to compare means. In a hypothesis test, we compare our sample mean (mean return) to our null hypothesis value (0.0). This is done by calculating the t-statistic, which we can get by dividing the sample mean (mean return) by the standard error of the mean. After calculating the t-statistic we can use it to measure the probability of the observed effect in our sample, assuming the null hypothesis is true for the population. This is the p-value. The p-value helps us determine which hypothesis to support. Next, compare the p-value to the pre-determined alpha (significance level). If the p-value is less than or equal to alpha, reject the null hypothesis and continue to develop the strategy. If the p-value is greater than alpha, do not reject the null hypothesis and go back to the drawing board to come up with a new observed effect and strategy.

I hope you can now see why it is called "seeking alpha." Analysts look for statistically significant effects in the market in order to outperform the market, where statistical significance is set by the alpha term in a hypothesis test. This is in opposition to something called the Efficient Market Theory, which is a hypothesis that the current price of a stock includes all relevant information, making alpha signals impossible to find in a perfectly efficient market. However, in practice that is rarely the case as ever changing and evolving events in the real world create effects that can cause changes in the markets that can be taken advantage of, even if the effect is short lived, which is often the case.

Here are some financial terms with definitions and information from different investing related sources to help with the rest of this project.

[**Alpha**](https://www.investopedia.com/terms/a/alpha.asp)

Alpha, as applied to financial research, is a term used to describe a strategy's ability to generate returns greater than the market returns. For example, if the S&P500 increased 5% in a particular year, a successful strategy or fund that achieved "alpha" may have returned 10% in the same year. Alpha strategies are usually mentioned in conjunction with beta strategies, which passively track the market. The use of alpha strategies is generally called *active* trading/management, while the use of beta strategies is generally referred to as *passive* trading/management.

> Alpha (α) is a term used in investing to describe an investment strategy's ability to beat the market, or it's "edge." *(Source: [Investopedia](https://www.investopedia.com/terms/a/alpha.asp))*

> An alpha is a combination of mathematical expressions, computer source code, and
configuration parameters that can be used, in combination with historical data, to make
predictions about future movements of various financial instruments. *(Source: [Finding Alphas by Igor Tulchinsky](https://www.amazon.com/dp/B014SX8LX2/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1))*

[**Systematic Trading**](https://en.wikipedia.org/wiki/Systematic_trading)

Systematic trading refers to the use of strategies, plans, and rules to guide a trader's approach and decisions while trading in the markets. Systematic trading strategies can be manual, hybrid manual/automated, or fully automated through the use of computers, algorithms, and code. This style of trading helps decrease or eliminate the consequences of emotional trading from fear or greed that could influence a trader's decisions.

> Systematic trading (also known as mechanical trading) is a way of defining trade goals, risk controls and rules that can make investment and trading decisions in a methodical way. *(Source: Wikipedia)*

> The use of computers to model, test and implement mathematical rules for how to trade. *(Source: [Clenow, Andreas. Trading Evolved](https://www.amazon.com/Trading-Evolved-Anyone-Killer-Strategies/dp/109198378X))*

[**Momentum**](https://www.investopedia.com/terms/m/momentum.asp)

Momentum is fundamentally based on Newton's first law of physics, the law of inertia. The law of inertia states that an object at rest will remain at rest or an object in motion will remain in motion in a straight line and with a constant speed until acted upon by a force. Momentum as applied to the markets refers to trend following, where a stock that is moving up or down on a trend will continue to move in the same direction until something acts upon that trend to change its direction (possibly reverse it). This trend following phenomenon is the main assumption behind momentum based strategies. It is the idea that outperforming stocks tend to keep moving upward over time while underperforming stocks tend to keep moving downward over time. I have included a couple of passages from Andreas Clenow's book, Trading Evolved, to describe momentum in respect to markets further.

> Momentum is the principle that stocks that moved up strongly in the recent past are a little more likely to do better than other stocks in the near future. *(Source: [Clenow, Andreas. Trading Evolved](https://www.amazon.com/Trading-Evolved-Anyone-Killer-Strategies/dp/109198378X))*

> Momentum is a market phenomenon that has been working well for decades. It has been confirmed by academics and practitioners and is universally known as a valid approach to the financial markets. *(Source: [Clenow, Andreas. Trading Evolved](https://www.amazon.com/Trading-Evolved-Anyone-Killer-Strategies/dp/109198378X))*

[**Stock Universe**](https://www.investopedia.com/terms/u/universeofsecurities.asp#:~:text=A%20universe%20of%20securities%20generally,parameters%20for%20a%20managed%20fund.)

A stock or securities universe refers to a group of stocks that share a common feature or features.

> A universe of securities generally refers to a set of securities that share a common feature. *(Source: Investopedia)*

> Stock universe is a general term in finance that refers to a group of stocks that share certain common features, belong to the same market or a set of stocks that are used in verifying or simulating trading strategies. *(Source: Udacity, AI for Trading)*


### Cross-Sectional Momentum Strategy

For this project I used a cross-sectional momentum strategy. This is a strategy where a trader invests in multiple stocks at the same time by ranking the stocks within a universe using historical data to calculate previous returns, then putting the best performers in a long portfolio and the worst performers in a short portfolio. For this specific project, I decided to try a different approach by putting tech stocks in a long universe and oil & gas stocks in a short universe by looking at the industry trends first, then using each industry as the stock universe to which I could short or long from. I then only added to my short and long portfolios from the short or long universes. In practice, this strategy is typically used with stocks that have some kind of common feature, such as "technology stocks" and there is only one stock universe to draw from. In other words, I would use an entire sector for my universe, such as technology, and then filter through all the stocks to get the *n* best and worst performers to add to my stock portfolios (short and long). This is where my implementation of this strategy deviates from the traditional approach for this particular project. For an example of the traditional cross-sectional momentum strategy check out the cs_momentum_strategy.py script. For this project I will be discussing the results from the cs_momentum_strategy_separated.py script, which ranks the stocks within each industry universe (long and short) and then picks the best performers within the long universe and the worst performers in the short universe and adds them to their respective portfolios. This deviation was because I wanted to use the overall industry trend first, then select the best stocks of the outperforming industry, while selecting the worst performers from the bottom performing industry. I also wanted to keep my data collection and stock universe small so I could spend more time trying to understand the strategy. This may not be a correct implementation of this strategy, however, I was curious to see the result.

Here is a diagram that shows identification of top and bottom performers within a stock universe and then selecting three stocks to put into the long and short portfolios. The stock tickers make up the universe, the plots of the closing prices shows the top and bottom performers. The list to the right shows the rank order of the stock returns. The folders on the far right represent the long and short portfolios and the stocks that were put into each portfolio.

![Cross-sectional Strategy](images/a_cross-sectional_strategy.png)
Cross-sectional Strategy *(Source: Udacity, AI for Trading)*

**Cross-sectional Strategy Steps:**
1. Choose a stock universe and get data (used daily data)
2. Re-sample prices for the desired investment interval, extract interval-end prices, compute log returns (used monthly)
3. Rank by interval-end returns, select top and bottom n stocks - put top performers in long portfolio and worst performers in short portfolio
4. Compute long and short portfolio returns (used arithmetic mean because I assumed each investment would get equal amount)
5. Combine portfolio returns
6. Continue to do this for the selected investment interval (trading)

For a breakdown of the anatomy of a cross-sectional momentum strategy see below. Each number generally corresponds to the numbers in the steps above.

![Cross-sectional Anatomy](images/a_anatomy_cross_sectional_strategy.png)
Cross-sectional Anatomy *(Source: Udacity, AI for Trading)*


## Project Details

The datasets were downloaded from Yahoo Finance in CSV format. Each dataset included daily Open, High, Low, Close, Adjusted Close, and Volume columns. I used the Adjusted Close column for my data. I chose 5 stocks from the technology sector and 5 stocks from the oil & gas sector. For this project, I then created a technology stock universe and an oil & gas stock universe and took only long positions from the tech universe and only short positions from the oil & gas universe. I resampled the data from daily into monthly intervals. This means trading/portfolio rebalancing was done on a monthly basis. Each month I selected the top and bottom two stocks, based on previous returns, and added the top performing stocks to the long portfolio and the bottom performing stocks to the short portfolio. For simplicity, each stock received equal investment. This allowed me to use the arithmetic mean when computing the portfolio returns. My null hypothesis was that the true monthly return mean was zero. I used a one-sample t-test to determine whether to reject the null hypothesis with a significance level (alpha) of .05.

For the purposes of plotting, I used Apple's stock ticker for the long portfolio and Exxon's stock ticker for the short portfolio.

Tech Stack: Python3, Pandas, Numpy, Matplotlib, Scipy

Script: cs_momentum_strategy_separated.py

**Research Information**

| Category | Information |
| --- | --- |
|Dataset Start Date:| 2015-11-06 00:00:00 |
|Dataset End Date:| 2020-11-05 00:00:00 |
|Dataset Base Time Interval:| Daily |
|Resampled Time Interval:| Monthly |
|Stock Universe:| ['FB', 'AMZN', 'AAPL', 'MSFT', 'GOOGL', 'CVX', 'EQT', 'MRO', 'RRC', 'XOM'] |
|Null Hypothesis| 0.0 |
|Alpha| .05 |

### Stock Universe & Data

**Long: AAPL, AMZN, FB, GOOGL, MSFT**

| Date | AAPL | AMZN | FB | GOOGL | MSFT |
| --- | --- | --- | --- | --- | --- |
|2020-10-30| 108.860001 | 3036.149902 | 263.109985 | 1616.109985 | 202.470001|
|2020-11-02| 108.769997 | 3004.479980 | 261.359985 | 1624.319946 | 202.330002|
|2020-11-03| 110.440002 | 3048.409912 | 265.299988 | 1645.660034 | 206.429993|
|2020-11-04| 114.949997 | 3241.159912 | 287.380005 | 1745.849976 | 216.389999|
|2020-11-05| 119.029999 | 3322.000000 | 294.679993 | 1762.500000 | 223.289993|

![APPL Stock Price](images/sep_stock_price_lng_AAPL.png)

**Short: CVX, EQT, MRO, RRC, XOM**

| Date | CVX | EQT | MRO | RRC | XOM |
| --- | --- | --- | --- | --- | --- |
|2020-10-30| 69.500000 | 15.14 | 3.96 | 6.58 | 32.619999|
|2020-11-02| 72.150002 | 15.47 | 4.15 | 6.59 | 33.990002|
|2020-11-03| 71.739998 | 14.86 | 4.14 | 6.36 | 33.410000|
|2020-11-04| 71.769997 | 14.20 | 4.27 | 6.04 | 33.230000|
|2020-11-05| 72.139999 | 14.26 | 4.29 | 6.04 | 33.169998|

![XOM Stock Price](images/sep_stock_price_sh_XOM.png)

### Trading Strategy Steps

**Re-sample Prices**

![APPL Resample](images/sep_resample_lng_AAPL.png)

![XOM Resample](images/sep_resample_sh_XOM.png)

This is where the daily prices were resampled into month-end prices to align with my monthly portfolio rebalance schedule.

**Compute Log Returns**

![APPL Resample](images/sep_log_returns_lng_AAPL.png)

![XOM Resample](images/sep_log_returns_sh_XOM.png)

Then the log returns were computed. There are multiple reasons for using log returns. From a machine learning perspective, stock prices are time series data and typically have something called a unit root, which makes time series data difficult to model. When working with time series we need to remove the trend and make the data stationary. Computing the log returns is one way to do this. Differencing is another. Here is a list of other reasons why analysts use log returns according to Udacity's AI for Trading course:
1. Log returns can be interpreted as continuously compounded returns.
2. Log returns are time-additive. The multi-period log return is simply the sum of single period log returns.
3. The use of log returns prevents security prices from becoming negative in models of security returns.
4. For many purposes, log returns of a security can be reasonably modeled as distributed according to a normal distribution.
5. When returns and log returns are small (their absolute values are much less than 1), their values are approximately equal.
6. Logarithms can help make an algorithm more numerically stable.

The first item in the list is used later when calculating the Annualized Rate of Return, which is used to help with human readability and comparison.

**Sample of Long Portfolio Trades (last 5 months)**

| Date | AAPL | AMZN | FB | GOOGL | MSFT |
| --- | :---: | :---: | :---: | :---: | :---: |
|2020-07-31| 1 | 1 | 0 | 0 | 0 |
|2020-08-31| 1 | 1 | 0 | 0 | 0 |
|2020-09-30| 1 | 0 | 1 | 0 | 0 |
|2020-10-31| 0 | 1 | 0 | 0 | 1 |
|2020-11-30| 0 | 0 | 1 | 1 | 0 |

**Sample of Short Portfolio Trades (last 5 months)**

| Date | CVX | EQT | MRO | RRC | XOM |
| --- | :---: | :---: | :---: | :---: | :---: |
|2020-07-31| 0 | 1 | 0 | 1 | 0 |
|2020-08-31| 1 | 0 | 1 | 0 | 0 |
|2020-09-30| 0 | 0 | 1 | 0 | 1 |
|2020-10-31| 0 | 1 | 1 | 0 | 0 |
|2020-11-30| 1 | 0 | 0 | 0 | 1 |

**Expected Portfolio Returns**

![Portfolio Returns](images/sep_portfolio_returns.png)


### Results

**Expected Portfolio Returns**

| Category | Information |
| --- | --- |
|Mean| 0.021220 |
|Standard Error| 0.008161 |
|Standard Deviation| 0.063739 |
|Annulaized Rate of Return| 29.00% |

**Alpha Analysis**

| Category | Information |
| --- | --- |
|Null Hypothesis| 0.0 |
|Alpha| .05 |
|t-value| 2.6 |
|p-value| 0.005857 |

Reject the Null Hypothesis: **True**

For this specific type of strategy I got a p-value of .005857, which is less than alpha, meaning I would reject the null hypothesis. My interpretation of this hypothesis test would be that if the strategy has no effect in the market, .5857% of research studies will obtain the effect observed in this sample, or larger, because of random sample error. It is worded this way because we assume the null hypothesis is true. However, since I was able to obtain a p-value less than alpha, I reject the null hypothesis, moving forward with this strategy in practice.


## Further Work
- Build more inclusive plots that look at all or more of the sample of stocks chosen.
- Query an API, such as Yahoo Finance, to create a dynamic filter for the long and short portfolios.
- Filter the top 10 best performing and worst performing equities into a long portfolio and a short portfolio dynamically.
- Apply a coefficient to discriminate between stocks that follow a more stable trend to stocks that are more volatile (i.e. apply R2 from a linear regression).
- Refactor code into a class to make it more usable in a repeatable way.


## Sources

Historical Stock Data (OHLC) was collected from [Yahoo Finance](https://finance.yahoo.com/)

Project was from [Udacity](https://www.udacity.com/), AI for Trading

Finding Alphas: A Quantitative Approach to Building Trading Strategies (Igor Tulchinsky)

Trading Evolved: Anyone can Build Killer Trading Strategies in Python (Andreas Clenow)

Systematic Trading: A unique new method for designing trading and investing systems (Robert Carver)

For hypothesis testing and statistics, I used Jim Frost's blog, [Statistics By Jim](https://statisticsbyjim.com/). I like his blog for an intuitive dive into statistics.
