# Portfolio Optimization

This software is just a coding test and should not be used for actual portfolio optimization. Always double check your own results. I take no responsability or make any guarantee for this software.

Download all files to the same folder, put index_portfolio (the HTML file) in its own folder called "template".

The goal of the program is to make an optimized allocation when adding new stocks to an existing portfolio.

Start out by adding your existing stock to the two left columns. Then add the new stocks to the last column. Put the total cash amount your stocks will cost in "Value Adding". The program will download stock data from Yahoo to run the process, in the Date box you can choose how old data you want to use (the end point for the data is always yesterday).

![alt text](https://github.com/CJRockball/Portfolio_optimization/blob/main/images/portfolio_input.png)

If you want a minimum/maximum allocation percentage of the final portfolio you can add that. It might be hard to estimate before any calculations so the possible range is calculate in the table section. So run it one time without any changes and then go back and rerun with a reasonable number.

![alt text](https://github.com/CJRockball/Portfolio_optimization/blob/main/images/table.png)

The program will do maximum sharpe value optimization for the modified portfolio (which isn't a good method to optimize stockportfolios because it uses historic results. But as I said this is a programming excercise). It will also do 3 portfolios (mximum sharpe, global minimum variance and equal risk) optimization without any initial portfolio values i.e. form scratch. 

The table tab will print out portfolio composition and some statistics for each portfolio.

You can change to the dash view at the tabs at the top of the window. The dash will visualize different aspects of the portfolios. In the top left is the effective frontier for the modified portfolio (in green) as well as a prtfolio without inital restarint. The top right graph shows the wealth development for the different portfolios for the time period you have choosen.

The center row panels show risk and weight allocation of each portfolio. The bottom left panel shows the allocation of each stock in the portfolio. The bottom right panel shows the covariance in between the stocks for the time period you have choosen.

![alt text](https://github.com/CJRockball/Portfolio_optimization/blob/main/images/dash.png)

The last tab shows some risk measures.

![alt text](https://github.com/CJRockball/Portfolio_optimization/blob/main/images/risk.png)

The top graph shows drawdown for the different portfolios. 

On the bottom you can see different VaR measures for Month, Week, Day for the modified portfolio.


















