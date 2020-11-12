# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:01:25 2020

@author: Snorlax
"""


from flask import Flask, request, redirect,render_template
import webbrowser
import numpy as np
import pandas as pd
from bokeh.plotting import show
from bokeh.layouts import column, row
from bokeh.models import Panel, Tabs
import stock_plot as sp
import ec_func as ef



###----------------------------------------------------------------------------
#Set up flask app and render data page
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index_portfolio6.html')

#------------------------------------------------------------------------------
#Set up graph pages and get input data to variables
@app.route('/', methods=["GET", "POST"])
def my_form_post():
    port_stock1 = request.form['port_stock1']
    port_stock2 = request.form['port_stock2']
    port_stock3 = request.form['port_stock3']
    port_stock4 = request.form['port_stock4']
    port_stock5 = request.form['port_stock5']
    port_stock6 = request.form['port_stock6']
    port_stock7 = request.form['port_stock7']
    port_stock8 = request.form['port_stock8']
    port_stock9 = request.form['port_stock9']
    port_stock10 = request.form['port_stock10']
    port_stock11 = request.form['port_stock11']
    port_stock12 = request.form['port_stock12']
    port_stock13 = request.form['port_stock13']
    port_stock14 = request.form['port_stock14']
    port_stock15 = request.form['port_stock15']
    port_stock16 = request.form['port_stock16']
    port_stock17 = request.form['port_stock17']
    port_stock18 = request.form['port_stock18']
    port_stock19 = request.form['port_stock19']
    
    value1 = request.form['value1']
    value2 = request.form['value2']
    value3 = request.form['value3']
    value4 = request.form['value4']
    value5 = request.form['value5']
    value6 = request.form['value6']
    value7 = request.form['value7']
    value8 = request.form['value8']
    value9 = request.form['value9']
    value10 = request.form['value10']
    value11 = request.form['value11']
    value12 = request.form['value12']
    value13 = request.form['value13']
    value14 = request.form['value14']
    value15 = request.form['value15']
    value16 = request.form['value16']
    value17 = request.form['value17']
    value18 = request.form['value18']
    value19 = request.form['value19']

    new_stock1 = request.form['new_stock1']
    new_stock2 = request.form['new_stock2']
    new_stock3 = request.form['new_stock3']
    new_stock4 = request.form['new_stock4']
    new_stock5 = request.form['new_stock5']
    new_stock6 = request.form['new_stock6']
    new_stock7 = request.form['new_stock7']
    new_stock8 = request.form['new_stock8']
    new_stock9 = request.form['new_stock9']
    new_stock10 = request.form['new_stock10']


    add_value = request.form['add_value']
    min_hold = request.form['min_hold']
    max_hold = request.form['max_hold']
    start_time = request.form['start_time']   
    
#put all input fields in lists   
    port_list = [port_stock1, port_stock2, port_stock3, port_stock4, port_stock5, port_stock6,
                  port_stock7, port_stock8, port_stock9, port_stock10, port_stock11, 
                  port_stock12, port_stock13, port_stock14, port_stock15, port_stock16, 
                  port_stock17, port_stock18, port_stock19]
    
    comp_list = [value1, value2, value3, value4, value5, value6, value7, value8, 
                 value9, value10, value11, value12, value13, value14, value15,
                 value16, value17, value18, value19]
    
    new_list = [new_stock1, new_stock2, new_stock3, new_stock4, new_stock5, new_stock6, 
                 new_stock7, new_stock8, new_stock9, new_stock10]

#set up lists for existing portfolio, new stocks
#get value of existing portfolio and new stocks   
    portfolio_stocks = []  
    portfolio_value = []
    for i,name in enumerate(port_list):
        if name != "Empty":
            portfolio_stocks.append(str(name))
            portfolio_value.append(float(comp_list[i]))
            
    num_org_stocks = len(portfolio_stocks)
    total_stocks_list = [i for i in portfolio_stocks]
    new_stock_list = []
    for name in new_list:
        if name != "Empty":
            total_stocks_list.append(str(name))
            new_stock_list.append(str(name))
            
    total_stocks = len(total_stocks_list) 

    curr_value = 0
    num_curr_stock = len(portfolio_stocks)
    for name in comp_list:
        if name != "Empty":
            curr_value += int(name)

#Download stock data and send to pandas dataframes       
    df_rets,df_rets_d, df_rets_w = ef.stock_data(total_stocks_list,start_time)
    df2 = pd.DataFrame(index=df_rets.index)
    df2['Date'] = pd.to_datetime(df2.index.astype(str))
#Calculate number of days, years of data    
    date_diff = (df2.iloc[-1]['Date']-df2.iloc[0]['Date'])
    day_diff = float(date_diff.days/365)
    
#If there IS money added to the portfolio calculate new total value of the portfolio
#based existing portfolio and money added.
#For the optimizer send weight of the existing stocks and calculate possible range
#for new stocks. set initial value for new stocks to average of added value.       
    if add_value != 'Empty':
        tot_new_value = curr_value + int(add_value)
        max_add = int(add_value)/(tot_new_value)    
        num_new_stocks = len(new_stock_list)
        eq_add = max_add/num_new_stocks
        init_guess = np.repeat(eq_add, total_stocks)

        #Set init guess for existing portfolio to new proportion
        for i in range(num_curr_stock):
            init_guess[i] = int(portfolio_value[i])/tot_new_value
    
        bounds = ()
        max_add_w_low_limits = max_add - float(min_hold)*(num_new_stocks - 1)
        # possible max add is eq_add to max_add
        for i in range(total_stocks):
            if  i < num_curr_stock:
                a = int(portfolio_value[i])/tot_new_value
                bounds += ((a,a),)
            else:
                if min_hold != "0":
                    a = float(min_hold)
                else:
                    a = 0
                if max_hold != "1":
                    if float(max_hold) < eq_add:
                        b = eq_add
                    elif float(max_hold) > max_add_w_low_limits:
                        b = max_add_w_low_limits
                    else:
                        b = float(max_hold)
                else:
                    b = max_add
                
                bounds += ((a,b),)
#if there is no money added to the portfolio calculate GMV portfolio
    else:
        bounds = ((0.0, 1.0),) * total_stocks # an N-tuple of 2-tuples!
        init_guess = np.repeat(1/total_stocks, total_stocks)


###--------------Make Portfolio -----------------------
    er = ef.annualize_rets(df_rets, 12)  #Calculate expected returns
    cov = df_rets.cov() #Calculate covariance matrix
    corr = df_rets.corr() #Calculate correlation matrix

#Set up efficient frontier data for modefied portfolio-------------------------
#Calc mod port max/min
#New weights for fixed portfolio
    new_weight = [x/tot_new_value for x in portfolio_value] #Calc weight of existing portfolio with new total value
    new_stock_er = er[-num_new_stocks:] #Get expected returns for portfolio stocks
    max_er = num_org_stocks + [i for i,j in enumerate(new_stock_er) if j==max(new_stock_er)][0] #Get position in list of new stock that has higherst er
    min_er = num_org_stocks + [i for i,j in enumerate(new_stock_er) if j==min(new_stock_er)][0] #Get position in list of new stock that has lowest return
 
#Extend list of weights with 0 for the new stocks
    for i in range(num_new_stocks):
        new_weight.append(0)
        
    new_weight_max = [i for i in new_weight]#Copy new_weight
    new_weight_min = [i for i in new_weight]#Copy new_weight
    new_weight_max[max_er] = max_add #Add max_weight as weight for the new stock with max er
    max_weight_np = np.array( new_weight_max )#Make list array
    new_weight_min[min_er] = max_add#Add min_weight as weight for the new stock with min er
    min_weight_np = np.array( new_weight_min )#make array
    
    max_ret = ef.portfolio_return(max_weight_np, er)#Calculate max return for portfolio with new stocks
    min_ret = ef.portfolio_return(min_weight_np, er)#Calculate min return for portfolio with new stocks

#Calculate bounds for efficient fronteir for modified portfolio    
    bounds2 = ()       
# possible max add is eq_add to max_add
    for i in range(total_stocks):
        if  i < num_curr_stock:
            a = int(portfolio_value[i])/tot_new_value
            bounds2 += ((a,a),)
        else:
            a = 0
            b = max_add
            
            bounds2 += ((a,b),)
    
# Calculate modified portfolio ------------------------------------------------
#The modified portfolio will be maximum sharpe optimized with respect to the new stocks
    m_port = ef.msr2(0, np.repeat(1, total_stocks), cov, init_guess, bounds)
    m_ret = ef.portfolio_return(m_port, er)
    m_cov = ef.portfolio_vol(m_port, cov)
#Portfolio returns    
    portfolio_ret_m = pd.DataFrame(data=(m_port*df_rets).sum(axis='columns'),
                                    index=df_rets.index,columns=['Portfolio'])
#Summary stats for modified portfolio
    m_sum = ef.summary_stats(portfolio_ret_m, riskfree_rate=0.03)
#Wealth for modified portfolio
    df2['mod_portfolio_wealth'] = (portfolio_ret_m + 1).cumprod()
#Calc peaks and drawdown
    previous_peaks = df2['mod_portfolio_wealth'].cummax()
    dd_m = (df2['mod_portfolio_wealth'] - previous_peaks)/previous_peaks
#Calculate annualized return on the modified portfolio
    port_m_dev = df2.iloc[-1]['mod_portfolio_wealth']
    port_m_ann = 100*(port_m_dev**(1/day_diff) - 1)
#Calculate risk contributions of individual stocks in modified portfolio
    m_risk = ef.risk_contribution(m_port,cov)
#Create dataframe for the data
    df_m = m_risk.index.to_frame()    
    df_m['weights'] = m_port
    df_m['risk'] = m_risk
    df_m['value'] = df_m['weights'] * tot_new_value
    del df_m[0]
#Calculate Cornish Fisher VAR at 1%
    cf99_m = ef.var_gaussian(portfolio_ret_m, level=1, modified=False)[0]
    
#Calc weekly and daily stats for modified portfolio for plotting comparison
    portfolio_ret_w = pd.DataFrame(data=(m_port*df_rets_w).sum(axis='columns'),
                                    index=df_rets_w.index,columns=['Portfolio'])
    m_sum_w = ef.summary_stats(portfolio_ret_w, riskfree_rate=0.03)
    cf99_w = ef.var_gaussian(portfolio_ret_w, level=1, modified=False)[0]
    
    portfolio_ret_d1 = pd.DataFrame(data=(m_port*df_rets_d).sum(axis='columns'),
                                    index=df_rets_d.index,columns=['Portfolio'])
    m_sum_d = ef.summary_stats(portfolio_ret_d1, riskfree_rate=0.03)
    cf99_d = ef.var_gaussian(portfolio_ret_d1, level=1, modified=False)[0]
    
#Calculate global minimum variance portfolio of all stocks
    b = ef.gmv(cov) 
    b_ret = ef.portfolio_return(b, er)
    b_cov = ef.portfolio_vol(b, cov)

    portfolio_ret_b = pd.DataFrame(data=(b*df_rets).sum(axis='columns'),
                                    index=df_rets.index,columns=['Portfolio'])
    b_sum = ef.summary_stats(portfolio_ret_b, riskfree_rate=0.03)
    df2['portfolio_wealth'] = (portfolio_ret_b + 1).cumprod()
    previous_peaks = df2['portfolio_wealth'].cummax()
    dd_b = (df2['portfolio_wealth'] - previous_peaks)/previous_peaks
    b1 = ef.risk_contribution(b,cov)
    df_b = b1.index.to_frame()    
    df_b['weights'] = b
    df_b['risk'] = b1
    df_b['value'] = df_b['weights'] * tot_new_value
    del df_b[0]
     
    
#Calculate equal risk portfolio of all stocks
    d = ef.weight_erc(df_rets,cov_estimator=ef.sample_cov)
    d_ret = ef.portfolio_return(d, er)
    d_cov = ef.portfolio_vol(d, cov)   

    portfolio_ret_d = pd.DataFrame(data=(d*df_rets).sum(axis='columns'),
                                    index=df_rets.index,columns=['Portfolio'])
    d_sum = ef.summary_stats(portfolio_ret_d, riskfree_rate=0.03)
    df2['weight_erc_wealth'] = (portfolio_ret_d + 1).cumprod()
    previous_peaks = df2['weight_erc_wealth'].cummax()
    dd_d = (df2['weight_erc_wealth'] - previous_peaks)/previous_peaks
    c = ef.risk_contribution(d,cov)   
    df_c = c.index.to_frame()
    df_c['weights'] = d
    df_c['risk'] = c
    df_c['value'] = df_c['weights'] * tot_new_value
    del df_c[0]

#Calculate max sharpe portfolio of all stocks
    msr_port = ef.msr(0.03, er, cov)
    msr_ret = ef.portfolio_return(msr_port, er)
    msr_cov = ef.portfolio_vol(msr_port, cov)

    portfolio_ret_msr = pd.DataFrame(data=(msr_port*df_rets).sum(axis='columns'),
                                    index=df_rets.index,columns=['Portfolio'])
    msr_sum = ef.summary_stats(portfolio_ret_msr, riskfree_rate=0.03)
    df2['msr_wealth'] = (portfolio_ret_msr + 1).cumprod()
    previous_peaks = df2['msr_wealth'].cummax()
    dd_msr = (df2['msr_wealth'] - previous_peaks)/previous_peaks
    msr_risk = ef.risk_contribution(msr_port,cov)
    df_msr = msr_risk.index.to_frame()    
    df_msr['weights'] = msr_port
    df_msr['risk'] = msr_risk
    df_msr['value'] = df_msr['weights'] * tot_new_value
    del df_msr[0]


#Concat portfolio weights for bar graph
    df_stack = c.index.to_frame()
    df_stack['port_mod'] = m_port
    df_stack['port_b'] = b
    df_stack['port_c'] = d
    df_stack["msr_port"] = msr_port
    del df_stack[0]
    
#Concatenate summary data of all portfolios for plotting
    df_sum = m_sum.copy()
    df_sum.loc[1] = b_sum.iloc[0,:]
    df_sum.loc[2] = d_sum.iloc[0,:]
    df_sum.loc[3] = msr_sum.iloc[0,:]
 
### Plotting -----------------------------------------------------------------
#Make table tab with portfolio data summary

    s1 = sp.tabel_func(df_b, df_c, df_m, df_sum, total_stocks_list, total_stocks, 
                       eq_add, max_add, max_add_w_low_limits)
    tab_dt = Panel(child=s1, title="Process Variables")

###----------------------------------------------------------------------------
#Make portfollio dashboard

    s = sp.port_stat(df2, df_stack, df_b, df_c, df_m, df_msr, corr, port_m_ann, er, cov,
              init_guess, bounds2, min_ret, max_ret, m_cov, b_cov, d_cov, msr_cov,
              m_ret, b_ret, d_ret, msr_ret)
    tab1 = Panel(child=s, title="Portfolio Dash")

###----------------------------------------------------------------------------
#Make portfolio risk tab
    
    h_m = sp.hist_VAR_plot(portfolio_ret_m,-df_sum.iloc[0,4], 'Month', cf99_m)
    h_w = sp.hist_VAR_plot(portfolio_ret_w,-m_sum_w.iloc[0,4], 'Week', cf99_w)
    h_d = sp.hist_VAR_plot(portfolio_ret_d1,-m_sum_d.iloc[0,4], 'Day', cf99_d)
    dd = sp.port_dd(df2, dd_m, dd_b, dd_d)
    s_heat = column(dd, row(h_m, h_w, h_d))

    tab_heat = Panel(child = s_heat, title="Risk")    
    
###---------------------------------------------------------------------------
#Plot tabs
    tabs = Tabs(tabs=[tab_dt,tab1, tab_heat])   
    show(tabs)


    return redirect('http://localhost:5000/')



if __name__ == "__main__":
    webbrowser.open('http://localhost:5000/')
    app.run()
        


