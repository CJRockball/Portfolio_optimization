# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:58:58 2020

@author: Snorlax
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from math import pi
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, LinearAxis, Range1d,\
     Label, Span, BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter, Label
from bokeh.models.widgets import DataTable, TableColumn, Paragraph
from bokeh.transform import dodge
import ec_func as ef


#VAR plot with histogram and normal overlay
def hist_VAR_plot(portfolio_ret_m,CornFish_var, title_text, cf99):
    pmin = portfolio_ret_m.Portfolio.min()
    pmax = portfolio_ret_m.Portfolio.max()
    p_diff = abs(pmax) + abs(pmin)
    
#Plot histogram blue
    hist, edges = np.histogram(portfolio_ret_m.Portfolio,  density=True, bins = 40)
    title_header = str(title_text) + ' Hist VAR 5%'
    h = figure(y_range = [0,max(hist)+5],title= title_header, tools='',y_axis_label = "Count",
               plot_width=500, plot_height=400, background_fill_color="#fafafa")
    h.quad(bottom = 0, top = hist,left = edges[:-1], 
            right = edges[1:], fill_color = "navy", 
            line_color = "white", fill_alpha = 0.7)
    
#Mark low 5% red
    hist_var = -np.percentile(portfolio_ret_m, 95)
    risk_edges = edges[edges < hist_var]
    risk_hist = hist[0:len(risk_edges)-1]
    h.quad(bottom = 0, top = risk_hist,left = risk_edges[:-1], 
            right = risk_edges[1:], fill_color = "red", 
            line_color = "white", fill_alpha = 0.8)
    
#Calc and plot Norm 
    mu = portfolio_ret_m.Portfolio.mean()
    sigma = portfolio_ret_m.Portfolio.std()
    x = np.linspace(pmin-0.5*p_diff, pmax+0.5*p_diff, 1000)
    pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    h.extra_y_ranges = {"foo": Range1d(start=0, end=1.1*max(pdf))}
    h.add_layout(LinearAxis(y_range_name="foo",axis_label='Prob',
                            axis_label_text_color = 'black'), 'right')

#Add shading for 5%
    x2 = np.linspace(pmin-0.5*p_diff, mu-1.645*sigma, 1000)
    y2 = 1/(sigma * np.sqrt(1.5*np.pi)) * np.exp(-(x2-mu)**2 / (2*sigma**2))
    h.varea(x=x2,y1=0, y2=y2,alpha=0.3, color='red',y_range_name="foo")
    
    h.line(x, pdf, line_color="orange", line_width=4, alpha=0.7, 
           legend_label="Gaussian", y_range_name="foo")
       
    text_string = 'Corn Fish VAR (5%): ' + str(round(-CornFish_var*100,2))+ '%'
    text_string2 = "Corn Fish VAR (1%): " + str(round(cf99*100,2)) + '%' 
    #+ 'num data points' + str(len(portfolio_ret_m))

    mytext = Label(x=10, y=325, x_units='screen', y_units='screen', text= text_string)
    mytext2 = Label(x=10, y=300, x_units='screen', y_units='screen',text= text_string2)
    h.add_layout(mytext)
    h.add_layout(mytext2)
    
#Calc and plot Normal with skew and kurt
    # mu2 = portfolio_ret_m.Portfolio.mean()
    # var2 = portfolio_ret_m.Portfolio.std()**2
    # skew2 = df_sum.iloc[0,2]
    # kurt2 = df_sum.iloc[0,3]     
    # f = extras.pdf_mvsk([mu2, var2, skew2, kurt2])
    # pdf2 = [f(i) for i in x]    
    # h.line(x, pdf2,line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF", y_range_name="foo")

#Mixed additional properties    
    # h.y_range.start = 0
    # h.legend.location = "center_right"
    # h.legend.background_fill_color = "#fefefe"
    # h.xaxis.axis_label = 'x'
    # h.yaxis.axis_label = 'Pr(x)'
    # h.grid.grid_line_color="white"

    return h

#Portfolio drawdown-----------------------------------------------------------
def port_dd(df2, dd_m, dd_b, dd_d):

    dd = figure(x_axis_type="datetime", plot_width=1200, plot_height = 300,
                toolbar_location=None, tools="", y_axis_label = "Portfolio drawdown", 
                x_axis_label = "Date")
    
    dd.line(df2['Date'], dd_m, color='blue', legend_label = 'Mod Port')
    dd.line(df2['Date'], dd_b, color='orange', legend_label = 'GMV Port')
    dd.line(df2['Date'], dd_d, color='red', legend_label = 'ERC Port')
    dd.title.text = str('Drawdown')
    dd.legend.location = "bottom_left"
    
    return dd

#Plot bar chart of portfolio weights and risk ---------------------------------
def risk_weight_plot(df, port_type):
    fmax = max(df.risk.max(), df.weights.max())
    data2 = {'names' : df.index.values,
            'risk' : df.risk,
            'weights' : df.weights}
    source2 = ColumnDataSource(data=data2)  
    f = figure(x_range = df.index.values, y_range=(0, fmax+0.1), plot_height = 300, 
               plot_width = 350, toolbar_location=None, tools="", title = 'Stock Risk Contribution')
    f.vbar(source = source2, x=dodge('names', -0.1, range = f.x_range), 
           top = 'risk', width = 0.2, color = 'aqua', legend_label = 'Risk') 
    f.vbar(source = source2, x=dodge('names', 0.1, range = f.x_range), 
           top = 'weights', width = 0.2, color = 'coral', legend_label = 'Weight') 
    f.title.text = str(port_type) + ' Port Stock Weight and Risk Contribution'
    f.legend.orientation = "horizontal"
    f.xaxis.major_label_orientation = pi/4

    return f


#Table tab function
def tabel_func(df_b, df_c, df_m, df_sum, total_stocks_list, total_stocks, eq_add, 
               max_add, max_add_w_low_limits):

#Make table of portfolio weightings and values
    gmvweight = df_b['weights']*100
    ercweight = df_c['weights']*100
    modweight = df_m['weights']*100
    
    data_tf = {'Stocks': total_stocks_list,
            'GMV': gmvweight.round(2),
            'GMV Value': df_b['value'].round(2),
            'ERC': ercweight.round(2),
            'ERC Value' : df_c['value'].round(2),
            'Mod Port': modweight.round(2),
            'Mod Port Value' : df_m['value'].round(2)}
    
       
    source = ColumnDataSource(data_tf)
    
    columns = [
            TableColumn(field="Stocks", title="Stocks"),
            TableColumn(field="Mod Port", title="Mod Port %"),
            TableColumn(field="Mod Port Value", title="Mod Port Value"),
            TableColumn(field="GMV", title="GMV %"),
            TableColumn(field="GMV Value", title="GMV Value"),
            TableColumn(field="ERC", title="ERC %"),
            TableColumn(field="ERC Value", title="ERC Value"),
                        ]
    
    data_table = DataTable(source=source, columns=columns, width=1000,
                           height=total_stocks*40,fit_columns=True)

###----------------------------------------------------------------------------    
#Table with portfolio properties
    port_names = ['Modefied Port', 'GMV', 'ERC', 'MSR']
    
    datas = {'port_names': port_names,
            'ann ret': df_sum.iloc[:,0].round(2),
            'ann vol': df_sum.iloc[:,1].round(2),
            'Skew': df_sum.iloc[:,2].round(2),
            'Kurt': df_sum.iloc[:,3].round(2),
            'Corn_fish_VAR' : round(100*df_sum.iloc[:,4],2),
            'Hist_CVAR': round(100*df_sum.iloc[:,5],2),
            'Sharp': df_sum.iloc[:,6].round(2),
            'Max_DD' : df_sum.iloc[:,7].round(2)}
    
       
    source_sum = ColumnDataSource(datas)
    
    columns_sum = [
            TableColumn(field="port_names", title="Portfolio Names"),
            TableColumn(field="ann ret", title="Ann Return"),
            TableColumn(field="ann vol", title="Ann Vol"),
            TableColumn(field="Skew", title="Skew"),
            TableColumn(field="Kurt", title="Kurtosis"),
            TableColumn(field="Corn_fish_VAR", title="Corn Fisher VAR (5%)"),
            TableColumn(field="Hist_CVAR", title="Hist CVAR (5%)"),
            TableColumn(field="Sharp", title="Sharpe Ratio"),
            TableColumn(field="Max_DD", title="Max DrawDown"),
                        ]
    
    data_table2 = DataTable(source=source_sum, columns=columns_sum, width=1000,
                            height=300,fit_columns=True)    
###-----------------------------------------------------------
#Given new value added and min restraint, calculate and print range of max weight
#for new stocks
    
    eq_add_proc = eq_add*100
    max_add_w_low_limits_proc = max_add_w_low_limits * 100
    mytext = str('max weight add is from: ') + str(round(eq_add_proc,2)) + \
                str('% to: ')+ str(round(max_add_w_low_limits_proc,2)) + str('%')
    pre = Paragraph(text=mytext,width=300, height=100)

    s1 = column(row(data_table,pre),data_table2)
    
    return s1
    

#Dash tab function
def port_stat(df2, df_stack, df_b, df_c, df_m, df_msr, corr, port_m_ann, er, cov,
              init_guess, bounds2, min_ret, max_ret, m_cov, b_cov, d_cov, msr_cov,
              m_ret, b_ret, d_ret, msr_ret):

#Portfolio growth--------------------------------------------------------------
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    
    p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=800, plot_height = 300,
                toolbar_location="right", y_axis_label = "Portfolio growth", x_axis_label = "Date")
    
    p.line(df2['Date'], df2['mod_portfolio_wealth'], color='green', legend_label = 'Mod')
    p.line(df2['Date'], df2['portfolio_wealth'], color='orange', legend_label = 'GMV')
    p.line(df2['Date'], df2['weight_erc_wealth'], color='red', legend_label = 'ERC')
    p.line(df2['Date'], df2['msr_wealth'], color='magenta', legend_label = 'MSR')
    p.title.text = str('Portfolio development')
    p.xaxis.major_label_orientation = pi/4
    p.legend.location = "top_left"
    text_string = 'Annualized rate for mod portfolio: ' + str(round(port_m_ann, 2)) + '%'
    mytext = Label(x=100, y=180, x_units='screen', y_units='screen', 
                   text= text_string)
    p.add_layout(mytext)
    
## Graph efficient frontier----------------------------------------------------
# Efficient frontier without any restraints
    weights = ef.optimal_weights(20, er, cov)
    rets = [ef.portfolio_return(w, er) for w in weights]
    vols = [ef.portfolio_vol(w, cov) for w in weights]
    front1 = pd.DataFrame({"Returns": rets, "Volatility": vols})

#Efficient frontier for modified portfolio
    weights2 = ef.optimal_weights2(10, er, cov, init_guess, bounds2, min_ret, max_ret)
    rets2 = [ef.portfolio_return(w, er) for w in weights2]
    vols2 = [ef.portfolio_vol(w, cov) for w in weights2]
    front2 = pd.DataFrame({"Returns": rets2, "Volatility": vols2})

    #TOOLS = "pan,wheel_zoom,box_zoom,reset,save"    
    g = figure(plot_height = 300, y_axis_label = "Portfolio Growth", x_axis_label = "Volatility",
               toolbar_location="left")
    
    g.line(front1['Volatility'], front1['Returns'], color='blue')
    g.circle(front1['Volatility'], front1['Returns'], color='blue', radius=0.0002)
    g.line(front2['Volatility'], front2['Returns'], color='green')
    g.circle(front2['Volatility'], front2['Returns'], color='green', radius=0.0002)
    g.title.text = str('Efficent Frontier')
    g.circle(x=m_cov, y=m_ret, color = 'lightgreen', radius = 0.0003, legend_label = 'Mod')
    g.circle(x=b_cov, y=b_ret, color = 'orange', radius = 0.0003, legend_label = 'GMV')
    g.circle(x=d_cov, y=d_ret, color = 'red', radius = 0.0003, legend_label = 'ERC')
    g.circle(x=msr_cov, y=msr_ret, color = 'magenta', radius = 0.0003, legend_label = 'MSR')
    g.legend.location = "top_right"
    
    
#Plot bar of portfolio weights ------------------------------------------------
    
    rb = risk_weight_plot(df_b, "GMV")
    f = risk_weight_plot(df_c, "ERC")
    rm = risk_weight_plot(df_m, "Mod Port")
    msrp = risk_weight_plot(df_msr, "MSR")
    
#Portfolio structure
    portf = ['mod port', 'GMV', 'ERC', 'MSR']
    stock = df_stack.index.tolist()
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
    '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324',
    '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
    '#ffffff', '#000000']
    
    use_color = colors[:len(df_stack)]
    
    data3 = {'portf' : portf}
    
    for i in range(len(df_stack)):
        key = df_stack.index[i]
        value = df_stack.iloc[i,:].tolist()
        data3[key] = value
    
    source3 = ColumnDataSource(data=data3)  
    
    p1 = figure(x_range=portf, plot_width=750, plot_height=300,  title="Portfolios",
               toolbar_location=None, tools="")
    
    p1.vbar_stack(stock, x='portf', width=0.3, color=use_color, source=source3)#,
#                 legend_label=stock)
    
    p1.y_range.start = 0
    p1.x_range.range_padding = 0.1
    p1.xgrid.grid_line_color = None
    p1.axis.minor_tick_line_color = None
    p1.outline_line_color = None
#    p1.legend.location = "top_right"
    p1.legend.label_text_font_size = "8px"

#Corralation graph ------------------------------------------------------------  
     # this is the colormap from the original NYTimes plot
    y_stock = corr.index.to_list()
    x_stock = corr.columns.to_list()
    corr_df = pd.DataFrame(corr.stack(), columns=['corrs']).reset_index()
    corr_df.columns = ['X_stock', 'Y_stock', 'corrs']
    corr_df = corr_df.round(2)
    
    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    mapper = LinearColorMapper(palette=colors, low=corr_df.corrs.min(), high=corr_df.corrs.max())
    
    
    hm = figure(title="Covariance",x_range=x_stock, y_range=list(reversed(y_stock)),
               x_axis_location="above", plot_width=700, plot_height=300,
               tools="", toolbar_location=None)
    
    hm.grid.grid_line_color = None
    hm.axis.axis_line_color = None
    hm.axis.major_tick_line_color = None
    hm.axis.major_label_text_font_size = "9px"
    hm.axis.major_label_standoff = 0
    hm.xaxis.major_label_orientation = pi / 6
    
    hm.rect(x="X_stock", y="Y_stock", width=1, height=1,
           source=corr_df,
           fill_color={'field': 'corrs', 'transform': mapper},
           line_color=None)
    
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="7px",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%.2f"),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    hm.add_layout(color_bar, 'right')

    hm.text(x='X_stock', y='Y_stock', text='corrs', source=corr_df,
            text_align="center", text_baseline="middle")

    s = column(row(g,p),row(rm,rb,f,msrp),row(p1,hm))
    
    return s







