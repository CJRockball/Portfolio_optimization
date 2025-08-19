# Portfolio Optimization Tool

## Overview

This portfolio optimization tool implements modern portfolio theory (MPT) with a focus on adding new assets to existing portfolios. Built using Python, Flask, and Bokeh, it provides an interactive web interface for portfolio analysis and optimization using multiple risk-based allocation strategies.

**⚠️ Disclaimer: This software is for educational and testing purposes only. Do not use for actual portfolio optimization without thorough validation. Always double-check results and consult financial professionals.**

## Mathematical Foundation

### Core Optimization Problems

The tool implements several portfolio optimization approaches based on modern portfolio theory:

#### 1. Mean-Variance Optimization
- **Portfolio Return**: `R_p = w^T μ`
- **Portfolio Variance**: `σ²_p = w^T Σ w`
- **Portfolio Volatility**: `σ_p = √(w^T Σ w)`

Where `w` is the weight vector, `μ` is the expected returns vector, and `Σ` is the covariance matrix.

#### 2. Maximum Sharpe Ratio Portfolio
Maximizes the risk-adjusted return:

```
Sharpe Ratio = (R_p - R_f) / σ_p
```

**Optimization Problem**:
```
maximize: (w^T μ - r_f) / √(w^T Σ w)
subject to: Σw_i = 1, w_i ≥ 0
```

#### 3. Risk Contribution Analysis
Based on Euler's theorem for homogeneous functions:

- **Marginal Risk Contribution**: `MRC_i = ∂σ(w)/∂w_i = (Σw)_i / √(w^T Σ w)`
- **Risk Contribution**: `RC_i = w_i × MRC_i`
- **Total Risk Decomposition**: `σ(w) = Σ RC_i`

#### 4. Equal Risk Contribution (ERC)
Risk parity approach where each asset contributes equally to portfolio risk:

```
Find w such that: RC_i = RC_j for all i,j
Subject to: Σw_i = 1, w_i ≥ 0
```

#### 5. Value at Risk (VaR)
- **Parametric VaR**: `VaR_α = μ - z_α × σ`
- **Cornish-Fisher VaR**: Adjusts for skewness and kurtosis in return distributions

## Features

### Portfolio Strategies
1. **Modified Portfolio**: Optimized allocation when adding new stocks to existing holdings
2. **Maximum Sharpe Ratio**: Unconstrained optimization for highest risk-adjusted returns
3. **Global Minimum Variance**: Lowest risk portfolio allocation
4. **Equal Risk Contribution**: Risk parity approach with balanced risk budgets

### Risk Analysis
- **Value at Risk (VaR)**: 1%, 5% confidence levels with Cornish-Fisher adjustments
- **Conditional VaR (CVaR)**: Expected shortfall analysis
- **Maximum Drawdown**: Peak-to-trough loss measurement
- **Risk Contribution**: Individual asset risk decomposition
- **Higher Moments**: Skewness and kurtosis analysis

### Visualization Dashboard
- **Efficient Frontier**: Risk-return optimization curves
- **Wealth Evolution**: Time series performance comparison
- **Risk/Weight Analysis**: Bar charts comparing allocations and risk contributions
- **Portfolio Allocation**: Stacked visualization of asset weights
- **Correlation Matrix**: Heatmap of asset correlations
- **Drawdown Analysis**: Time series of portfolio declines
- **VaR Histograms**: Return distribution with risk overlays

## Installation and Setup

### Prerequisites
- Python 3.7+
- Required packages: `flask`, `numpy`, `pandas`, `scipy`, `bokeh`, `pandas-datareader`

### Installation Steps
1. Clone or download all files to the same directory
2. Create a `template` folder and move `index_portfolio.html` into it
3. Install required dependencies:
   ```bash
   pip install flask numpy pandas scipy bokeh pandas-datareader
   ```
4. Run the application:
   ```bash
   python portfolio.py
   ```
5. Open browser to `http://localhost:5000`

## Usage Guide

### Input Parameters
1. **Existing Portfolio**: Enter current stock symbols and their values in the first two columns
2. **New Stocks**: Add prospective stocks in the third column
3. **Value Adding**: Total cash amount for new investments
4. **Weight Constraints**: Optional minimum and maximum allocation percentages
5. **Date Range**: Historical data period for analysis (format: YYYY-MM-DD)

### Workflow
1. **Initial Run**: Execute without weight constraints to see possible allocation ranges
2. **Constraint Setting**: Use the calculated ranges to set reasonable min/max weights
3. **Re-optimization**: Run again with refined constraints for final allocation
4. **Analysis**: Review results across multiple tabs (Tables, Dashboard, Risk Analysis)

### Output Interpretation
- **Process Variables Tab**: Portfolio composition and summary statistics
- **Portfolio Dashboard Tab**: Visual analysis with efficient frontier and performance charts
- **Risk Tab**: Detailed risk analysis including VaR measures and drawdown plots

## Technical Implementation

### Architecture
- **Flask Backend**: Web server and request handling
- **Core Mathematics**: `ec_func.py` - Financial calculations and optimization
- **Visualization**: `stock_plot.py` - Bokeh charts and interactive plots
- **Frontend**: HTML template with form-based user interface

### Optimization Engine
- **Solver**: SciPy's SLSQP (Sequential Least Squares Programming)
- **Constraints**: Equality (sum weights = 1) and inequality (weight bounds)
- **Objective Functions**: Maximum Sharpe ratio, minimum variance, equal risk contribution

### Data Pipeline
1. **Data Source**: Yahoo Finance via pandas-datareader
2. **Frequency Conversion**: Daily → Weekly → Monthly returns
3. **Risk Calculation**: Covariance matrix estimation
4. **Return Processing**: Percentage changes and compounding

## Limitations and Considerations

### Mathematical Limitations
- **Historical Data Dependency**: Optimization based on past performance
- **Normal Distribution Assumption**: May not capture tail risks adequately
- **Single Period Model**: No consideration of dynamic rebalancing
- **No Transaction Costs**: Ignores real-world trading frictions

### Technical Limitations
- **External Data Dependency**: Relies on Yahoo Finance availability
- **Fixed Risk-Free Rate**: Hard-coded at 3% instead of dynamic rates
- **Limited Error Handling**: Minimal validation of optimization convergence
- **No Backtesting**: No out-of-sample performance validation

### Practical Considerations
- **Market Regime Changes**: Historical correlations may not persist
- **Liquidity Constraints**: No consideration of trading volumes or market impact
- **Regulatory Constraints**: No incorporation of position limits or compliance rules
- **Tax Implications**: No optimization for tax efficiency

## File Structure

```
├── portfolio.py              # Main Flask application
├── ec_func.py               # Financial mathematics and optimization
├── stock_plot.py            # Bokeh visualization functions
├── template/
│   └── index_portfolio.html # Web interface template
├── images/                  # Application screenshots
│   ├── dashboard.png
│   ├── portfolio_input.png
│   ├── risk.png
│   └── table.png
└── README.md               # This file
```

## Mathematical Validation

The implementation follows established financial mathematics principles:

- **Markowitz Mean-Variance Framework**: Classic portfolio optimization theory
- **Sharpe Ratio Maximization**: Standard risk-adjusted performance measure
- **Euler's Theorem**: Proper risk decomposition for homogeneous risk functions
- **Lagrange Multipliers**: Constrained optimization with equality and inequality constraints

## Future Enhancements

### Mathematical Improvements
- **Black-Litterman Model**: Incorporate investor views and market equilibrium
- **Robust Optimization**: Handle parameter uncertainty with confidence intervals
- **Multi-Period Optimization**: Dynamic rebalancing with transaction costs
- **Alternative Risk Measures**: CVaR optimization, maximum drawdown constraints

### Technical Improvements
- **Real-Time Data**: Integration with live market data feeds
- **Backtesting Engine**: Out-of-sample performance validation
- **Monte Carlo Simulation**: Scenario analysis and stress testing
- **API Development**: RESTful interface for programmatic access
- **Database Integration**: Historical data storage and management

### User Experience
- **Responsive Design**: Mobile-friendly interface
- **Export Functionality**: PDF reports and CSV data downloads
- **User Authentication**: Portfolio saving and sharing capabilities
- **Advanced Charting**: Interactive financial charts with technical indicators

## References

1. Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.
2. Sharpe, W. F. (1966). Mutual Fund Performance. Journal of Business, 39(1), 119-138.
3. Maillard, S., Roncalli, T., & Teïletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios. Journal of Portfolio Management, 36(4), 60-70.
4. Roncalli, T. (2013). Introduction to Risk Parity and Budgeting. Chapman and Hall/CRC.

## License

This project is provided as-is for educational purposes. Users are responsible for validating all calculations and should not rely on this tool for actual investment decisions without proper due diligence and professional consultation.

---

**Contact**: This is an open-source educational project. Please verify all mathematical implementations before use in any real-world applications.