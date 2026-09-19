from shiny import ui, render, reactive
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from src.config import AVAILABLE_INTERVALS, MANDATORY_CRYPTO, IGNORED_CRYPTO, API_BASE_URL
from src.logger import logger

def portfolio_allocation_ui():
    return ui.navset_card_underline(
        ui.nav_panel("State 1: Estimation",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Parameters"),
                    ui.input_selectize(
                        "pa_symbols",
                        "Select Tickers",
                        choices=MANDATORY_CRYPTO,
                        selected=MANDATORY_CRYPTO[:5],
                        multiple=True
                    ),
                    ui.input_select("pa_interval", "Interval", choices=AVAILABLE_INTERVALS, selected="1d"),
                    ui.input_numeric("pa_horizon", "Target Holding Time (h)", value=7, min=1),
                    ui.input_numeric("pa_short_window", "Short Window", value=14, min=2),
                    ui.input_numeric("pa_long_window", "Long Window", value=60, min=10),
                    ui.input_slider("pa_corr_alpha", "Correlation Alpha (Short Weight)", min=0.0, max=1.0, value=0.4, step=0.05),
                    ui.input_slider("pa_split_ratio", "Train/Test Split", min=0.1, max=0.9, value=0.6, step=0.05),
                    
                    ui.input_action_button("btn_pa_estimate", "Estimate Parameters", class_="btn-primary w-100 mt-3")
                ),
                ui.card(
                    ui.card_header("Horizon-Shifted OLS Estimates"),
                    ui.output_ui("pa_estimation_grid"),
                    ui.p("Note: Manually overriding expected volatility here will trigger a full covariance matrix rebuild (D*C*D).", class_="text-muted mt-2")
                ),
                ui.card(
                    ui.card_header("Validation Metrics (Out-of-Sample)"),
                    ui.output_data_frame("pa_validation_metrics")
                )
            )
        ),
        ui.nav_panel("State 2: Optimization",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Optimizer Settings"),
                    ui.input_select(
                        "pa_opt_method", 
                        "Objective", 
                        choices={"max_sharpe": "Max Sharpe", "min_variance": "Min Variance", "risk_parity": "Risk Parity"},
                        selected="max_sharpe"
                    ),
                    ui.input_numeric("pa_risk_free", "Risk-Free Rate (Annual %)", value=0.0, step=1.0),
                    ui.input_numeric("pa_target_vol", "Target Volatility (Annual %)", value=15.0, min=1.0, step=1.0),
                    
                    ui.h5("Constraints", class_="mt-3"),
                    ui.input_switch("pa_long_only", "Long Only (0 to 1)", value=True),
                    ui.input_numeric("pa_min_weight", "Min Weight (if Long Only)", value=0.0, min=0.0, max=1.0, step=0.05),
                    ui.input_numeric("pa_max_weight", "Max Weight (if Long Only)", value=1.0, min=0.0, max=1.0, step=0.05),
                    
                    ui.input_action_button("btn_pa_optimize", "Optimize Portfolio", class_="btn-success w-100 mt-3")
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Efficient Frontier"),
                        output_widget("pa_frontier_chart", height="400px")
                    ),
                    ui.card(
                        ui.card_header("Target Volatility Allocation"),
                        output_widget("pa_allocation_chart", height="400px")
                    ),
                    col_widths=[7, 5]
                ),
                ui.card(
                    ui.card_header("Allocation Details"),
                    ui.output_data_frame("pa_allocation_table")
                )
            )
        )
    )

def portfolio_allocation_server(input, output, session):
    
    # Store estimation results
    est_results = reactive.Value(None)
    
    # Store manually overridden parameters
    manual_mu = reactive.Value({})
    manual_vol = reactive.Value({})
    
    @reactive.Effect
    @reactive.event(input.btn_pa_estimate)
    def handle_estimate():
        req_data = {
            "symbols": list(input.pa_symbols()),
            "interval": input.pa_interval(),
            "target_horizon": input.pa_horizon(),
            "short_window": input.pa_short_window(),
            "long_window": input.pa_long_window(),
            "correlation_alpha": input.pa_corr_alpha(),
            "split_ratio": input.pa_split_ratio()
        }
        
        try:
            res = requests.post(f"{API_BASE_URL}/portfolio/estimate", json=req_data)
            if res.status_code == 200:
                data = res.json()
                est_results.set(data)
                
                # Initialize manual overrides with estimated values
                mu = {sym: data['assets'][sym]['expected_return'] for sym in data['assets']}
                vol = {sym: data['assets'][sym]['expected_volatility'] for sym in data['assets']}
                manual_mu.set(mu)
                manual_vol.set(vol)
                
                ui.notification_show("Estimation complete!", type="success")
            else:
                ui.notification_show(f"API Error: {res.text}", type="error")
        except Exception as e:
            ui.notification_show(f"Request failed: {str(e)}", type="error")

    @render.ui
    def pa_estimation_grid():
        data = est_results.get()
        if not data:
            return ui.p("Click 'Estimate Parameters' to begin.", class_="text-muted")
            
        assets = data['assets']
        
        # We will create a simple HTML table with inputs for overrides (since DataGrid doesn't natively support easy editing in Shiny for Python yet)
        
        rows = []
        for sym, metrics in assets.items():
            er = metrics['expected_return']
            vol = metrics['expected_volatility']
            mdd = metrics['expected_maxdd']
            
            # Using raw HTML inputs for editing, tied to reactive updates via JS or just read at optimize time.
            # For simplicity in V1, we'll display a static table and allow sliders/inputs if we wanted, 
            # but to make it truly editable we can use ui.input_numeric in a loop.
            
            row = ui.tags.tr(
                ui.tags.td(sym, class_="fw-bold"),
                ui.tags.td(ui.input_numeric(f"er_{sym}", "", value=round(er, 6), step=0.001, width="100px")),
                ui.tags.td(ui.input_numeric(f"vol_{sym}", "", value=round(vol, 6), step=0.001, min=0.0, width="100px")),
                ui.tags.td(f"{mdd:.4f}")
            )
            rows.append(row)
            
        table = ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("Asset"),
                    ui.tags.th("Exp. Return (Er)"),
                    ui.tags.th("Exp. Volatility"),
                    ui.tags.th("Exp. Max DD (Diag)")
                )
            ),
            ui.tags.tbody(*rows),
            class_="table table-striped table-sm align-middle"
        )
        return table
        
    @render.data_frame
    def pa_validation_metrics():
        data = est_results.get()
        if not data:
            return pd.DataFrame()
            
        rows = []
        for sym, metrics in data['assets'].items():
            val = metrics['validation']
            # We focus on the Return model validation as requested
            r_val = val['return']
            rows.append({
                "Asset": sym,
                "Model": "Return",
                "R2": r_val['r2'],
                "RMSE": r_val['rmse'],
                "MAE": r_val['mae'],
                "IC": r_val['ic']
            })
            
        df = pd.DataFrame(rows)
        return render.DataGrid(df.round(4))
        
    
    @reactive.Effect
    @reactive.event(input.btn_pa_optimize)
    def handle_optimize():
        data = est_results.get()
        if not data:
            ui.notification_show("Please run estimation first.", type="warning")
            return
            
        symbols = list(data['assets'].keys())
        
        # Read overrides from UI
        mu = {}
        vol = {}
        for sym in symbols:
            # We access the input values dynamically
            er_val = getattr(input, f"er_{sym}")()
            vol_val = getattr(input, f"vol_{sym}")()
            mu[sym] = float(er_val) if er_val is not None else data['assets'][sym]['expected_return']
            vol[sym] = float(vol_val) if vol_val is not None else data['assets'][sym]['expected_volatility']
            
        # Rebuild full covariance matrix (D * C * D)
        corr_matrix = pd.DataFrame(data['correlation'])
        vols_series = pd.Series(vol)
        symbols = corr_matrix.columns
        D = np.diag(vols_series.reindex(symbols).fillna(0.0).values)
        C = corr_matrix.values
        Sigma = D @ C @ D
        sigma_df = pd.DataFrame(Sigma, index=symbols, columns=symbols)
        
        req_data = {
            "mu": mu,
            "sigma": sigma_df.to_dict(),
            "method": input.pa_opt_method(),
            "risk_free_rate": input.pa_risk_free() / 100.0, # convert % to decimal
            "target_volatility": input.pa_target_vol() / 100.0,
            "long_only": input.pa_long_only(),
            "min_weight": input.pa_min_weight(),
            "max_weight": input.pa_max_weight()
        }
        
        try:
            res = requests.post(f"{API_BASE_URL}/portfolio/optimize", json=req_data)
            if res.status_code == 200:
                opt_data = res.json()
                opt_results.set(opt_data)
                ui.notification_show("Optimization complete!", type="success")
            else:
                ui.notification_show(f"API Error: {res.text}", type="error")
        except Exception as e:
            ui.notification_show(f"Request failed: {str(e)}", type="error")

    opt_results = reactive.Value(None)
    
    @render_widget
    def pa_frontier_chart():
        data = opt_results.get()
        if not data:
            return go.Figure()
            
        fig = go.Figure()
        
        # 1. Random cloud
        cloud = data['random_cloud']
        fig.add_trace(go.Scatter(
            x=cloud['volatilities'],
            y=cloud['returns'],
            mode='markers',
            marker=dict(color='rgba(150, 150, 150, 0.2)', size=4),
            name='Random Portfolios',
            showlegend=False
        ))
        
        # 2. Efficient frontier
        frontier = data['efficient_frontier']
        fig.add_trace(go.Scatter(
            x=frontier['volatilities'],
            y=frontier['returns'],
            mode='lines',
            line=dict(color='blue', width=3),
            name='Efficient Frontier'
        ))
        
        # 3. Individual Assets
        est_data = est_results.get()
        if est_data:
            assets = est_data['assets']
            for sym, metrics in assets.items():
                fig.add_trace(go.Scatter(
                    x=[metrics['expected_volatility']],
                    y=[metrics['expected_return']],
                    mode='markers+text',
                    text=[sym],
                    textposition="top center",
                    marker=dict(size=10, symbol='diamond'),
                    name=sym
                ))
                
        # 4. Optimized Portfolio (Pre-Targeting Volatility!)
        metrics = data['metrics']
        fig.add_trace(go.Scatter(
            x=[metrics['pre_scaled_volatility']],
            y=[metrics['portfolio_return_relative']],
            mode='markers',
            marker=dict(symbol='star', size=20, color='gold', line=dict(width=2, color='DarkSlateGrey')),
            name='Optimized Portfolio'
        ))
        
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Expected Volatility",
            yaxis_title="Expected Return",
            template="plotly_dark",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig
        
    @render_widget
    def pa_allocation_chart():
        data = opt_results.get()
        if not data:
            return go.Figure()
            
        final_w = data['final_weights']
        labels = list(final_w.keys())
        values = list(final_w.values())
        
        # Handle leverage/shorts gracefully in pie chart by using absolute values if needed
        # but long-only is default
        abs_values = [abs(v) for v in values]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=abs_values, 
            textinfo='label+percent',
            hole=0.4
        )])
        
        fig.update_layout(
            title=f"Allocation (Target Vol: {data['metrics']['target_volatility']*100:.1f}%)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        return fig
        
    @render.data_frame
    def pa_allocation_table():
        data = opt_results.get()
        if not data:
            return pd.DataFrame()
            
        rel_w = data['relative_weights']
        final_w = data['final_weights']
        rc = data['risk_contributions']
        
        rows = []
        for sym in rel_w.keys():
            rows.append({
                "Asset": sym,
                "Relative Weight": rel_w[sym],
                "Final Weight": final_w[sym],
                "Risk Contribution": rc.get(sym, 0.0)
            })
            
        df = pd.DataFrame(rows)
        return render.DataGrid(df.round(4))
