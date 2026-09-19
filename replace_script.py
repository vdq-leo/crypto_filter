import re

with open('modules/predictive.py', 'r') as f:
    content = f.read()

start_marker = "    @reactive.event(input.btn_run_analysis)\n    async def run_analysis():"
end_marker = "    @reactive.calc\n    def labeling_result():"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Markers not found!")
    exit(1)

new_func = """    @reactive.event(input.btn_run_analysis)
    async def run_analysis():
        \"\"\"Run ML analysis via API\"\"\"
        direction = input.trade_direction()
        tickers = inventory.get()
        if not tickers:
            ui.notification_show("Please add at least one ticker.", type="warning")
            return
            
        interval = input.interval()
        features = list(input.eng_features())
        if not features:
            ui.notification_show("Please select features in Data Engineering.", type="warning")
            return
            
        v_sync = vol_th_sync.get()
        vol_th = v_sync if v_sync is not None else input.vol_bar_th()
        d_sync = dollar_th_sync.get()
        dollar_th = d_sync if d_sync is not None else input.dollar_bar_th()
            
        req_data = {
            "tickers": tickers,
            "interval": interval,
            "trade_direction": direction,
            "features": features,
            "reg_type": input.reg_type(),
            "test_ratio": input.test_ratio(),
            "rf_max_depth": input.rf_max_depth(),
            "eng_lookback": input.eng_lookback(),
            "pred_lookback": input.pred_lookback(),
            "min_samples": input.min_samples(),
            "standardize_flag": input.standardize_flag(),
            "vif_th": input.vif_th(),
            "bar_type": input.bar_type(),
            "vol_th": int(vol_th),
            "dollar_th": int(dollar_th),
            "labeler_type": input.labeler_type(),
            "labeler_params": {
                "amp_th": input.labeler_amp_th(),
                "max_inactive": input.labeler_max_inactive(),
                "vol_window": input.tbm_vol_window(),
                "upper_mult": input.tbm_upper_mult(),
                "lower_mult": input.tbm_lower_mult(),
                "max_holding": input.tbm_max_holding(),
                "stat_window": input.stat_window(),
                "vote_th": input.stat_vote_th(),
                "tail_window": input.tail_window(),
                "tail_threshold": input.tail_threshold()
            },
            "max_positions": input.max_positions(),
            "tp_mult_bt": input.tp_mult_bt(),
            "sl_mult_bt": input.sl_mult_bt(),
            "min_holding_bt": input.min_holding_bt(),
            "max_holding_bt": input.max_holding_bt()
        }
        
        ui.notification_show("Running ML Analysis via API...", duration=5)
        
        try:
            import requests
            from src.config import API_BASE_URL
            import pandas as pd
            res = requests.post(f"{API_BASE_URL}/predictive/analyze", json=req_data)
            if res.status_code == 200:
                data = res.json()
                
                # Reconstruct pandas DataFrame and Series objects from the JSON strings
                import json
                
                def restore_df(json_str):
                    if not json_str: return None
                    return pd.read_json(json.dumps(json_str), orient="split")
                def restore_series(json_str):
                    if not json_str: return None
                    return pd.read_json(json.dumps(json_str), orient="split", typ="series")
                
                data["X_test"] = restore_df(data["X_test"])
                data["y_test"] = restore_series(data["y_test"])
                data["y_test_preds"] = restore_series(data["y_test_preds"])
                
                if data["shap_results"] is not None:
                    data["shap_results"] = restore_df(data["shap_results"])
                
                results.set(data)
                ui.notification_show("Analysis Complete!", type="message", duration=5)
            else:
                ui.notification_show(f"API Error: {res.text}", type="error", duration=10)
        except Exception as e:
            logger.log("Predictive", "ERROR", f"API Error: {e}")
            ui.notification_show(f"Connection Error: {e}", type="error")

"""

new_content = content[:start_idx] + new_func + content[end_idx:]

with open('modules/predictive.py', 'w') as f:
    f.write(new_content)
    
print("Successfully replaced run_analysis")
