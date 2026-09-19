
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

class StationarityLabeler:
    def __init__(self, window=100, back_span=100, vote_th=0.4, adf_alpha=0.3):
        self.window = window
        self.back_span = back_span if back_span is not None else window
        self.vote_th = vote_th
        self.adf_alpha = adf_alpha
        
        # hardcoded heuristic thresholds
        self.mean_drift_th = 0.5
        self.var_ratio_th = 2
        self.trend_th = 0.2
        self.acf_sum_th = 2.5
        self.range_ratio_th = 3

    def _acf_sum(self, x, k=100):
        x = x - x.mean()
        denom = np.dot(x, x)
        if denom == 0:
            return 0.0
        return sum(abs(np.dot(x[:-i], x[i:]) / denom) for i in range(1, min(k + 1, len(x))))

    def _stationarity_score(self, x):
        w = len(x)
        roll_mean = pd.Series(x).rolling(max(1, w // 3)).mean().dropna()
        roll_std = pd.Series(x).rolling(max(1, w // 3)).std().dropna()
        mean_drift = (roll_mean.max() - roll_mean.min()) / np.std(x)
        var_ratio = roll_std.max() / roll_std.min()
        t = np.arange(w)
        beta = np.polyfit(t, x, 1)[0]
        resid = x - (beta * t + np.mean(x))
        trend_strength = abs(beta) * w / np.std(resid)
        acf_strength = self._acf_sum(x)
        r1 = x[: w // 2].max() - x[: w // 2].min()
        r2 = x[w // 2 :].max() - x[w // 2 :].min()
        range_ratio = r2 / r1 if r1 > 0 else np.inf
        votes = [
            mean_drift < self.mean_drift_th,
            var_ratio < self.var_ratio_th,
            trend_strength < self.trend_th,
            acf_strength < self.acf_sum_th,
            range_ratio < self.range_ratio_th
        ]
        return np.mean(votes)

    def _adf_pvalue(self, x):
        try:
            return adfuller(x, autolag="AIC")[1]
        except:
            return 1.0

    def label(self, series):
        series = pd.Series(series)
        n = len(series)
        labels = np.zeros(n)
        scores = np.full(n, np.nan)

        for t in range(self.window - 1, n):
            x = series.iloc[t - self.window + 1 : t + 1].values
            scores[t] = self._stationarity_score(x)
            adf_pass = self._adf_pvalue(x) < self.adf_alpha
            if scores[t] >= self.vote_th and adf_pass:
                start = max(0, t - self.back_span + 1)
                labels[start : t + 1] = 1

        return pd.Series(labels, index=series.index), pd.Series(scores, index=series.index)


class Labeler:
    """
    Labels upward and downward momentum (or trend) movements where:
    - price movement amplitude exceeds a threshold
    - movement establishes a new high or low within a maximum inactivity period
    - movement is not invalidated by a reversal beyond the threshold
    """

    def __init__(self, amplitude_threshold: float, max_inactive_period: int):
        """
        :param amplitude_threshold: minimum movement required to define a trend (bps or %)
        :param max_inactive_period: maximum number of samples allowed without extending the trend
        """
        self.amplitude_threshold = float(amplitude_threshold)
        self.max_inactive_period = int(max_inactive_period)
        self.df = None

    # =====================================================
    def label(self, prices: pd.Series) -> pd.DataFrame:
        """
        Perform labeling on a price series.

        :param prices: pandas Series of prices or cumulative returns
        :return: DataFrame with 'price' and 'label'
        """
        prices = pd.Series(prices).astype(float).dropna().reset_index(drop=True)

        cumr = np.log(prices / prices.iloc[0]) * 1e4  # convert to cumulative basis points
        n = len(cumr)
        labels = np.zeros(n, dtype=float)

        self._pass1(cumr, labels)
        self._filter(cumr, labels)

        self.df = pd.DataFrame({
            'price': cumr,
            'label': labels
        })
        return self.df

    def _pass1(self, cumr: np.ndarray, labels: np.ndarray):
        """
        First-pass trend labeling using amplitude (minamp) and inactivity (Tinactive).

        Logic:
        - Detects new uptrends or downtrends when price moves beyond minamp from the last extreme.
        - Ends a trend if price goes flat too long (Tinactive samples) or reverses enough.
        """

        n = len(cumr)
        if n == 0:
            return

        # --- Initialize tracking variables ---
        start_idx = 0              # Start of current segment
        cursor = 0                 # Current position in the time series
        min_idx = max_idx = 0      # Indices of local min and max
        min_val = max_val = cumr[0]

        while cursor < n:
            v = cumr[cursor]  # current value

            # ==========================================================
            # 1️⃣ Direction Reversal Detection
            # ==========================================================

            # Case 1: Previously falling (min after max) → now rising enough
            if (max_val - min_val) >= self.amplitude_threshold  and min_idx > max_idx and (v - min_val) >= self.amplitude_threshold :
                # label the last downward move
                self._apply_label(labels, start_idx, max_idx - 1, 0.0)
                self._apply_label(labels, max_idx, min_idx, -1.0)
                # start new upward leg
                start_idx = min_idx
                max_idx = cursor
                max_val = v

            # Case 2: Previously rising (max after min) → now falling enough
            elif (max_val - min_val) >= self.amplitude_threshold  and max_idx > min_idx and (max_val - v) >= self.amplitude_threshold :
                # label the last upward move
                self._apply_label(labels, start_idx, min_idx - 1, 0.0)
                self._apply_label(labels, min_idx, max_idx, +1.0)
                # start new downward leg
                start_idx = max_idx
                min_idx = cursor
                min_val = v

            # ==========================================================
            # 2️⃣ Inactivity Handling
            # ==========================================================

            # (a) Uptrend but flat for too long
            elif max_idx > min_idx and (cursor - max_idx) >= self.max_inactive_period  and v <= max_val:
                if (max_val - min_val) >= self.amplitude_threshold :
                    # Up move confirmed, then flat
                    self._apply_label(labels, start_idx, min_idx - 1, 0.0)
                    self._apply_label(labels, min_idx, max_idx, +1.0)
                    self._apply_label(labels, max_idx + 1, cursor, 0.0)
                else:
                    # Weak move — just flat
                    self._apply_label(labels, start_idx, cursor, 0.0)

                start_idx = cursor
                min_idx = max_idx = cursor
                min_val = max_val = v

            # (b) Downtrend but flat for too long
            elif min_idx > max_idx and (cursor - min_idx) >= self.max_inactive_period  and v >= min_val:
                if (max_val - min_val) >= self.amplitude_threshold :
                    # Down move confirmed, then flat
                    self._apply_label(labels, start_idx, max_idx - 1, 0.0)
                    self._apply_label(labels, max_idx, min_idx, -1.0)
                    self._apply_label(labels, min_idx + 1, cursor, 0.0)
                else:
                    self._apply_label(labels, start_idx, cursor, 0.0)

                start_idx = cursor
                min_idx = max_idx = cursor
                min_val = max_val = v

            # ==========================================================
            # 3️⃣ Update Local Extremes
            # ==========================================================
            if v >= max_val:
                max_idx = cursor
                max_val = v
            if v <= min_val:
                min_idx = cursor
                min_val = v

            cursor += 1

        # ==========================================================
        # 4️⃣ Finalize labeling for end of series
        # ==========================================================
        if (max_val - min_val) >= self.amplitude_threshold  and min_idx > max_idx:
            self._apply_label(labels, start_idx, max_idx - 1, 0.0)
            self._apply_label(labels, max_idx, min_idx, -1.0)
            self._apply_label(labels, min_idx + 1, cursor - 1, 0.0)
        elif (max_val - min_val) >= self.amplitude_threshold  and max_idx > min_idx:
            self._apply_label(labels, start_idx, min_idx - 1, 0.0)
            self._apply_label(labels, min_idx, max_idx, +1.0)
            self._apply_label(labels, max_idx + 1, cursor - 1, 0.0)
        else:
            self._apply_label(labels, start_idx, cursor - 1, 0.0)

    def _filter(self, cumr: np.ndarray, labels: np.ndarray):
        """
        Refine the raw trend labels using OLS-based filtering.

        This step removes weak or inconsistent moves.
        For each contiguous region of +1 or -1 labels:
        - Fit OLS slopes forward and backward
        - Measure how strong the directional trend is
        - Keep only strong sections; set weak ones to 0 (neutral)

        Parameters
        ----------
        cumr : np.ndarray
            Cumulative returns (e.g., log(price) * 1e4)
        labels : np.ndarray
            Array of labels (+1 for uptrend, -1 for downtrend, 0 for neutral)
        """

        n = len(cumr)
        pos = 0  # current position in array

        while pos < n:
            direction = labels[pos]

            # Skip neutral regions
            if direction == 0.0:
                pos += 1
                continue

            # -----------------------------
            # Identify contiguous labeled region
            # -----------------------------
            start = pos
            end = pos
            while end < n and labels[end] == direction:
                end += 1
            end -= 1  # move back to last matching index

            # -----------------------------
            # Measure OLS slope strength
            # -----------------------------
            forward_strength, forward_max_idx = self._ols_distance(
                cumr, start, end, direction, forward=True
            )
            backward_strength, backward_max_idx = self._ols_distance(
                cumr, start, end, direction, forward=False
            )

            # -----------------------------
            # Remove weak or inconsistent regions
            # -----------------------------
            if forward_strength < self.amplitude_threshold  and backward_strength < self.amplitude_threshold :
                # Entire region too weak — reset to neutral
                self._apply_label(labels, start, end, 0.0)
            else:
                # Forward (start → forward_max_idx)
                if forward_strength >= self.amplitude_threshold :
                    self._apply_label(labels, start, forward_max_idx, direction)
                    self._apply_label(labels, forward_max_idx + 1, backward_max_idx - 1, 0.0)
                else:
                    self._apply_label(labels, start, backward_max_idx, 0.0)

                # Backward (backward_max_idx → end)
                if backward_strength >= self.amplitude_threshold :
                    self._apply_label(labels, backward_max_idx, end, direction)
                else:
                    neutral_start = max(backward_max_idx, forward_max_idx + 1)
                    self._apply_label(labels, neutral_start, end, 0.0)

            # Move to the next region
            pos = end + 1

    # =====================================================
    def _ols_distance(self, cumr: np.ndarray, start: int, end: int, direction: float, forward=True):
        """
        Estimate how far prices deviate from a local OLS trend within a region.

        Parameters
        ----------
        cumr : np.ndarray
            Cumulative return or price series (already log-transformed or normalized).
        start : int
            Starting index of the segment to evaluate.
        end : int
            Ending index of the segment to evaluate.
        direction : float
            +1 for upward trends, -1 for downward trends.
        forward : bool, optional
            If True, evaluate forward from start→end.
            If False, evaluate backward from end→start.

        Returns
        -------
        (max_distance, max_index)
            max_distance : float
                Maximum directional deviation in OLS slope.
            max_index : int
                Index where that maximum occurs.
        """
        max_distance = 0.0
        max_index = start if forward else end

        # Define iteration direction
        if forward:
            indices = range(start, end + 1)
        else:
            indices = range(end, start - 1, -1)

        # Track cumulative sums for OLS regression
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_xx = 0.0

        for i, idx in enumerate(indices):
            x = float(i)
            y = cumr[idx]

            # Incrementally update sums
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_xx += x * x

            # Need at least 2 points to fit OLS
            if i < 1:
                continue

            n = i + 1
            denom = (sum_xx - (sum_x ** 2) / n)
            if denom == 0:
                continue

            # Compute slope (beta) of OLS regression line
            beta = (sum_xy - (sum_x * sum_y) / n) / denom

            # Compute deviation along the trend direction
            distance = direction * beta * x if forward else -direction * beta * x

            # Track maximum deviation
            if distance > max_distance:
                max_distance = distance
                max_index = idx

        return max_distance, max_index

    # =====================================================
    @staticmethod
    def _apply_label(labels: np.ndarray, start_idx: int, end_idx: int, direction: float):
        """
        Assign a trend label to a contiguous segment of the labels array.

        Parameters:
        -----------
        labels : np.ndarray
            Array storing trend labels.
        start_idx : int
            Starting index of the segment.
        end_idx : int
            Ending index of the segment.
        direction : float
            Trend direction:
            +1 → uptrend
            -1 → downtrend
            0  → neutral / no trend

        Updates the labels array in-place.
        """
        if end_idx < start_idx:
            return  # nothing to label
        labels[start_idx:end_idx + 1] = direction

class TripleBarrierLabeler:

    def __init__(self, vol_window, upper_mult, lower_mult, max_holding_period, use_log_return=True):
        self.vol_window = int(vol_window)
        self.upper_mult = float(upper_mult)
        self.lower_mult = float(lower_mult)
        self.max_holding_period = int(max_holding_period)
        self.use_log_return = use_log_return

    def label(self, prices):
        prices = pd.Series(prices).astype(float).dropna().reset_index(drop=True)
        n = len(prices)

        if self.use_log_return:
            returns = np.log(prices).diff().fillna(0.0)
        else:
            returns = prices.pct_change().fillna(0.0)

        rolling_vol = returns.ewm(span=self.vol_window).std().bfill().values
        cum_returns = returns.cumsum().values
        labels = np.zeros(n)

        for i in range(n):

            sigma = rolling_vol[i]
            upper_barrier = self.upper_mult * sigma
            lower_barrier = self.lower_mult * sigma

            end = min(i + self.max_holding_period, n - 1)
            start_val = cum_returns[i]
            path = cum_returns[i:end + 1] - start_val

            up_hit = np.where(path >= upper_barrier)[0]
            down_hit = np.where(path <= -lower_barrier)[0]

            if len(up_hit) == 0 and len(down_hit) == 0:
                labels[i] = 0
            elif len(up_hit) == 0:
                labels[i] = -1
            elif len(down_hit) == 0:
                labels[i] = 1
            else:
                labels[i] = 1 if up_hit[0] < down_hit[0] else -1

        return pd.DataFrame({"price": prices, "label": labels})

class CombinedLabeler:
    """
    Consensus labeling: TBL verified by Trend.
    Only assigns a directional label (+1 or -1) if both labelers agree.
    """
    def __init__(self, amplitude_threshold, max_inactive_period, vol_window, upper_mult, lower_mult, max_holding):
        self.trend_labeler = Labeler(amplitude_threshold=amplitude_threshold, max_inactive_period=max_inactive_period)
        self.tbl_labeler = TripleBarrierLabeler(vol_window=vol_window, upper_mult=upper_mult, lower_mult=lower_mult, max_holding_period=max_holding)

    def label(self, prices):
        # 1. Run individual labelers
        res_trend = self.trend_labeler.label(prices)
        res_tbl = self.tbl_labeler.label(prices)
        
        # 2. Extract labels
        y_trend = res_trend['label'].values
        y_tbl = res_tbl['label'].values
        
        # 3. Consensus Logic: 
        # If both are same sign and non-zero, keep. Otherwise 0.
        combined = np.where(
            (np.sign(y_trend) == np.sign(y_tbl)) & (y_trend != 0),
            y_trend,
            0.0
        )
        
        return pd.DataFrame({"price": prices.values, "label": combined}, index=prices.index)

class TailSetLabeler:
    def __init__(self, std_window=100, ret_window=1, threshold=1.0, min_periods=None, eps=1e-12):
        self.std_window = std_window
        self.ret_window = ret_window
        self.threshold = threshold
        self.min_periods = min_periods if min_periods is not None else std_window
        self.eps = eps

    def label(self, prices):
        prices = pd.Series(prices).astype(float).dropna().reset_index(drop=True)

        if len(prices) < 2:
            return pd.DataFrame({"price": prices, "label": np.zeros(len(prices))})

        log_rets = np.log(prices).diff()

        if self.ret_window > 1:
            ret_series = log_rets.rolling(self.ret_window, min_periods=self.ret_window).sum()
        else:
            ret_series = log_rets

        rolling_std = ret_series.rolling(self.std_window, min_periods=self.min_periods).std()
        rolling_std = rolling_std.clip(lower=self.eps)

        upper = self.threshold * rolling_std
        lower = -self.threshold * rolling_std

        labels = np.zeros(len(prices))
        labels[ret_series > upper] = 1
        labels[ret_series < lower] = -1

        return pd.DataFrame(
            {
                "price": prices,
                "label": labels,
                "ret": ret_series,
                "rolling_std": rolling_std
            },
            index=prices.index,    
        )
