import time
import requests
import json
import os
import numpy as np
import scipy.stats as stats
from collections import Counter

class MaxPingTitanPredictor:
    def __init__(self, api_url, initial_bankroll=100000000, log_file="maxping_log.txt", history_json_file="strategy_history.json"):
        self.api_url = api_url
        self.bankroll = initial_bankroll
        self.log_file = log_file
        self.history_json_file = history_json_file
        
        # Cấu trúc dữ liệu Vector hóa theo yêu cầu phần I
        self.sids = np.array([], dtype=int)
        self.dice_history = []  
        self.total_scores = np.array([], dtype=int)
        self.binary_labels = np.array([], dtype=int)  # 1: Tài, 0: Xỉu
        self.results_str = []                         # "Tài" hoặc "Xỉu"
        self.delta_scores = np.array([], dtype=int)
        
        self.last_processed_sid = None
        self.current_champion = "MARKOV_TRANSITION"
        
        # 10 Strategy độc lập yêu cầu bởi Anhprodev
        self.strategy_names = [
            "FOLLOW_LAST", "REVERSE_LAST", "ALTERNATING_PATTERN", 
            "RUN_BREAK", "RUN_FOLLOW", "BIAS_MEAN_REVERSION", 
            "BIAS_MOMENTUM", "MARKOV_TRANSITION", "ANTI_RAW", "RANDOM_BALANCED_FALLBACK"
        ]
        
        # Trọng số động ban đầu cho 9 Lõi thuật toán truyền thống
        self.algo_weights = {
            'hmm_markov_v4': 1/9, 'vector_knn': 1/9, 'ema_reversion': 1/9, 
            'distance_spacing': 1/9, 'shannon_entropy': 1/9, 'autoregressive_prediction': 1/9, 
            'pearson_correlation': 1/9, 'runs_test': 1/9, 'delta_reversion': 1/9
        }
        self.algo_performance = {k: [] for k in self.algo_weights.keys()}

        # Tải lịch sử chiến lược để tính win_rate riêng
        self.execution_logs = self.load_strategy_history()
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'═'*75}\n🔥 MAXPING TITAN-V5 ANHPRODEV MASTER ENGINE STARTED AT {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'═'*75}\n")

    def write_log(self, message):
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def load_strategy_history(self):
        if os.path.exists(self.history_json_file):
            try:
                with open(self.history_json_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def save_strategy_history(self):
        try:
            with open(self.history_json_file, "w", encoding="utf-8") as f:
                json.dump(self.execution_logs[-300:], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def fetch_and_validate_data(self):
        """Phần I: Tiền xử lý toán học & Đồng bộ thời gian tuyến tính (200 phần tử)"""
        try:
            response = requests.get(self.api_url, timeout=5)
            if response.status_code != 200: return False
            
            data = response.json()
            if isinstance(data, dict) and "htr" in data:
                history_list = data["htr"]
            else:
                history_list = data
                
            if not history_list or not isinstance(history_list, list): return False

            # Đảm bảo trục thời gian tuyến tính tăng dần (Cũ -> Mới)
            if len(history_list) > 1 and history_list[0].get('sid', 0) > history_list[-1].get('sid', 0):
                history_list = list(reversed(history_list))

            self.sids = np.array([x['sid'] for x in history_list])
            self.dice_history = [(x['d1'], x['d2'], x['d3']) for x in history_list]
            self.total_scores = np.array([x['d1'] + x['d2'] + x['d3'] for x in history_list])
            self.binary_labels = np.where(self.total_scores >= 11, 1, 0)
            self.results_str = ["Tài" if score >= 11 else "Xỉu" for score in self.total_scores]
            
            if len(self.total_scores) > 1:
                self.delta_scores = np.diff(self.total_scores)
            else:
                self.delta_scores = np.array([0])
                
            return True
        except Exception:
            return False

    # ==========================================================
    # PHẦN II: ĐẶC TẢ LOGIC 9 LÕI THUẬT TOÁN ĐẶC QUYỀN
    # ==========================================================
    def _predict_markov_v4(self, order=4):
        n = len(self.binary_labels)
        if n < (order + 1): return None, 0.5
        target = self.binary_labels[-order:]
        match_tai = match_xiu = 0
        for i in range(n - order):
            if np.array_equal(self.binary_labels[i:i+order], target):
                if self.binary_labels[i+order] == 1: match_tai += 1
                else: match_xiu += 1
        total = match_tai + match_xiu
        if total == 0: return None, 0.5
        p_tai = match_tai / total
        return (1, p_tai) if p_tai >= 0.5 else (0, 1 - p_tai)

    def _predict_vector_knn(self, vector_size=6, k_neighbors=5):
        n = len(self.binary_labels)
        if n < (vector_size + 1): return None, 0.5
        target_vector = self.binary_labels[-vector_size:]
        distances, next_values = [], []
        for i in range(n - vector_size):
            dist = np.linalg.norm(target_vector - self.binary_labels[i:i+vector_size])
            distances.append(dist)
            next_values.append(self.binary_labels[i+vector_size])
        nearest_indices = np.argsort(distances)[:k_neighbors]
        counts = Counter([next_values[idx] for idx in nearest_indices])
        p_tai = counts.get(1, 0) / k_neighbors
        return (1, p_tai) if p_tai >= 0.5 else (0, 1 - p_tai)

    def _predict_ema_reversion(self, period=6):
        if len(self.total_scores) < period: return None, 0.5
        scores = self.total_scores[-period:]
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        ema_score = np.dot(scores, weights)
        if ema_score >= 12.8:
            return 0, min(0.90, 0.5 + (ema_score - 10.5) / 12)
        elif ema_score <= 8.2:
            return 1, min(0.90, 0.5 + (10.5 - ema_score) / 12)
        return None, 0.5

    def _predict_distance_spacing(self):
        n = len(self.binary_labels)
        if n < 20: return None, 0.5
        current_label = self.binary_labels[-1]
        streak_len = 0
        for val in reversed(self.binary_labels):
            if val == current_label: streak_len += 1
            else: break
        max_streak = temp_streak = 1
        for i in range(1, n):
            if self.binary_labels[i] == self.binary_labels[i-1]:
                temp_streak += 1
                max_streak = max(max_streak, temp_streak)
            else: temp_streak = 1
        if streak_len >= max_streak - 1:
            return (1 if current_label == 0 else 0), 0.80
        elif streak_len >= 3:
            return current_label, 0.65
        return None, 0.5

    def _predict_shannon_entropy(self, window_size=15):
        n = len(self.binary_labels)
        if n < window_size: return None, 0.5
        recent = self.binary_labels[-window_size:]
        p1 = np.sum(recent == 1) / window_size
        p0 = 1.0 - p1
        if p1 == 0 or p0 == 0: entropy = 0
        else: entropy = - (p1 * np.log2(p1) + p0 * np.log2(p0))
        if entropy < 0.72:
            return int(self.binary_labels[-1]), 0.72
        return None, 0.5

    def _predict_autoregressive(self, lags=4):
        n = len(self.total_scores)
        if n < 30: return None, 0.5
        recent_scores = self.total_scores[-lags:]
        mean_score = np.mean(self.total_scores[-30:])
        pred_score = mean_score + 0.3 * (recent_scores[-1] - mean_score) - 0.12 * (recent_scores[-2] - mean_score)
        if pred_score >= 11.2: return 1, min(0.82, 0.5 + (pred_score - 10.5)/10)
        elif pred_score <= 9.8: return 0, min(0.82, 0.5 + (10.5 - pred_score)/10)
        return None, 0.5

    def _predict_pearson_correlation(self, window_size=8):
        n = len(self.binary_labels)
        if n < (window_size * 2): return None, 0.5
        target = self.binary_labels[-window_size:]
        best_corr = 0
        pred_dir = None
        for i in range(n - (window_size * 2)):
            candidate = self.binary_labels[i : i + window_size]
            corr, _ = stats.pearsonr(target, candidate)
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                pred_dir = self.binary_labels[i + window_size]
        if abs(best_corr) > 0.48 and pred_dir is not None:
            direction = pred_dir if best_corr > 0 else (1 - pred_dir)
            return int(direction), 0.5 + (abs(best_corr) / 3)
        return None, 0.5

    def _predict_runs_test(self):
        n = len(self.binary_labels)
        if n < 20: return None, 0.5
        recent = self.binary_labels[-20:]
        n1 = np.sum(recent == 1)
        n0 = np.sum(recent == 0)
        if n1 == 0 or n0 == 0: return None, 0.5
        runs = 1 + sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
        mu_r = (2 * n1 * n0 / (n1 + n0)) + 1
        if runs < mu_r - 2: # Thiên về bệt
            return int(self.binary_labels[-1]), 0.68
        elif runs > mu_r + 2: # Thiên về đảo
            return int(1 - self.binary_labels[-1]), 0.68
        return None, 0.5

    def _predict_delta_reversion(self):
        if len(self.total_scores) < 2: return None, 0.5
        last_delta = self.total_scores[-1] - self.total_scores[-2]
        if last_delta >= 6:
            return 0, 0.70
        elif last_delta <= -6:
            return 1, 0.70
        return None, 0.5

    # ==========================================================
    # PHẦN III & IV: BỘ LỌC CHI-SQUARE & TRỌNG SỐ THÍCH ỨNG ĐỘNG
    # ==========================================================
    def _apply_chisquare_bias_filter(self):
        if len(self.binary_labels) < 100: return 1.0
        recent = self.binary_labels[-100:]
        o1 = np.sum(recent == 1)
        o0 = np.sum(recent == 0)
        chi2 = ((o1 - 50)**2 / 50) + ((o0 - 50)**2 / 50)
        return 1.35 if chi2 > 3.84 else 1.0

    def _optimize_weights(self, actual_label):
        if not self.algo_performance['hmm_markov_v4']: return
        adjusted = {}
        total_rate = 0
        for algo, perf in self.algo_performance.items():
            recent = perf[-12:] if len(perf) > 0 else []
            wr = np.mean(recent) if recent else 0.5
            adjusted[algo] = wr
            total_rate += wr
        if total_rate > 0:
            self.algo_weights = {k: v / total_rate for k, v in adjusted.items()}

    # ==========================================================
    # CÁC STRATEGY ĐỘC LẬP TRẢ VỀ OBJECT (CHUẨN ANHPRODEV)
    # ==========================================================
    def strategy_follow_last(self):
        pred = self.results_str[-1] if self.results_str else "Tài"
        return {"name": "FOLLOW_LAST", "prediction": pred, "local_confidence": 50, "reason": "Bám sát ván trước"}

    def strategy_reverse_last(self):
        last = self.results_str[-1] if self.results_str else "Tài"
        pred = "Xỉu" if last == "Tài" else "Tài"
        return {"name": "REVERSE_LAST", "prediction": pred, "local_confidence": 50, "reason": "Đảo ngược ván trước"}

    def strategy_alternating_pattern(self):
        if len(self.results_str) < 6: return self.strategy_reverse_last()
        recent = self.results_str[-6:]
        alts = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
        alt_rate = alts / (len(recent) - 1)
        last = self.results_str[-1]
        pred = ("Xỉu" if last == "Tài" else "Tài") if alt_rate >= 0.6 else self.strategy_reverse_last()["prediction"]
        return {"name": "ALTERNATING_PATTERN", "prediction": pred, "local_confidence": int(alt_rate * 60), "reason": f"Alternating rate: {alt_rate:.2f}"}

    def strategy_run_break(self):
        if len(self.results_str) < 3: return {"name": "RUN_BREAK", "prediction": "Tài", "local_confidence": 45, "reason": "Thiếu mẫu"}
        current_res = self.results_str[-1]
        run_len = 0
        for res in reversed(self.results_str):
            if res == current_res: run_len += 1
            else: break
        if run_len >= 3:
            pred = "Xỉu" if current_res == "Tài" else "Tài"
            return {"name": "RUN_BREAK", "prediction": pred, "local_confidence": 65, "reason": f"Bẻ cầu bệt {run_len}"}
        return {"name": "RUN_BREAK", "prediction": self.strategy_reverse_last()["prediction"], "local_confidence": 50, "reason": "Cầu ngắn"}

    def strategy_run_follow(self):
        if len(self.results_str) < 3: return {"name": "RUN_FOLLOW", "prediction": "Tài", "local_confidence": 45, "reason": "Thiếu mẫu"}
        current_res = self.results_str[-1]
        run_len = 0
        for res in reversed(self.results_str):
            if res == current_res: run_len += 1
            else: break
        if run_len >= 5:
            pred = "Xỉu" if current_res == "Tài" else "Tài"
            reason = f"Bệt dài {run_len}, chuyển bẻ"
        elif 2 <= run_len <= 3:
            pred = current_res
            reason = f"Theo nhịp bệt {run_len}"
        else:
            pred = self.strategy_reverse_last()["prediction"]
            reason = "Cầu ngắn"
        return {"name": "RUN_FOLLOW", "prediction": pred, "local_confidence": 55, "reason": reason}

    def strategy_bias_mean_reversion(self):
        if len(self.binary_labels) < 20: return self.strategy_reverse_last()
        p_tai_20 = np.mean(self.binary_labels[-20:])
        if p_tai_20 >= 0.60: pred, reason = "Xỉu", f"P(Tài)_20={p_tai_20:.2f} cao"
        elif p_tai_20 <= 0.40: pred, reason = "Tài", f"P(Tài)_20={p_tai_20:.2f} thấp"
        else: pred, reason = self.strategy_reverse_last()["prediction"], "Trung lập"
        return {"name": "BIAS_MEAN_REVERSION", "prediction": pred, "local_confidence": 58, "reason": reason}

    def strategy_bias_momentum(self):
        if len(self.binary_labels) < 12: return self.strategy_follow_last()
        p_tai_12 = np.mean(self.binary_labels[-12:])
        if p_tai_12 >= 0.58: pred, reason = "Tài", f"Momentum Tài {p_tai_12:.2f}"
        elif p_tai_12 <= 0.42: pred, reason = "Xỉu", f"Momentum Xỉu {p_tai_12:.2f}"
        else: pred, reason = self.strategy_follow_last()["prediction"], "Trung lập"
        return {"name": "BIAS_MOMENTUM", "prediction": pred, "local_confidence": 56, "reason": reason}

    def strategy_markov_transition(self):
        n = len(self.binary_labels)
        if n < 5: return self.strategy_reverse_last()
        TT, TX, XT, XX = 1.0, 1.0, 1.0, 1.0
        for i in range(max(0, n - 81), n - 1):
            if self.binary_labels[i] == 1 and self.binary_labels[i+1] == 1: TT += 1
            elif self.binary_labels[i] == 1 and self.binary_labels[i+1] == 0: TX += 1
            elif self.binary_labels[i] == 0 and self.binary_labels[i+1] == 1: XT += 1
            elif self.binary_labels[i] == 0 and self.binary_labels[i+1] == 0: XX += 1
        last_val = self.binary_labels[-1]
        p_tai = (TT / (TT + TX)) if last_val == 1 else (XT / (XT + XX))
        pred = "Tài" if p_tai >= 0.5 else "Xỉu"
        return {"name": "MARKOV_TRANSITION", "prediction": pred, "local_confidence": int(max(p_tai, 1-p_tai)*60), "reason": f"Markov P={p_tai:.2f}"}

    def strategy_anti_raw(self):
        if len(self.execution_logs) < 10: return self.strategy_reverse_last()
        recent = self.execution_logs[-20:]
        wrong = sum(1 for log in recent if log.get("final_prediction") != log.get("actual"))
        wrong_rate = (wrong / len(recent)) if recent else 0
        if wrong_rate >= 0.50 and self.execution_logs:
            last_pred = self.execution_logs[-1].get("final_prediction", "Tài")
            pred = "Xỉu" if last_pred == "Tài" else "Tài"
            reason = f"Sai sót cao {wrong_rate:.2f}"
        else:
            pred = self.strategy_reverse_last()["prediction"]
            reason = "Bình thường"
        return {"name": "ANTI_RAW", "prediction": pred, "local_confidence": 54, "reason": reason}

    def strategy_random_balanced_fallback(self):
        pred = "Tài" if len(self.results_str) % 2 == 0 else "Xỉu"
        return {"name": "RANDOM_BALANCED_FALLBACK", "prediction": pred, "local_confidence": 50, "reason": "Parity balanced"}

    def get_all_strategy_objects(self):
        objs = [
            self.strategy_follow_last(), self.strategy_reverse_last(), self.strategy_alternating_pattern(),
            self.strategy_run_break(), self.strategy_run_follow(), self.strategy_bias_mean_reversion(),
            self.strategy_bias_momentum(), self.strategy_markov_transition(), self.strategy_anti_raw(),
            self.strategy_random_balanced_fallback()
        ]
        return {obj["name"]: obj for obj in objs}

    def get_strategy_stats(self, strategy_name, window_size):
        if not self.execution_logs: return {"strategy": strategy_name, "total": 0, "win": 0, "loss": 0, "win_rate": 0.5}
        target = self.execution_logs[-window_size:]
        total, win = 0, 0
        for log in target:
            preds = log.get("strategy_predictions_map", {})
            actual = log.get("actual")
            if strategy_name in preds and actual:
                total += 1
                if preds[strategy_name] == actual: win += 1
        return {"strategy": strategy_name, "total": total, "win": win, "loss": total - win, "win_rate": (win/total) if total > 0 else 0.5}

    def get_strategy_loss_streak(self, strategy_name):
        streak = 0
        for log in reversed(self.execution_logs):
            preds = log.get("strategy_predictions_map", {})
            actual = log.get("actual")
            if strategy_name in preds and actual:
                if preds[strategy_name] != actual: streak += 1
                else: break
        return streak

    def select_champion_strategy(self):
        best_score = -999
        best_strategy = "MARKOV_TRANSITION"
        valid_found = False

        for name in self.strategy_names:
            s_st = self.get_strategy_stats(name, 20)
            m_st = self.get_strategy_stats(name, 50)
            l_st = self.get_strategy_stats(name, 100)

            score = (0.50 * s_st["win_rate"]) + (0.35 * m_st["win_rate"]) + (0.15 * l_st["win_rate"])
            if s_st["total"] < 10: score -= 0.05
            if m_st["total"] < 25: score -= 0.05
            if self.get_strategy_loss_streak(name) >= 3: score -= 0.07
            if s_st["win_rate"] < 0.45: score -= 0.10
            if m_st["win_rate"] < 0.48: score -= 0.05
            if s_st["win_rate"] >= 0.55: score += 0.05
            if m_st["win_rate"] >= 0.53: score += 0.05

            if (s_st["total"] >= 10 or m_st["total"] >= 25) and score >= 0.50:
                valid_found = True

            if score > best_score:
                best_score = score
                best_strategy = name

        if not valid_found:
            best_strategy = "MARKOV_TRANSITION" if self.get_strategy_stats("MARKOV_TRANSITION", 20)["total"] >= 5 else "REVERSE_LAST"

        # Champion Lock: Chống đổi liên tục
        current_loss = self.get_strategy_loss_streak(self.current_champion)
        current_s = self.get_strategy_stats(self.current_champion, 20)["win_rate"]
        if current_loss < 3 and current_s >= 0.48:
            if best_strategy != self.current_champion:
                new_sc = self.calc_score_val(best_strategy)
                old_sc = self.calc_score_val(self.current_champion)
                if new_sc - old_sc < 0.08: best_strategy = self.current_champion

        self.current_champion = best_strategy
        return best_strategy

    def calc_score_val(self, name):
        s = self.get_strategy_stats(name, 20)["win_rate"]
        m = self.get_strategy_stats(name, 50)["win_rate"]
        l = self.get_strategy_stats(name, 100)["win_rate"]
        return (0.50 * s) + (0.35 * m) + (0.15 * l)

    # ==========================================================
    # PHẦN V & VI: HÀM TỔNG HỢP KELLY & CHU TRÌNH THỰC THI THỜI GIAN THỰC
    # ==========================================================
    def run_pipeline(self):
        if len(self.sids) == 0: return
        latest_sid = self.sids[-1]
        if self.last_processed_sid == latest_sid: return
        
        actual_last_result = self.results_str[-1]
        self.last_processed_sid = latest_sid
        next_sid = latest_sid + 1

        # Thực thi 9 lõi thuật toán truyền thống để cập nhật trọng số
        preds, confs = {}, {}
        preds['hmm_markov_v4'], confs['hmm_markov_v4'] = self._predict_markov_v4()
        preds['vector_knn'], confs['vector_knn'] = self._predict_vector_knn()
        preds['ema_reversion'], confs['ema_reversion'] = self._predict_ema_reversion()
        preds['distance_spacing'], confs['distance_spacing'] = self._predict_distance_spacing()
        preds['shannon_entropy'], confs['shannon_entropy'] = self._predict_shannon_entropy()
        preds['autoregressive_prediction'], confs['autoregressive_prediction'] = self._predict_autoregressive()
        preds['pearson_correlation'], confs['pearson_correlation'] = self._predict_pearson_correlation()
        preds['runs_test'], confs['runs_test'] = self._predict_runs_test()
        preds['delta_reversion'], confs['delta_reversion'] = self._predict_delta_reversion()

        actual_binary = self.binary_labels[-1]
        for algo in self.algo_performance.keys():
            if preds[algo] is not None:
                self.algo_performance[algo].append(1 if preds[algo] == actual_binary else 0)
        self._optimize_weights(actual_binary)
        bias_multiplier = self._apply_chisquare_bias_filter()

        # Thực thi 10 Strategy độc lập của ANHPRODEV
        strat_objects = self.get_all_strategy_objects()
        strategy_preds_map = {name: obj["prediction"] for name, obj in strat_objects.items()}
        champion = self.select_champion_strategy()
        final_prediction = strategy_preds_map.get(champion, "Tài")

        # Thống kê hiệu suất hệ thống
        recent_logs = self.execution_logs[-50:]
        normal_win_rate = (sum(1 for log in recent_logs if log.get("final_prediction") == log.get("actual")) / len(recent_logs)) if recent_logs else 0.5
        recent_20 = self.execution_logs[-20:]
        win_rate_20 = (sum(1 for log in recent_20 if log.get("final_prediction") == log.get("actual")) / len(recent_20)) if recent_20 else 0.5
        win_rate_50 = normal_win_rate

        loss_streak = 0
        for log in reversed(self.execution_logs):
            if log.get("final_prediction") != log.get("actual"): loss_streak += 1
            else: break

        # 1. Recovery Mode & 3. AntiPhase Signal
        recovery_mode = (win_rate_50 < 0.45) or (win_rate_20 < 0.40) or (loss_streak >= 5)
        
        wrong_count_20 = sum(1 for log in recent_20 if log.get("final_prediction") != log.get("actual"))
        wrong_rate_20 = (wrong_count_20 / len(recent_20)) if recent_20 else 0.0
        
        anti_tai = anti_xiu = 0
        if wrong_rate_20 >= 0.55:
            for log in recent_20:
                if log.get("final_prediction") != log.get("actual"):
                    if log.get("final_prediction") == "Tài": anti_xiu += 1
                    else: anti_tai += 1

        # 4. Reverse Mode
        reverse_mode = False
        if win_rate_20 < 0.45 and len(recent_20) >= 20: reverse_mode = True
        reverse_win_count = sum(1 for log in recent_20 if ("Xỉu" if log.get("final_prediction")=="Tài" else "Tài") == log.get("actual"))
        reverse_win_rate = (reverse_win_count / len(recent_20)) if recent_20 else 0.5
        if reverse_win_rate - win_rate_20 >= 0.15: reverse_mode = True

        if reverse_mode or (recovery_mode and win_rate_20 < 0.40):
            final_prediction = "Xỉu" if final_prediction == "Tài" else "Tài"

        # 8. Confidence Mới & Kelly Quản Lý Vốn
        champ_20 = self.get_strategy_stats(champion, 20)
        champ_50 = self.get_strategy_stats(champion, 50)
        model_confidence = 50 + max(0, champ_20["win_rate"] - 0.50) * 100 + max(0, champ_50["win_rate"] - 0.50) * 50
        if champ_20["total"] < 10: model_confidence -= 5
        if self.get_strategy_loss_streak(champion) >= 3: model_confidence -= 8
        if champ_20["win_rate"] < 0.48: model_confidence -= 8
        if champ_50["win_rate"] < 0.50: model_confidence -= 5
        model_confidence = max(45, min(75, model_confidence))

        # 5. Display Confidence
        if recovery_mode: display_confidence = min(58, model_confidence)
        elif win_rate_20 < 0.45: display_confidence = min(58, model_confidence + 2)
        elif 55 <= model_confidence < 65: display_confidence = min(68, model_confidence + 5)
        elif model_confidence >= 65: display_confidence = min(78, model_confidence + 6)
        else: display_confidence = min(58, model_confidence + 3)

        # 9. Status Hệ Thống
        if model_confidence >= 65 and champ_20["win_rate"] >= 0.55 and not recovery_mode: status = "MẠNH"
        elif model_confidence >= 58: status = "TRUNG BÌNH"
        elif model_confidence >= 50: status = "YẾU"
        else: status = "NGUY HIỂM"
        if champ_20["total"] < 10 and status in ["MẠNH", "TRUNG BÌNH"]: status = "YẾU"

        # Đóng gói kết quả log theo chuẩn yêu cầu
        round_log = {
            "round": next_sid,
            "actual": actual_last_result,
            "final_prediction": final_prediction,
            "champion_strategy": champion,
            "recovery_mode": recovery_mode,
            "reverse_mode": reverse_mode,
            "anti_tai": anti_tai,
            "anti_xiu": anti_xiu,
            "normal_win_rate": normal_win_rate,
            "reverse_win_rate": reverse_win_rate,
            "strategy_predictions_map": strategy_preds_map,
            "strategy_predictions": {name: obj for name, obj in strat_objects.items()},
            "strategy_correct": {name: (pred == actual_last_result) for name, pred in strategy_preds_map.items()}
        }

        if self.execution_logs and self.execution_logs[-1].get("round") == latest_sid:
            self.execution_logs[-1]["actual"] = actual_last_result
            self.execution_logs[-1]["strategy_correct"] = {name: (pred == actual_last_result) for name, pred in self.execution_logs[-1]["strategy_predictions_map"].items()}

        self.execution_logs.append(round_log)
        self.save_strategy_history()

        # In báo cáo màn hình console
        log_block = []
        log_block.append("\n" + "═"*75)
        log_block.append(f"📡 TITAN-V5 ANHPRODEV | SID HOÀN TẤT: {latest_sid} | KẾT QUẢ: {self.total_scores[-1]} ({actual_last_result})")
        log_block.append(f"👑 CHAMPION STRATEGY: {champion} | WR-20: {champ_20['win_rate']*100:.1f}%")
        log_block.append(f"⚙️ TRẠNG THÁI: Recovery={recovery_mode} | Reverse={reverse_mode} | Status={status}")
        log_block.append(f"📊 CHỈ SỐ: Normal WR={normal_win_rate*100:.1f}% | Reverse WR={reverse_win_rate*100:.1f}% | Loss Streak={loss_streak}")
        log_block.append(f"🎯 PHÂN TÍCH CHO PHIÊN: {next_sid}")
        log_block.append(f"   [-] HƯỚNG DỰ ĐOÁN CUỐI  : [{final_prediction.upper()}]")
        log_block.append(f"   [-] ĐỘ TIN CẬY HIỂN THỊ  : {display_confidence:.1f}%")
        log_block.append(f"   [-] TÍN HIỆU ANTI-PHASE  : Anti-Tài={anti_tai} | Anti-Xỉu={anti_xiu}")
        log_block.append(f"   [-] CHẾ ĐỘ HOẠT ĐỘNG     : {'Recovery + Reverse' if (recovery_mode and reverse_mode) else 'Recovery Mode' if recovery_mode else 'Reverse Mode' if reverse_mode else 'Forced Prediction'}")
        log_block.append("═"*75)
        
        self.write_log("\n".join(log_block))

    def start_engine(self):
        self.write_log(f"🔥 KIẾN TRÚC MAXPING TITAN-V5 ANHPRODEV ĐÃ KHỞI CHẠY.\n📡 API Endpoint: {self.api_url}\n📝 Log Output: {self.log_file}")
        while True:
            if self.fetch_and_validate_data():
                self.run_pipeline()
            time.sleep(3.5)

if __name__ == "__main__":
    URL = "https://max789txth-058y.onrender.com/api/tx"
    quant_bot = MaxPingTitanPredictor(api_url=URL, initial_bankroll=100000000, log_file="maxping_log.txt")
    quant_bot.start_engine()