from incremental_dbscan import IncrementalDBSCAN, adjusted_rand_index, evaluate_clustering
from sklearn.cluster  import DBSCAN, KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import pandas as pd

class BenchmarkPipeline:
    def __init__(self, data, num_batches =5, times=30):
        self.data = data
        self.num_batches = num_batches
        self.times = times
        self.raw_results = []
        self.batches = self.prepare_stream()

    def plot_runtime(self):
        df = pd.DataFrame(self.raw_results)
        time_cols = [c for c in df.columns if c.startswith("Time_")]
        plot_data = df.melt( id_vars=["Batch_Index"], value_vars=time_cols, var_name="Algorithm", value_name="Runtime")

        plot_data["Algorithm"] = plot_data["Algorithm"].str.replace("Time_", "")

        plt.figure(figsize=(12, 6))

        sns.boxplot(data=plot_data, x="Batch_Index", y="Runtime", hue="Algorithm")
        plt.xlabel("Lần cập nhật")
        plt.title(f"Thời gian chạy giữa các thuật toán trên từng batch")
        plt.ylabel("Thời gian (giây)")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.show()

    def plot_quality(self):
        df = pd.DataFrame(self.raw_results)
        metrics = { "Silhouette Score": "Sil_", "Davies-Bouldin Index": "Davies_", "Calinski-Harabasz Index": "Calinski_"}

        for metric_name, prefix in metrics.items():
            cols = [c for c in df.columns if c.startswith(prefix)]
            plot_data = df.melt(id_vars=["Batch_Index"], value_vars=cols, var_name="Algorithm", value_name="Score")

            plot_data["Algorithm"] = plot_data["Algorithm"].str.replace(prefix, "")

            plt.figure(figsize=(10, 6))
            sns.lineplot(
                          data=plot_data,   x="Batch_Index",    y="Score",
                          hue="Algorithm",  style="Algorithm",  markers=True,
                          dashes=True,      linewidth=2.5,      markersize=9
                        )

            title = f"{metric_name} qua các lần cập nhật"

            plt.title(title)
            plt.xlabel("Lần cập nhật")
            plt.ylabel(metric_name)
            plt.xticks(plot_data["Batch_Index"])
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


    def prepare_stream(self):
        N = len(self.data)
        b1_size = int(0.7 * N)  # Lấy 70%
        batches = [ self.data[0 : b1_size] ]

        N_rest = self.num_batches - 1
        rem_size = N - b1_size
        step = rem_size // N_rest

        start = b1_size
        for i in range(N_rest):
            if i == (N_rest - 1):
                end = N
            else:
                end = start + step

            batches.append( self.data[start : end] )
            start = end

        return batches

    def run(self):
        def measure_time(func, data, run_type):
            start_time = time.perf_counter()
            if run_type == "batch":
                label = func.fit_predict(data)
            elif run_type == "partial":
                func.partial_fit(data)
                label = func.labels_
            end_time = time.perf_counter()
            return end_time - start_time, label

        for i in range(self.times):
            static_method = {
                "DBSCAN": DBSCAN(eps=0.65, min_samples=12),
                "KMeans": KMeans(n_clusters=16, n_init=10, random_state=42),
            }
            inc_dbscan = IncrementalDBSCAN(eps=0.65, min_samples=12)
            minibatch = MiniBatchKMeans(n_clusters=16, n_init=10, random_state=42)

            accumulated_data = None

            for j, batch in enumerate(self.batches):
                step_log = {"Run_ID": i, "Batch_Index": j+1}
                accumulated_data = np.concatenate((accumulated_data, batch)) if accumulated_data is not None else batch

                batch_labels = {}

                for method_name, method in static_method.items():
                    t, lbl = measure_time(method, accumulated_data, "batch")
                    step_log[f"Time_{method_name}"] = t
                    batch_labels[method_name] = lbl

                t_mini, _ = measure_time(minibatch, batch, "partial")
                step_log["Time_MiniBatch"] = t_mini
                batch_labels["MiniBatch"] = minibatch.predict(accumulated_data)

                t_inc, lbl_inc = measure_time(inc_dbscan, batch, "partial")
                step_log["Time_IncDBSCAN"] = t_inc
                batch_labels["IncDBSCAN"] = lbl_inc

                ground_truth = batch_labels["DBSCAN"]

                for name, lbl in batch_labels.items():
                    metrics = evaluate_clustering(accumulated_data, lbl)
                    step_log[f"Sil_{name}"] = metrics["silhouette"]
                    step_log[f"Davies_{name}"] = metrics["davies_bouldin"]
                    step_log[f"Calinski_{name}"] = metrics["calinski_harabasz"]

                    if name in ["DBSCAN", "IncDBSCAN"]:
                        step_log[f"Noise_{name}"] = metrics["noise_ratio"]

                    if name != "DBSCAN":
                        step_log[f"ARI_{name}"] = adjusted_rand_index(ground_truth, lbl)

                self.raw_results.append(step_log)

    def report(self):
        if not self.raw_results:
            return pd.DataFrame(), pd.DataFrame()

        df = pd.DataFrame(self.raw_results)
        grouped = df.groupby("Batch_Index")

        time_cols = [c for c in df.columns if c.startswith("Time_")]
        time_agg = {c: ["mean", "std"] for c in time_cols}
        time_df = grouped.agg(time_agg)
        time_df.columns = [f"{col[0]}_{col[1]}" for col in time_df.columns]

        qual_cols = [c for c in df.columns if c.startswith(("Sil_", "Noise_", "ARI_", "Davies_", "Calinski_"))]
        qual_agg = {c: ["mean"] for c in qual_cols}
        qual_df = grouped.agg(qual_agg)
        qual_df.columns = [f"{col[0]}_{col[1]}" for col in qual_df.columns]

        return time_df.reset_index(), qual_df.reset_index()