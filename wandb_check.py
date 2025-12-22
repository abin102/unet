import subprocess
import time
import psutil
import wandb
import threading

def get_gpu_stats():
    # uses nvidia-smi to return (util_pct, mem_used_mb, mem_total_mb) for GPU 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], encoding="utf8"
        ).strip()
        util, mem_used, mem_total = out.split(",")
        return int(util.strip()), int(mem_used.strip()), int(mem_total.strip())
    except Exception:
        return None

def background_sys_logger(run, interval=2.0, stop_event=None):
    while not (stop_event and stop_event.is_set()):
        gpu = get_gpu_stats()
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        data = {"system/cpu_percent": cpu, "system/ram_percent": ram}
        if gpu:
            util, used, total = gpu
            data.update({
                "system/gpu_util_percent": util,
                "system/gpu_mem_used_mb": used,
                "system/gpu_mem_total_mb": total
            })
        run.log(data)
        time.sleep(interval)

# example use
run = wandb.init(project="checking project", entity="abin24-cids")
stop = threading.Event()
t = threading.Thread(target=background_sys_logger, args=(run, 2.0, stop))
t.start()

# simulate training
for epoch in range(2, 20):
    # your training code...
    wandb.log({"acc": 0.5, "loss": 0.1})

# stop logger and finish
stop.set()
t.join()
run.finish()
