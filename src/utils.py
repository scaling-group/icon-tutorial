from functools import wraps, partial
import time
import matplotlib.pyplot as plt
import io
import os
import json
import pytz
from datetime import datetime, timedelta
import re
import numpy as np
import torch.optim as optim
import random
import torch
from PIL import Image
import wandb
import matplotlib
import logging


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def strip_ansi_codes(s):
    return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', s)


def get_git_hash():
  import subprocess
  return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def load_json(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

def print_dot(i, freq = 100, marker = "."):
  if i % freq == 0:
    print(i, end = "", flush = True)
  print(marker, end = "", flush=True)
  if (i+1) % freq == 0:
    print("", flush=True)
    

def timeit_full(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds', flush = True)
        return result
    return timeit_wrapper

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds', flush = True)
        return result
    return timeit_wrapper

def get_days_hours_mins_seconds(time_consumed_in_seconds):
    time_consumed = time_consumed_in_seconds
    days_consumed = int(time_consumed // (24 * 3600))
    time_consumed %= (24 * 3600)
    hours_consumed = int(time_consumed // 3600)
    time_consumed %= 3600
    minutes_consumed = int(time_consumed // 60)
    seconds_consumed = int(time_consumed % 60)
    return days_consumed, hours_consumed, minutes_consumed, seconds_consumed

class TicToc:
    def __init__(self):
        self.start_time = {}
        self.end_time = {}

    def get_time_stamp(self):
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return time_stamp

    def tic(self, name):
        self.start_time[name] = time.perf_counter()

    def toc(self, name):
        self.end_time[name] = time.perf_counter()
        total_time = self.end_time[name] - self.start_time[name]
        print(f'{name} Took {total_time:.4f} seconds', flush=True)

    def get_time(self, name):
        return self.end_time[name] - self.start_time[name]

    def estimate_time(self, name, ratio, samples_processed=None):
        print('==========================Time Estimation Starts==========================')
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        print("Current time:", current_time_str)
        self.end_time[name] = time.perf_counter()
        time_consumed = self.end_time[name] - self.start_time[name]
        days_consumed, hours_consumed, minutes_consumed, seconds_consumed = get_days_hours_mins_seconds(time_consumed)
        print(f"Time consumed: {days_consumed}-{hours_consumed:02d}:{minutes_consumed:02d}:{seconds_consumed:02d}")
        if samples_processed is not None:
            samples_processed_per_second = samples_processed / time_consumed
            print(f"Samples processed per second: {samples_processed_per_second:.2f}")
        time_remaining = time_consumed * (1 - ratio) / ratio
        days_remaining, hours_remaining, minutes_remaining, seconds_remaining = get_days_hours_mins_seconds(time_remaining)
        print(f"Estimated remaining time: {days_remaining}-{hours_remaining:02d}:{minutes_remaining:02d}:{seconds_remaining:02d}")
        time_total = time_consumed / ratio
        days_total, hours_total, minutes_total, seconds_total = get_days_hours_mins_seconds(time_total)
        print(f"Estimated total time: {days_total}-{hours_total:02d}:{minutes_total:02d}:{seconds_total:02d}")
        finish_time = current_time + timedelta(seconds=time_remaining)
        finish_time_str = finish_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Estimated finishing time:", finish_time_str)
        print('==========================Time Estimation Ends==========================', flush=True)


timer = TicToc()


class AppLogger:
    def __init__(self, name, level=logging.INFO):
        # Initialize the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Record the start time of the logger
        self.start_time = time.time()

        # Create a stream handler with a custom format
        handler = logging.StreamHandler()
        formatter = self.CustomFormatter(self.start_time, fmt='%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(handler)
        self.logger.propagate = False

    class CustomFormatter(logging.Formatter):
        def __init__(self, start_time, fmt):
            super().__init__(fmt)
            self.start_time = start_time

        def format(self, record):
            # Calculate the elapsed time in seconds since the logger was created
            elapsed_seconds = int(record.created - self.start_time)

            # Convert elapsed time to days, hours, minutes, and seconds
            days, remainder = divmod(elapsed_seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)

            # Format the relative time as [D days, HH:MM:SS]
            relative_time = f"[{days}d {hours:02}:{minutes:02}:{seconds:02}s]"

            # Customize the message to include the relative time
            record.msg = f"{relative_time} {record.msg}"
            return super().format(record)

    def get_logger(self):
        return self.logger

logger = AppLogger("logger").get_logger()

if __name__ == "__main__":

  logger.info("Logger initialized and ready to use.")
  time.sleep(1)
  logger.info("Another log entry.")
  logger.warning("Another log entry.")
