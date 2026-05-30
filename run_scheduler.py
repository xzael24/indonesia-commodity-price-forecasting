# run_scheduler.py
"""
Agriforecast Background Retraining Scheduler.
Runs continuously in the background and triggers run_pipeline.py daily at exactly 23:00.
Save logs to models/scheduler.log.
"""

import os
import sys
import time
import subprocess
from datetime import datetime

# Setup directories
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
log_path = os.path.join(MODEL_DIR, "scheduler.log")

def write_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    print(log_line.strip())
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_line)

write_log("=== Agriforecast Daily Retraining Scheduler Started ===")
write_log("Automatic daily retraining is scheduled for 23:00 (11:00 PM).")

try:
    while True:
        now = datetime.now()
        
        # Trigger daily retraining at 23:00
        if now.hour == 23 and now.minute == 0:
            write_log("Time is 23:00. Initiating automated daily model retraining...")
            
            try:
                # Prepare env with UTF-8 encoding
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                
                # Execute training pipeline
                process = subprocess.Popen(
                    [sys.executable, "run_pipeline.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env
                )
                
                # Stream outputs directly to log file
                with open(os.path.join(MODEL_DIR, "pipeline.log"), "w", encoding="utf-8") as pipeline_log:
                    pipeline_log.write(f"► Automated Retraining Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    for line in process.stdout:
                        pipeline_log.write(line)
                        pipeline_log.flush()
                
                process.wait()
                
                if process.returncode == 0:
                    write_log("✓ Automated daily retraining completed successfully!")
                else:
                    write_log(f"✗ Automated daily retraining failed with exit code: {process.returncode}")
                    
            except Exception as e:
                write_log(f"✗ Error during automated daily retraining execution: {e}")
                
            # Sleep for 65 seconds to prevent multiple triggerings during the 23:00 minute
            time.sleep(65)
            
        # Sleep for 30 seconds before checking the clock again
        time.sleep(30)

except KeyboardInterrupt:
    write_log("=== Scheduler Stopped by User ===")
