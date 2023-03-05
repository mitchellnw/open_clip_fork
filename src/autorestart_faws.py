"""
This is helper script to launch sbatch jobs and to handle two issues
we encountered:

- freezing/hanging
- limited maximum job time (24 hours in the best case, can be 6 hours when total compute budget is over)

The script automatically relaunch the sbatch script when the job either freezes
or is stopped/canceled.

How to use it?

## Step 1

install the clize package using:

`pip install clize`

##  Step 2

Since the script needs to be running indefinitely, we launch a screen:

`screen -S screen_name`

## Step 3

`python autorestart.py "sbatch <your_script.sh> <your_arguments>" --output-file-template="sopenclip-{job_id}.out" --check-interval-secs=900 --verbose`

It is necessary to replace the `output-file-template` with the one you use since it is the output
file which is used to figure out if the job is freezing or not.
`check-interval-secs` determines the interval by which the job is checked.

## Step 4

CTRL + A then D to leave the screen and keep the script running indefinitely.

"""
import os
import re
import time
from subprocess import call, check_output
from clize import run

def main(cmd, *, output_file_template="openclip_{job_id}.out", check_interval_secs=900, termination_str="", verbose=True):
    cmd_check_job_in_queue = "squeue -j {job_id}"
    cmd_check_job_running = "squeue -j {job_id} -t R"
    while True:
        if verbose:
            print("Launch a new job")
            print(cmd)
        # launch job
        output = check_output(cmd, shell=True).decode()
        # get job id
        job_id = get_job_id(output)
        if verbose:
            print("Current job ID:", job_id)
        while True:
            # Infinite-loop, check each `check_interval_secs` whether job is present
            # in the queue, then, if present in the queue check if it is still running
            # and not frozen. The job is relaunched when it is no longuer running or
            # frozen. Then the same process is repeated.

            try:
                data = check_output(cmd_check_job_in_queue.format(job_id=job_id), shell=True).decode()
            except Exception as ex:
                # Exception after checking, which means that the job id no longer exists.
                # In this case, we relaunch directly except if termination string is found
                if verbose:
                    print(ex)
                if check_if_done(output_file_template.format(job_id=job_id), termination_str):
                    if verbose:
                        print("Termination string found, finishing")
                    return
                break
            # if job is not present in the queue, relaunch it directly, except if termination string is found
            if str(job_id) not in data:
                if check_if_done(output_file_template.format(job_id=job_id), termination_str):
                    if verbose:
                        print("Termination string found, finishing")
                    return
                break
            # Check first if job is specifically on a running state (to avoid the case where it is on pending state etc)
            data = check_output(cmd_check_job_running.format(job_id=job_id), shell=True).decode()
            if str(job_id) in data:
                # job on running state
                output_file = output_file_template.format(job_id=job_id)
                if verbose:
                    print("Check if the job is freezing...")
                # if job is on running state, check the output file
                output_data_prev = get_file_content(output_file)
                # wait few minutes
                time.sleep(check_interval_secs)
                # check again the output file
                output_data = get_file_content(output_file)
                # if the file did not change, then it is considered
                # to be frozen
                # (make sure there are is output before checking)
                if output_data and output_data_prev and output_data == output_data_prev:
                    if verbose:
                        print("Job frozen, stopping the job then restarting it")
                    call(f"scancel {job_id}", shell=True)
                    break
            else:
                # job not on running state, so it is present in the queue but in a different state
                # In this case, we wait for seconds, to check again if the job is still on the queue
                time.sleep(check_interval_secs)

def check_if_done(logfile, termination_str):
    return os.path.exists(logfile) and (termination_str != "") and re.search(termination_str, open(logfile).read())

def get_file_content(output_file):
    return open(output_file).read()

def get_job_id(s):
    return int(re.match("Submitted batch job ([0-9]+)", s).group(1))

if __name__ == "__main__":
    run(main)