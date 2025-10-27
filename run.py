import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
from dotenv import load_dotenv


def _preparse_accelerator_config(argv):
    config_value = None
    for index, arg in enumerate(argv):
        if arg.startswith("--accelerator-config="):
            config_value = arg.split("=", 1)[1]
            break
        if arg == "--accelerator-config" and index + 1 < len(argv):
            config_value = argv[index + 1]
            break
    if config_value and "AIT_ACCELERATOR_CONFIG" not in os.environ:
        os.environ["AIT_ACCELERATOR_CONFIG"] = config_value


_preparse_accelerator_config(sys.argv[1:])

# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)
import argparse
import subprocess
from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file

accelerator = None


def print_end_message(jobs_completed, jobs_failed):
    global accelerator
    if accelerator is None or not accelerator.is_main_process:
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    
    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )
    parser.add_argument(
        '--accelerate',
        action='store_true',
        help='Launch this script through accelerate for distributed execution.'
    )
    parser.add_argument(
        '--accelerate-num-processes',
        type=int,
        default=None,
        help='Number of processes to launch when using --accelerate.'
    )
    parser.add_argument(
        '--accelerate-num-machines',
        type=int,
        default=None,
        help='Number of machines to use when using --accelerate.'
    )
    parser.add_argument(
        '--accelerate-machine-rank',
        type=int,
        default=None,
        help='Rank of the current machine when using --accelerate.'
    )
    parser.add_argument(
        '--accelerate-main-process-ip',
        type=str,
        default=None,
        help='Main process IP address for multi-machine accelerate runs.'
    )
    parser.add_argument(
        '--accelerate-main-process-port',
        type=int,
        default=None,
        help='Main process port for multi-machine accelerate runs.'
    )
    parser.add_argument(
        '--accelerate-mixed-precision',
        type=str,
        default=None,
        help='Override accelerate mixed precision mode when launching.'
    )
    parser.add_argument(
        '--accelerate-cpu',
        action='store_true',
        help='Force accelerate to run on CPU when launching.'
    )
    parser.add_argument(
        '--accelerator-config',
        type=str,
        default=os.environ.get('AIT_ACCELERATOR_CONFIG'),
        help='Path or JSON string with kwargs for Accelerator().'
    )

    args = parser.parse_args()

    if args.accelerator_config:
        os.environ["AIT_ACCELERATOR_CONFIG"] = args.accelerator_config

    if args.accelerate and os.environ.get("ACCELERATE_LAUNCHED_PROCESS") != "1":
        launch_cmd = ["accelerate", "launch"]
        if args.accelerate_num_processes:
            launch_cmd += ["--num_processes", str(args.accelerate_num_processes)]
        if args.accelerate_num_machines:
            launch_cmd += ["--num_machines", str(args.accelerate_num_machines)]
        if args.accelerate_machine_rank is not None:
            launch_cmd += ["--machine_rank", str(args.accelerate_machine_rank)]
        if args.accelerate_main_process_ip:
            launch_cmd += ["--main_process_ip", args.accelerate_main_process_ip]
        if args.accelerate_main_process_port is not None:
            launch_cmd += ["--main_process_port", str(args.accelerate_main_process_port)]
        if args.accelerate_mixed_precision:
            launch_cmd += ["--mixed_precision", args.accelerate_mixed_precision]
        if args.accelerate_cpu:
            launch_cmd.append("--cpu")

        filtered_args = []
        skip_next = False
        flags_with_values = {
            "--accelerate-num-processes",
            "--accelerate-num-machines",
            "--accelerate-machine-rank",
            "--accelerate-main-process-ip",
            "--accelerate-main-process-port",
            "--accelerate-mixed-precision",
            "--accelerator-config",
        }
        flags_no_values = {"--accelerate", "--accelerate-cpu"}

        for arg in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if arg in flags_no_values:
                continue
            if arg in flags_with_values:
                skip_next = True
                continue
            if any(arg.startswith(f"{flag}=") for flag in flags_with_values):
                continue
            filtered_args.append(arg)

        command = launch_cmd + [sys.executable, sys.argv[0]] + filtered_args
        subprocess.check_call(command)
        return
    
    if args.log is not None:
        setup_log_to_file(args.log)

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    global accelerator
    accelerator = get_accelerator()

    jobs_completed = 0
    jobs_failed = 0

    if accelerator.is_main_process:
        print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        try:
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
        except KeyboardInterrupt as e:
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e


if __name__ == '__main__':
    main()
