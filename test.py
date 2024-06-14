#!/usr/bin/python

import sys
import subprocess
import re

# in: data_in = list(dict{k:v})
# out: dict{k: list(v)}
def transpose(data_in):
    headers = set()
    for d in data_in:
        for h in d.keys():
            headers.add(h)
    headers = list(headers)

    data_out = {}
    for h in headers:
        data_out[h] = []

    for d in data_in:
        for h in headers:
            assert h in d
            data_out[h].append(d[h])

    return data_out

def print_aligned_table(headers, data, print_headers=True):

    rows = transpose(data)

    assert len(headers) > 0
    nrows = len(rows[headers[0]])
    for h in headers:
        assert len(rows[h]) == nrows

    def convert(x):
        if isinstance(x, int):
            x = f'{x:>6d}'
        if isinstance(x, float):
            s = f'{x:>3.2e}'
        elif isinstance(x, str):
            s = x
        else:
            print(x)
            assert False
        return s

    for h in headers:
        rows[h] = [convert(x) for x in rows[h]]

    col_width = [max(len(str(h)), max(len(x) for x in rows[h])) + 1 for h in headers]

    if print_headers:
        headers_str = [f'{str(h):>{c}}' for h, c in zip(headers, col_width)]
        print(','.join(headers_str))

    for row in range(nrows):
        row_str = [f'{rows[h][row]:>{c}}' for h, c in zip(headers, col_width)]
        print(','.join(row_str))




def dict_play(data):
    # New dictionary to store combined results
    combined_data = {}

    # Iterate through the original list
    for entry in data:
        # Create a key based on the common attributes
        key = (entry['precision'], entry['ag_or_rs'], entry['s'], entry['n'], entry['d'])

        # Initialize the entry in the combined_data dictionary if it doesn't exist
        if key not in combined_data:
            combined_data[key] = {
                'precision': entry['precision'],
                'ag_or_rs': entry['ag_or_rs'],
                's': entry['s'],
                'n': entry['n'],
                'd': entry['d']
            }

        # Update the time based on the value of 'ub_or_nvshmem'
        if entry['ub_or_nvshmem'].strip() == 'ub':
            combined_data[key]['time_ub'] = entry['time']
        elif entry['ub_or_nvshmem'].strip() == 'nvshmem':
            combined_data[key]['time_nvshmem'] = entry['time']

    # Convert the combined_data dictionary back to a list of dictionaries
    result = list(combined_data.values())

    # Calculate the speedup and add it to each dictionary
    for item in result:
        if 'time_ub' in item and 'time_nvshmem' in item:
            item['speedup_ub_over_nvshmem'] = item['time_nvshmem'] / item['time_ub']

    return result


def find_max_time(log_string):
    # Regular expression to find all GEMM Time values
    gemm_time_pattern = r'GEMM Time = ([\d\.]+) sec'

    # Find all matches in the string
    gemm_times = re.findall(gemm_time_pattern, log_string)

    # Convert the extracted times to floats
    gemm_times = [float(time) for time in gemm_times]

    # Return the maximum GEMM Time
    if gemm_times:
        return max(gemm_times)
    else:
        return None

def run(ub_or_nvshmem, ag_or_rs, precision, check, s=2048, n=64, d=128, eos=False, p2p=True):
    exe = ""

    nproc = 8 if eos else 4

    p2p_str = "--p2p " if p2p else ""

    if ub_or_nvshmem == "ub":
        exe = f"LD_LIBRARY_PATH=/workdir/libnvshmem_2.11.0-5+cuda12.0_x86_64/lib:/workdir/cublasmplite/install/lib/:$LD_LIBRARY_PATH torchrun --nproc-per-node={nproc} examples/pytorch/comm_gemm_overlap/test_gemm.py {p2p_str} "
    else:
        exe = f"NVTE_NVSHMEM=1 NVSHMEM_DISABLE_NCCL=1 NVSHMEM_REMOTE_TRANSPORT=none LD_LIBRARY_PATH=/workdir/libnvshmem_2.11.0-5+cuda12.0_x86_64/lib:/workdir/cublasmplite/install/lib/:/usr/local/cuda/compat/lib.real:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 torchrun --nproc-per-node={nproc} examples/pytorch/comm_gemm_overlap/test_gemm.py {p2p_str} "

    if ag_or_rs == "ag":
        exe += "--comm-type ag "
    elif ag_or_rs == "rs":
        exe += "--comm-type rs "
    else:
        print("Invalid comm_type")
        sys.exit(1)

    if precision == "fp8":
        exe += "--fp8 "
    elif precision == "bf16":
        pass
    else:
        print("Invalid precision")
        sys.exit(1)

    exe += f"-s {s} -n {n} -d {d} "

    if check == True:
        exe += "--check-numerics "

    return exe

if __name__ == "__main__":
    # set eos = True if eos passed as argument
    eos = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "eos":
            eos = True

    ub_or_nvshmem = ["ub", "nvshmem"]
    ag_or_rs = ["ag", "rs"]
    precisions = ["bf16"]
    if eos:
        precisions = ["fp8","bf16"]

    data = []

    check = True
    for p in precisions:
        for i in ub_or_nvshmem:
            istr = "userbuffers" if i == "ub" else "nvshmem    "
            for j in ag_or_rs:
                line = run(i, j, p, check, eos=eos, p2p=True)
                # print(line)
                print(f"{p} {istr} {j} {p} p2p=True ",end=" ")
                e = subprocess.run(line, shell=True, check=True, capture_output=True)
                result = u"\u2705" if "PASSED" in e.stdout.decode() else u"\u274c"
                print(f"{result} [check numerics]")

        if eos:
            line = run("ub", "rs", p, check, eos=eos, p2p=False)
            print(f"{p} userbuffers rs {p} p2p=False",end=" ")
            e = subprocess.run(line, shell=True, check=True, capture_output=True)
            result = u"\u2705" if "PASSED" in e.stdout.decode() else u"\u274c"
            print(f"{result} [check numerics]")

    ss = [2048, 4096, 8192]
    ns = [64, 96]
    ds = [128]

    if False: # TODO: just testing
        ub_or_nvshmem = ["ub","nvshmem"]
        ag_or_rs = ["rs","ag"]
        precisions = ["bf16"]
        ss = [2048]
        ns = [64]
        ds = [128]


    check = False
    for p in precisions:
        for j in ag_or_rs:
            for s in ss:
                for n in ns:
                    for d in ds:
                        for i in ub_or_nvshmem:
                            istr = "userbuffers" if i == "ub" else "nvshmem    "

                            line = run(i, j, p, check, s, n, d, eos=eos)
                            e = subprocess.run(line, shell=True, check=True, capture_output=True)
                            result = find_max_time(e.stdout.decode())
                            print(f"{p} {istr} {j} {s} {n} {d} {result*1000:8.3f} ms")
                            data.append({
                                "precision": p,
                                "ub_or_nvshmem": i,
                                "ag_or_rs": j,
                                "s": s,
                                "n": n,
                                "d": d,
                                "time": result
                            })


    # print("Before")
    # print(data)
    data = dict_play(data)

    # print(data)

    # [{'precision': 'bf16', 'ag_or_rs': 'rs', 's': 2048, 'n': 64, 'd': 128, 'time_ub': 0.011100159645080567, 'time_nvshmem': 0.009777152061462402, 'speedup_ub_over_nvshmem': 0.880811841818464}, {'precision': 'bf16', 'ag_or_rs': 'ag', 's': 2048, 'n': 64, 'd': 128, 'time_ub': 0.009209856033325196, 'time_nvshmem': 0.00628326416015625, 'speedup_ub_over_nvshmem': 0.6822326144318341}]
    cols = ["precision", "ag_or_rs", "s", "n", "d", "time_ub", "time_nvshmem", "speedup_ub_over_nvshmem"]
    print_aligned_table(cols, data)
#test s n d m prec
# test 2048, 64, 128, 3, bf16
# test 2048, 96, 128, 3, bf16
# test 4096, 64, 128, 3, bf16
# test 4096, 96, 128, 3, bf16
# test 8192, 64, 128, 3, bf16
# test 8192, 96, 128, 3, bf16