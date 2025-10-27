import json
import os
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_accelerate_cpu_launch(tmp_path):
    script_path = tmp_path / "distributed_script.py"
    script_path.write_text(
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "from torch.utils.data import DataLoader, Dataset\n"
        "from toolkit.accelerator import get_accelerator\n"
        "\n"
        "class DummyDataset(Dataset):\n"
        "    def __init__(self):\n"
        "        self.data = list(range(4))\n"
        "\n"
        "    def __len__(self):\n"
        "        return len(self.data)\n"
        "\n"
        "    def __getitem__(self, idx):\n"
        "        return self.data[idx]\n"
        "\n"
        "def main():\n"
        "    output_dir = Path(sys.argv[1])\n"
        "    output_dir.mkdir(parents=True, exist_ok=True)\n"
        "    accelerator = get_accelerator()\n"
        "    loader = DataLoader(DummyDataset(), batch_size=1, shuffle=False, num_workers=0)\n"
        "    loader = accelerator.prepare_data_loader(loader)\n"
        "    seen = []\n"
        "    for batch in loader:\n"
        "        value = batch\n"
        "        if isinstance(value, (list, tuple)):\n"
        "            value = value[0]\n"
        "        if hasattr(value, 'item'):\n"
        "            value = value.item()\n"
        "        seen.append(int(value))\n"
        "    result = {'rank': accelerator.process_index, 'items': seen, 'world_size': accelerator.num_processes}\n"
        "    (output_dir / f'rank_{accelerator.process_index}.json').write_text(json.dumps(result))\n"
        "    accelerator.wait_for_everyone()\n"
        "    if accelerator.is_main_process:\n"
        "        combined = []\n"
        "        for idx in range(accelerator.num_processes):\n"
        "            combined.append(json.loads((output_dir / f'rank_{idx}.json').read_text()))\n"
        "        (output_dir / 'combined.json').write_text(json.dumps(combined))\n"
        "    accelerator.wait_for_everyone()\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    output_dir = tmp_path / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    port = str(_get_free_port())
    base_env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": port,
            "ACCELERATE_USE_CPU": "true",
            "ACCELERATE_NUM_PROCESSES": "2",
            "ACCELERATE_NUM_MACHINES": "1",
            "ACCELERATE_DISTRIBUTED_TYPE": "MULTI_CPU",
            "ACCELERATE_MIXED_PRECISION": "no",
        }
    )

    processes = []
    for rank in range(2):
        env = base_env.copy()
        env.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": "2",
                "LOCAL_WORLD_SIZE": "2",
                "ACCELERATE_PROCESS_INDEX": str(rank),
                "ACCELERATE_LOCAL_PROCESS_INDEX": str(rank),
            }
        )
        proc = subprocess.Popen(
            [sys.executable, str(script_path), str(output_dir)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(proc)

    for proc in processes:
        stdout, stderr = proc.communicate()
        assert proc.returncode == 0, f"Process failed: {stdout}\n{stderr}"

    combined_path = output_dir / "combined.json"
    assert combined_path.exists()
    combined = json.loads(combined_path.read_text())
    assert len(combined) == 2
    world_sizes = {entry['world_size'] for entry in combined}
    assert world_sizes == {2}
    all_items = sorted(item for entry in combined for item in entry['items'])
    assert all_items == [0, 1, 2, 3]
    for entry in combined:
        assert len(entry['items']) == 2
