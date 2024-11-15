import torch
import os
from datetime import datetime
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional
import sys
from transformers import TrainingArguments

import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


def gen_data():
    x = torch.randn(10, 10).to('cuda')
    return x


def main() -> int:

    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, train_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, train_args = parser.parse_args_into_dataclasses()

    print(model_args)
    print(train_args)

    gen_data()

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    cur_dateime = datetime.now()
    print("*" * 80)
    value = torch.cuda.device_count()
    print(
        f"----> cur_datetime: {cur_dateime},world size: {world_size}, local_rank: {local_rank}, gpu count: {value}")
    return value


if __name__ == '__main__':
    main()