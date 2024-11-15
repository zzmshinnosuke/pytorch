import torch
import os
from datetime import datetime
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional
import sys

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help":"Path to pretrained model"}
    )

def main() -> int:
    parser = HfArgumentParser((ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args = parser.parse_args_into_dataclasses()
    
    print(model_args[0])

    local_rank = int(os.environ["LOCAL_RANK"])
    word_size = int(os.environ["WORLD_SIZE"])
    cur_datetime = datetime.now()

    print("*" * 80)
    value = torch.cuda.device_count()
    print(cur_datetime, word_size, local_rank, value)
    return value

if __name__ == '__main__':
    main()