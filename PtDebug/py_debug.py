import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="随便输入一下")
    parser.add_argument("--gpu_nums", type=int, default=50, help="the number of gpus")
    args = parser.parse_args()
    a = 1
    b = args.gpu_nums
    for i in range(100):
        a += 1
    print(a)
    print(b)
