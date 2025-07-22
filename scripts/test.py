import os
import sys
import json
from argparse import ArgumentParser

# Add the parent directory to Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tomlkit
from tqdm import tqdm

from DTransformer.data import KTData
from DTransformer.eval import Evaluator

DATA_DIR = "data"

# configure the main parser
parser = ArgumentParser()

# general options
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument("-bs", "--batch_size", help="batch size", default=64, type=int)

# data setup
datasets = tomlkit.load(open(os.path.join(DATA_DIR, "datasets.toml")))
parser.add_argument(
    "-d",
    "--dataset",
    help="choose from a dataset",
    choices=datasets.keys(),
    required=True,
)
parser.add_argument(
    "-p", "--with_pid", help="provide model with pid", action="store_true"
)

# model setup
parser.add_argument("-m", "--model", help="choose model")
parser.add_argument("--d_model", help="model hidden size", type=int, default=128)
parser.add_argument("--n_layers", help="number of layers", type=int, default=3)
parser.add_argument("--n_heads", help="number of heads", type=int, default=8)
parser.add_argument(
    "--n_know", help="dimension of knowledge parameter", type=int, default=32
)
parser.add_argument("--dropout", help="dropout rate", type=float, default=0.2)
parser.add_argument("--proj", help="projection layer before CL", action="store_true")
parser.add_argument(
    "--hard_neg", help="use hard negative samples in CL", action="store_true"
)
parser.add_argument(
    "--lambda", help="CL loss weight", type=float, default=0.1, dest="lambda_cl"
)
parser.add_argument("--window", help="prediction window", type=int, default=1)
parser.add_argument("--max_seq_len", help="maximum sequence length to truncate", type=int, default=None)

# test setup
parser.add_argument("-f", "--from_file", help="test existing model file", required=True)
parser.add_argument("-N", help="T+N prediction window size", type=int, default=1)


# testing logic
def main(args):
    # prepare datasets
    dataset = datasets[args.dataset]
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None
    
    # Override seq_len if max_seq_len is specified
    if args.max_seq_len is not None:
        seq_len = args.max_seq_len
        print(f"Limiting sequences to max length: {seq_len}")
    
    test_data = KTData(
        os.path.join(DATA_DIR, dataset["test"]),
        dataset["inputs"],
        seq_len=seq_len,
        batch_size=args.batch_size,
    )

    # prepare model
    if args.model == "DKT":
        from baselines.DKT import DKT

        model = DKT(dataset["n_questions"], args.d_model)
    elif args.model == "DKVMN":
        from baselines.DKVMN import DKVMN

        model = DKVMN(dataset["n_questions"], args.batch_size)
    elif args.model == "AKT":
        from baselines.AKT import AKT

        model = AKT(
            dataset["n_questions"],
            dataset["n_pid"],
            d_model=args.d_model,
            n_heads=args.n_heads,
            dropout=args.dropout,
        )
    else:
        from DTransformer.model import DTransformer

        model = DTransformer(
            dataset["n_questions"],
            dataset["n_pid"],
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_know=args.n_know,
            lambda_cl=args.lambda_cl,
            dropout=args.dropout,
            proj=args.proj,
            hard_neg=args.hard_neg,
            window=args.window,
        )

    model.load_state_dict(torch.load(args.from_file, map_location=lambda s, _: s))
    model.to(args.device)
    model.eval()

    # test
    evaluator = Evaluator()

    with torch.no_grad():
        it = tqdm(iter(test_data))
        for batch in it:
            if args.with_pid:
                q, s, pid = batch.get("q", "s", "pid")
            else:
                q, s = batch.get("q", "s")
                pid = None if seq_len is None else [None] * len(q)
            if seq_len is None:
                q, s, pid = [q], [s], [pid]
            for q, s, pid in zip(q, s, pid):
                q = q.to(args.device)
                s = s.to(args.device)
                if pid is not None:
                    pid = pid.to(args.device)
                y, *_ = model.predict(q, s, pid, args.N)
                evaluator.evaluate(s[:, (args.N - 1) :], torch.sigmoid(y))
            # it.set_postfix(evaluator.report())

    output_path = args.from_file + ".json"
    if os.path.exists(output_path):
        output = json.load(open(output_path))
    else:
        output = {"args": vars(args), "metrics": {}}

    output["metrics"][args.N] = evaluator.report()
    print(output["metrics"][args.N])

    json.dump(output, open(output_path, "w"), indent=2)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
