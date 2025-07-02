import os
import sys
import json
from argparse import ArgumentParser

# Add the parent directory to Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tomlkit
from tqdm import tqdm
import random

# Add wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

from DTransformer.data import KTData
from DTransformer.eval import Evaluator

DATA_DIR = "data"

# configure the main parser
parser = ArgumentParser()

# general options
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument("-bs", "--batch_size", help="batch size", default=8, type=int)
parser.add_argument(
    "-tbs", "--test_batch_size", help="test batch size", default=64, type=int
)

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

# training setup
parser.add_argument("-n", "--n_epochs", help="training epochs", type=int, default=100)
parser.add_argument(
    "-es",
    "--early_stop",
    help="early stop after N epochs of no improvements",
    type=int,
    default=10,
)
parser.add_argument(
    "-lr", "--learning_rate", help="learning rate", type=float, default=1e-3
)
parser.add_argument("-l2", help="L2 regularization", type=float, default=1e-5)
parser.add_argument(
    "-cl", "--cl_loss", help="use contrastive learning loss", action="store_true"
)
parser.add_argument(
    "--lambda", help="CL loss weight", type=float, default=0.1, dest="lambda_cl"
)
parser.add_argument("--window", help="prediction window", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", help="number of steps to accumulate gradients", type=int, default=1)
parser.add_argument("--max_seq_len", help="maximum sequence length to truncate", type=int, default=None)
parser.add_argument("--analyze_data", help="analyze sequence lengths before training", action="store_true")

# validation setup
parser.add_argument("--val_ratio", help="validation split ratio from training data", type=float, default=0.1)
parser.add_argument("--use_test_as_val", help="use test data as validation (not recommended)", action="store_true")

# snapshot setup
parser.add_argument("-o", "--output_dir", help="directory to save model files and logs")
parser.add_argument(
    "-f", "--from_file", help="resume training from existing model file", default=None
)

# wandb setup
parser.add_argument("--wandb", help="use wandb for logging", action="store_true")
parser.add_argument("--wandb_project", help="wandb project name", default="dtransformer")
parser.add_argument("--wandb_entity", help="wandb entity/username", default=None)
parser.add_argument("--wandb_run_name", help="wandb run name", default=None)

# reproducibility setup
parser.add_argument("--seed", help="random seed for reproducibility", type=int, default=42)


# Add this function before main()
def analyze_sequence_lengths(data_path):
    """Analyze sequence lengths in the dataset"""
    seq_lengths = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip().isdigit():
                seq_len = int(lines[i].strip())
                seq_lengths.append(seq_len)
                i += 3  # Skip question and answer lines
            else:
                i += 1
    
    if seq_lengths:
        print(f"Sequence length stats for {data_path}:")
        print(f"  Min: {min(seq_lengths)}")
        print(f"  Max: {max(seq_lengths)}")
        print(f"  Mean: {sum(seq_lengths)/len(seq_lengths):.2f}")
        print(f"  Sequences > 500: {sum(1 for x in seq_lengths if x > 500)}")
        print(f"  Sequences > 1000: {sum(1 for x in seq_lengths if x > 1000)}")
    return seq_lengths


def create_validation_split(train_path, val_ratio=0.1, seed=42):
    """Create validation split from training data"""
    random.seed(seed)
    
    # Read all training sequences
    sequences = []
    with open(train_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip().isdigit():
                # Found a sequence - read length, questions, answers
                seq_data = [lines[i], lines[i+1], lines[i+2]]
                sequences.append(seq_data)
                i += 3
            else:
                i += 1
    
    # Split sequences
    n_val = int(len(sequences) * val_ratio)
    random.shuffle(sequences)
    
    train_sequences = sequences[n_val:]
    val_sequences = sequences[:n_val]
    
    print(f"Created validation split: {len(train_sequences)} train, {len(val_sequences)} validation sequences")
    
    return train_sequences, val_sequences


def write_sequences_to_file(sequences, filepath):
    """Write sequences to file"""
    with open(filepath, 'w') as f:
        for seq in sequences:
            for line in seq:
                f.write(line)


# training logic
def main(args):
    # prepare dataset
    dataset = datasets[args.dataset]
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None
    
    # Analyze sequence lengths if requested
    if args.analyze_data:
        print("Analyzing training data...")
        analyze_sequence_lengths(os.path.join(DATA_DIR, dataset["train"]))
        print("\nAnalyzing test data...")
        analyze_sequence_lengths(os.path.join(DATA_DIR, dataset["test"]))
        print()
    
    # Override seq_len if max_seq_len is specified
    if args.max_seq_len is not None:
        seq_len = args.max_seq_len
        print(f"Limiting sequences to max length: {seq_len}")
    
    # Determine training and validation data files
    if "valid" in dataset:
        # Use existing validation split
        print("Using existing validation split from dataset configuration")
        train_file = os.path.join(DATA_DIR, dataset["train"])
        valid_file = os.path.join(DATA_DIR, dataset["valid"])
    elif args.use_test_as_val:
        # Use test as validation (current behavior - with warning)
        print("WARNING: Using test data as validation. This may lead to data leakage!")
        print("Consider using --val_ratio to create a proper validation split from training data.")
        train_file = os.path.join(DATA_DIR, dataset["train"])
        valid_file = os.path.join(DATA_DIR, dataset["test"])
    else:
        # Create validation split from training data
        print(f"Creating {args.val_ratio:.1%} validation split from training data...")
        train_sequences, val_sequences = create_validation_split(
            os.path.join(DATA_DIR, dataset["train"]), 
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        
        # Create temporary files (use output directory if available)
        temp_dir = args.output_dir if args.output_dir else DATA_DIR
        os.makedirs(temp_dir, exist_ok=True)
        
        train_file = os.path.join(temp_dir, f"temp_train_{args.dataset}.txt")
        valid_file = os.path.join(temp_dir, f"temp_val_{args.dataset}.txt")
        
        write_sequences_to_file(train_sequences, train_file)
        write_sequences_to_file(val_sequences, valid_file)
        
        print(f"Created temporary training file: {train_file}")
        print(f"Created temporary validation file: {valid_file}")
    
    train_data = KTData(
        train_file,
        dataset["inputs"],
        seq_len=seq_len,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_data = KTData(
        valid_file,
        dataset["inputs"],
        seq_len=seq_len,
        batch_size=args.test_batch_size,
    )

    # prepare logger and output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, "config.json")
        json.dump(vars(args), open(config_path, "w"), indent=2)
        
        # Create training log file
        log_path = os.path.join(args.output_dir, "training_log.csv")
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_auc,val_acc,val_precision,val_recall,val_f1,val_mae,val_rmse\n")
    else:
        log_path = None

    # Initialize wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            tags=[args.dataset, args.model if args.model else "DTransformer"]
        )
        # Log dataset info
        wandb.config.update({
            "n_questions": dataset["n_questions"],
            "n_pid": dataset["n_pid"],
            "inputs": dataset["inputs"],
            "effective_seq_len": seq_len
        })

    # prepare model and optimizer
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

    if args.from_file:
        model.load_state_dict(torch.load(args.from_file, map_location=lambda s, _: s))
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.l2
    )
    model.to(args.device)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # training
    best = {"auc": 0}
    best_epoch = 0
    for epoch in range(1, args.n_epochs + 1):
        print("start epoch", epoch)
        model.train()
        it = tqdm(iter(train_data))
        total_loss = 0.0
        total_pred_loss = 0.0
        total_cl_loss = 0.0
        total_cnt = 0
        accumulated_loss = 0.0
        
        for batch_idx, batch in enumerate(it):
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

                if args.cl_loss:
                    loss, pred_loss, cl_loss = model.get_cl_loss(q, s, pid)
                else:
                    loss = model.get_loss(q, s, pid)

                # Scale loss by accumulation steps
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                accumulated_loss += loss.item()
                total_cnt += 1

                # Only update weights every N steps
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    optim.zero_grad()
                    
                    total_loss += accumulated_loss
                    accumulated_loss = 0.0

                postfix = {"loss": total_loss / max(1, total_cnt // args.gradient_accumulation_steps)}
                if args.cl_loss:
                    total_pred_loss += pred_loss.item()
                    total_cl_loss += cl_loss.item()
                    postfix["pred_loss"] = total_pred_loss / total_cnt
                    postfix["cl_loss"] = total_cl_loss / total_cnt
                it.set_postfix(postfix)

        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / max(1, total_cnt // args.gradient_accumulation_steps)

        # validation
        model.eval()
        evaluator = Evaluator()

        with torch.no_grad():
            it = tqdm(iter(valid_data))
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
                    y, *_ = model.predict(q, s, pid)
                    evaluator.evaluate(s, torch.sigmoid(y))

        r = evaluator.report()
        print(r)

        # Log to wandb
        if args.wandb and WANDB_AVAILABLE:
            log_dict = {
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "val/auc": r["auc"],
                "val/acc": r["acc"],
                "val/mae": r["mae"],
                "val/rmse": r["rmse"]
            }
            
            # Only add precision, recall, f1 if they exist in the results
            if "precision" in r:
                log_dict["val/precision"] = r["precision"]
            if "recall" in r:
                log_dict["val/recall"] = r["recall"]
            if "f1" in r:
                log_dict["val/f1"] = r["f1"]
            
            if args.cl_loss:
                log_dict.update({
                    "train/pred_loss": total_pred_loss / total_cnt,
                    "train/cl_loss": total_cl_loss / total_cnt
                })
            
            wandb.log(log_dict)

        # Log to CSV file
        if log_path:
            with open(log_path, "a") as f:
                # Handle missing metrics with default values
                precision = r.get("precision", 0.0)
                recall = r.get("recall", 0.0)
                f1 = r.get("f1", 0.0)
                f.write(f"{epoch},{avg_train_loss:.6f},{r['auc']:.6f},{r['acc']:.6f},"
                       f"{precision:.6f},{recall:.6f},{f1:.6f},{r['mae']:.6f},{r['rmse']:.6f}\n")

        if r["auc"] > best["auc"]:
            best = r
            best_epoch = epoch
            
            # Log best metrics to wandb
            if args.wandb and WANDB_AVAILABLE:
                wandb.run.summary["best_epoch"] = best_epoch
                wandb.run.summary["best_auc"] = best["auc"]
                wandb.run.summary["best_acc"] = best["acc"]
                wandb.run.summary["best_mae"] = best["mae"]
                wandb.run.summary["best_rmse"] = best["rmse"]
                if "precision" in best:
                    wandb.run.summary["best_precision"] = best["precision"]
                if "recall" in best:
                    wandb.run.summary["best_recall"] = best["recall"]
                if "f1" in best:
                    wandb.run.summary["best_f1"] = best["f1"]

        if args.output_dir:
            model_path = os.path.join(
                args.output_dir, f"model-{epoch:03d}-{r['auc']:.4f}.pt"
            )
            print("saving snapshot to:", model_path)
            torch.save(model.state_dict(), model_path)
            
            # Log model artifact to wandb (optional)
            if args.wandb and WANDB_AVAILABLE and r["auc"] == best["auc"]:
                artifact = wandb.Artifact(f"model-{args.dataset}", type="model")
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)

        if args.early_stop > 0 and epoch - best_epoch > args.early_stop:
            print(f"did not improve for {args.early_stop} epochs, stop early")
            break

    # Finish wandb run
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()

    return best_epoch, best


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    best_epoch, best = main(args)
    print(args)
    print("best epoch:", best_epoch)
    print("best result", {k: f"{v:.4f}" for k, v in best.items()})
