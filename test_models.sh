#!/bin/bash

# Example script to test your trained DTransformer models on the eedi test set

# Test the DTransformer model from split1
# echo "Testing DTransformer split1 model..."
# python scripts/test.py \
#   --dataset moocRadar \
#   --model DTransformer \
#   --d_model 128 \
#   --n_layers 3 \
#   --n_heads 8 \
#   --n_know 32 \
#   --dropout 0.2 \
#   --lambda 0.1 \
#   --window 1 \
#   --max_seq_len 200 \
#   --device cuda \
#   --batch_size 64 \
#   --from_file /mnt/ceph_rbd/baseline_results/moocradar/dtrans/model-006-0.8850.pt \
#   -N 1

# Test the DTransformer model from split1
# echo "Testing DTransformer split2 model..."
# python scripts/test.py \
#   --dataset moocRadar \
#   --model DTransformer \
#   --d_model 128 \
#   --n_layers 3 \
#   --n_heads 8 \
#   --n_know 32 \
#   --dropout 0.2 \
#   --lambda 0.1 \
#   --window 1 \
#   --max_seq_len 200 \
#   --device cuda \
#   --batch_size 64 \
#   --from_file /mnt/ceph_rbd/baseline_results/moocradar/dtrans/model-005-0.8846.pt \
#   -N 1

# Test the DTransformer model from split1
# echo "Testing DTransformer split3 model..."
# python scripts/test.py \
#   --dataset moocRadar \
#   --model DTransformer \
#   --d_model 128 \
#   --n_layers 3 \
#   --n_heads 8 \
#   --n_know 32 \
#   --dropout 0.2 \
#   --lambda 0.1 \
#   --window 1 \
#   --max_seq_len 200 \
#   --device cuda \
#   --batch_size 64 \
#   --from_file /mnt/ceph_rbd/baseline_results/moocradar/dtrans/model-007-0.8858.pt\
#   -N 1



# You can also test baseline models if needed
echo "Testing AKT split1 model..."
python scripts/test.py \
  --dataset eedi_coldstart \
  --model AKT \
  --d_model 128 \
  --n_heads 8 \
  --dropout 0.2 \
  --device cuda \
  --batch_size 8 \
  --from_file /mnt/ceph_rbd/baseline_results/eedi/coldstart/akt/model-007-0.7141.pt \
  -N 1

# You can also test baseline models if needed
echo "Testing AKT split2 model..."
python scripts/test.py \
  --dataset eedi_coldstart \
  --model AKT \
  --d_model 128 \
  --n_heads 8 \
  --dropout 0.2 \
  --device cuda \
  --batch_size 8 \
  --from_file /mnt/ceph_rbd/baseline_results/eedi/coldstart/akt/model-006-0.7124.pt \
  -N 1

# You can also test baseline models if needed
echo "Testing AKT split3 model..."
python scripts/test.py \
  --dataset eedi_coldstart \
  --model AKT \
  --d_model 128 \
  --n_heads 8 \
  --dropout 0.2 \
  --device cuda \
  --batch_size 8 \
  --from_file /mnt/ceph_rbd/baseline_results/eedi/coldstart/akt/model-004-0.7123.pt \
  -N 1




echo "Testing DKT split1 model..."
python scripts/test.py \
  --dataset eedi_coldstart \
  --model DKT \
  --d_model 128 \
  --device cuda \
  --batch_size 8 \
  --from_file /mnt/ceph_rbd/baseline_results/eedi/coldstart/dkt/model-005-0.7067.pt \
  -N 1

echo "Testing DKT split2 model..."
python scripts/test.py \
  --dataset eedi_coldstart \
  --model DKT \
  --d_model 128 \
  --device cuda \
  --batch_size 8 \
  --from_file /mnt/ceph_rbd/baseline_results/eedi/coldstart/dkt/model-006-0.7049.pt \
  -N 1

echo "Testing DKT split3 model..."
python scripts/test.py \
  --dataset eedi_coldstart \
  --model DKT \
  --d_model 128 \
  --device cuda \
  --batch_size 8 \
  --from_file /mnt/ceph_rbd/baseline_results/eedi/coldstart/dkt/model-007-0.7114.pt \
  -N 1

#   echo "Testing DKT split2 model..."
# python scripts/test.py \
#   --dataset eedi \
#   --model DKT \
#   --d_model 128 \
#   --device cuda \
#   --batch_size 64 \
#   --from_file /mnt/ceph_rbd/baseline_results/dkt/split2/model-008-0.7052.pt \
#   -N 1

# echo "Testing DKT split3 model..."
# python scripts/test.py \
#   --dataset eedi \
#   --model DKT \
#   --d_model 128 \
#   --device cuda \
#   --batch_size 64 \
#   --from_file /mnt/ceph_rbd/baseline_results/dkt/split3/model-009-0.7104.pt \
#   -N 1

# echo "Testing DKT split4 model..."
# python scripts/test.py \
#   --dataset eedi \
#   --model DKT \
#   --d_model 128 \
#   --device cuda \
#   --batch_size 64 \
#   --from_file /mnt/ceph_rbd/baseline_results/dkt/split5/model-011-0.7141.pt \
#   -N 1

# echo "Testing DKT split5 model..."
# python scripts/test.py \
#   --dataset eedi \
#   --model DKT \
#   --d_model 128 \
#   --device cuda \
#   --batch_size 64 \
#   --from_file /mnt/ceph_rbd/baseline_results/dkt/split4/model-008-0.7143.pt \
#   -N 1
