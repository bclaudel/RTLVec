# 🚀 Getting Started
### Environment Requirement 🌍
StructRTL has been implemented and tested on PyTorch 2.7.1 with python 3.10.

We recommend you first use ```conda``` to create virtual environment, and install ```pytorch``` following [official installation instructions](https://pytorch.org/). For example:
```
conda create -n structrtl python=3.10
conda activate structrtl
pip install torch
```
Then, you can install ```torch_geometric``` with
```
pip install torch_geometric
```

After that, you can train and evaluate StructRTL using the script:
```
python train.py \
--gnn_type gin \
--gnn_hidden_dim 256 \
--gnn_output_dim 256 \
--num_gnn_layers 8 \
--gnn_dropout 0.2 \
--transformer_num_heads 4 \
--transformer_hidden_dim 512 \
--transformer_feedforward_dim 2048 \
--num_transformer_layers 8 \
--transformer_dropout 0.2 \
--model_init_method xavier \
--batch_size 16 \
--num_epochs 2000 \
--k 16 \
--loss_type cb_focal \
--delete_node_width_embedding \
--random_flip_posenc \
--mask_ratio 0.2 \
--stratified_masking \
--learning_rate 2e-5 \
--weight_decay 1e-4
```

### Special Notes 🤯
The training and evaluation accuracies for the edge prediction task increase quickly and converge early in the training process. However, the training and evaluation accuracies for the structure-aware masked node modeling task take longer to improve, typically showing significant progress around 500-600 epochs.