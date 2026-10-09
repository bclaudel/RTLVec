# ------------------------------------------------------------
# Train CdfgNN jointly on Masked-Node-Modeling + Edge-Prediction
# ------------------------------------------------------------
import json, torch, random
import math
from pathlib import Path
from uuid import uuid4
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from pyg_compat import DataLoader
from datetime import datetime

# ---------- project-specific helpers ----------
from args import parse_args
from utils import *
from models import CdfgNN
from dataset import load_datasets
from rtl_graph import NODE_TYPES
# ------------------------------------------------------------


# =====  Edge-prediction helper  =====
def generate_negative_edges(positive_edges, batch, edge_index, directed=True):
    """
    Vectorised negative-edge sampler (same as your EP file, trimmed).
    Returns negative_edges, positive_edges_trimmed with equal counts.
    """
    num_neg_needed = positive_edges.size(1)
    device = positive_edges.device
    if num_neg_needed == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device), positive_edges

    # Fast adjacency lookup on CPU
    adj = set(map(tuple, edge_index.t().cpu().tolist()))
    if not directed:
        adj |= set((v, u) for u, v in adj)

    # nodes per graph
    g_nodes = {}
    for n, g in enumerate(batch.cpu().tolist()):
        g_nodes.setdefault(g, []).append(n)

    occupied = {g: 0 for g in g_nodes}
    node_groups = batch.cpu().tolist()
    for u, v in adj:
        if u != v and node_groups[u] == node_groups[v]:
            occupied[node_groups[u]] += 1
    eligible = [g for g, nodes in g_nodes.items()
                if len(nodes) * (len(nodes) - 1) > occupied[g]]
    if not eligible:
        raise ValueError("No negative directed edges exist in this batch")
    neg = []
    for _ in range(num_neg_needed * 10 + 100):
        if len(neg) == num_neg_needed:
            break
        g = random.choice(eligible)
        u, v = random.sample(g_nodes[g], 2)
        if (u, v) not in adj:
            neg.append([u, v])
    # Bound rejection sampling even for nearly complete graphs. Sampling with
    # replacement matches the original objective and does not require N² storage.
    if not neg:
        nodes = g_nodes[eligible[0]]
        pair = next((u, v) for u in nodes for v in nodes if u != v and (u, v) not in adj)
        neg.append(list(pair))
    while len(neg) < num_neg_needed:
        neg.append(random.choice(neg))

    neg = torch.tensor(neg, device=device).t()          # [2, M]
    # trim positives if more than negatives
    if positive_edges.size(1) > neg.size(1):
        idx = torch.randperm(positive_edges.size(1), device=device)[:neg.size(1)]
        pos_trim = positive_edges[:, idx]
    else:
        pos_trim = positive_edges
    return neg, pos_trim


# =====  One epoch of *joint* training  =====
def train_joint_epoch(args, model, loader, optimizer,
                      loss_fn_mnm, loss_fn_ep, device):
    model.train()
    total_loss, tot_mnm_cor, tot_mnm_samp = 0, 0, 0
    tot_ep_cor,  tot_ep_samp  = 0, 0
    num_batches = 0

    for batch in loader:
        num_batches += 1
        batch = batch.to(device)
        optimizer.zero_grad()

        # ---------- shared graph data ----------
        x, edge_index = batch.x.clone(), batch.edge_index
        batch_id, pos_enc = batch.batch, batch.pos_enc       # [N], [N,k]

        if args.random_flip_posenc:
            num_graphs = batch_id.max().item() + 1
            pos_enc_flipped = pos_enc.clone()
            for g in range(num_graphs):
                mask = (batch_id == g)
                assert mask.sum() > 0, "No nodes in graph"
                sign_flip = 2 * torch.randint(0, 2, (1, pos_enc.shape[1]), device=pos_enc.device) - 1
                pos_enc_flipped[mask] = pos_enc[mask] * sign_flip.float()
            pos_enc = pos_enc_flipped

        # ---------- 1) Structure-Aware Masked-Node-Modeling (MNM) ----------
        target_type = batch.y[:, 0].long()
        mask = (stratified_mask(target_type, args.mask_ratio).to(device)
                if args.stratified_masking else
                (torch.rand(x.size(0), device=device) < args.mask_ratio))

        # forward *with* mask
        logits_mnm = model(x, edge_index, batch_id, pos_enc, mask)
        loss_mnm = (loss_fn_mnm(logits_mnm[mask], target_type[mask])
                    if mask.any() else logits_mnm.sum() * 0)

        preds_mnm = logits_mnm[mask].argmax(1)
        tot_mnm_cor  += (preds_mnm == target_type[mask]).sum().item()
        tot_mnm_samp += mask.sum().item()

        # ---------- 2) Edge-Prediction (EP) ----------
        n_edges   = edge_index.size(1)
        k_mask    = max(1, int(n_edges * args.mask_ratio))
        mask_idx  = torch.randperm(n_edges, device=device)[:k_mask]
        updated   = model.encode_nodes(x, edge_index, batch_id, pos_enc)

        pos_e = edge_index[:, mask_idx]
        neg_e, pos_trim = generate_negative_edges(pos_e, batch_id, edge_index)
        all_e  = torch.cat([pos_trim, neg_e], 1)
        labels = torch.cat([torch.ones(pos_trim.size(1), device=device),
                            torch.zeros(neg_e.size(1),  device=device)]).long()

        src, dst = updated[all_e[0]], updated[all_e[1]]
        edge_logits = model.edge_predictor(torch.cat([src, dst], 1))
        loss_ep = loss_fn_ep(edge_logits, labels)

        preds_ep = edge_logits.argmax(1)
        tot_ep_cor  += (preds_ep == labels).sum().item()
        tot_ep_samp += labels.size(0)

        # ---------- Joint loss & update ----------
        loss = loss_mnm + loss_ep
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return (total_loss / max(num_batches, 1),
            tot_mnm_cor / max(tot_mnm_samp, 1),
            tot_ep_cor  / max(tot_ep_samp,  1))


# =====  Validation helpers (task-specific, unchanged) =====
def validate_mnm(args, model, loader, loss_fn, device):
    model.eval(); tl, tc, ts = 0, 0, 0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x, ei, y, b, pe = batch.x, batch.edge_index, batch.y, batch.batch, batch.pos_enc
            tgt = y[:, 0].long()
            mask = (stratified_mask(tgt, args.mask_ratio).to(device)
                    if args.stratified_masking else
                    (torch.rand(x.size(0), device=device) < args.mask_ratio))
            if mask.sum() == 0: continue
            num_batches += 1
            logits = model(x, ei, b, pe, mask)
            tl += loss_fn(logits[mask], tgt[mask]).item()
            preds = logits[mask].argmax(1)
            tc += (preds == tgt[mask]).sum().item(); ts += mask.sum().item()
    return tl/max(num_batches,1), tc/max(ts,1)


def validate_ep(args, model, loader, loss_fn, device):
    model.eval(); tl, tc, ts = 0, 0, 0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            num_batches += 1
            batch = batch.to(device)
            x, ei, b, pe = batch.x, batch.edge_index, batch.batch, batch.pos_enc
            n_e   = ei.size(1); k_m = max(1, int(n_e*args.mask_ratio))
            m_idx = torch.randperm(n_e, device=device)[:k_m]
            upd   = model.encode_nodes(x, ei, b, pe)
            pos_e = ei[:, m_idx]
            neg_e, pos_trim = generate_negative_edges(pos_e, b, ei)
            all_e = torch.cat([pos_trim, neg_e], 1)
            lbls  = torch.cat([torch.ones(pos_trim.size(1), device=device),
                               torch.zeros(neg_e.size(1),  device=device)]).long()
            src, dst = upd[all_e[0]], upd[all_e[1]]
            logits = model.edge_predictor(torch.cat([src, dst], 1))
            tl += loss_fn(logits, lbls).item()
            tc += (logits.argmax(1) == lbls).sum().item()
            ts += lbls.size(0)
    return tl/max(num_batches,1), tc/max(ts,1)


# =====================  MAIN  =====================
def main():
    args = parse_args()
    if args.num_epochs < 1 or args.batch_size < 1 or not 0 < args.mask_ratio <= 1:
        raise ValueError("epochs/batch size must be positive and mask_ratio in (0, 1]")
    if args.k != 16 or args.jk_mode != "last" or args.num_gnn_layers < 2:
        raise ValueError("Stage 1 requires k=16, jk_mode=last and >=2 GNN layers")
    if args.num_classes is not None and args.num_classes != len(NODE_TYPES):
        raise ValueError("num_classes does not match the versioned node-type vocabulary")
    args.num_classes = len(NODE_TYPES)
    if args.dynamic_mask_ratio or args.transfer_learning or args.model_path:
        raise ValueError("Dynamic masking and transfer learning are not implemented")
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Data ---
    train_set, val_set, dataset_metadata = load_datasets(args.data_dir, mask_ratio=args.mask_ratio)
    if args.size_aware_batching:
        if args.max_nodes_per_batch < max(g.num_nodes for g in train_set + val_set):
            raise ValueError("max_nodes_per_batch must accommodate the largest individual graph")
        train_loader = DataLoader(train_set, batch_sampler=SizeAwareBatchSampler(train_set, args.max_nodes_per_batch, max_graphs=args.batch_size))
        val_loader = DataLoader(val_set, batch_sampler=SizeAwareBatchSampler(val_set, args.max_nodes_per_batch, shuffle=False, max_graphs=args.batch_size))
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # --- Model ---
    model_config = dict(gnn_type=args.gnn_type,
                   gnn_input_dim=args.num_classes + 32,
                   gnn_hidden_dim=args.gnn_hidden_dim,
                   gnn_output_dim=args.gnn_output_dim,
                   num_gnn_layers=args.num_gnn_layers,
                   gnn_num_heads=args.gnn_num_heads,
                   jk_mode=args.jk_mode,
                   gnn_dropout=args.gnn_dropout,
                   transformer_hidden_dim=args.transformer_hidden_dim,
                   transformer_feedforward_dim=args.transformer_feedforward_dim,
                   transformer_num_heads=args.transformer_num_heads,
                   num_transformer_layers=args.num_transformer_layers,
                   transformer_dropout=args.transformer_dropout,
                   num_classes=args.num_classes,
                   pos_enc_dim=args.k,
                   delete_node_width_embedding=args.delete_node_width_embedding,
                   edge_predictor_input_dim=2 * args.transformer_hidden_dim)
    model = CdfgNN(**model_config)
    if args.model_init_method == "xavier":
        model.apply(init_weights_xavier)
    model.to(device)

    # --- Losses ---
    cls_w = (compute_class_weights(train_set, args.num_classes).to(device)
             if args.use_class_weights else None)
    loss_mnm = (ClassBalancedFocalLoss(compute_class_counts(train_set, args.num_classes).to(device),
                                       beta=0.9999, gamma=2.0).to(device)
                if args.loss_type == "cb_focal"
                else CrossEntropyLoss(weight=cls_w, label_smoothing=args.label_smoothing).to(device))
    loss_ep = nn.CrossEntropyLoss()

    # --- Optimiser & Scheduler ---
    opt = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs,
                                                 eta_min=args.learning_rate*0.01)

    run_dir = Path(args.output_dir) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    (run_dir / "dataset.json").write_text(json.dumps(dataset_metadata, indent=2) + "\n")
    print(f"Run directory: {run_dir}")

    best_score = -float("inf")
    for epoch in range(args.num_epochs):
        tr_loss, tr_acc_mnm, tr_acc_ep = train_joint_epoch(
            args, model, train_loader, opt, loss_mnm, loss_ep, device)

        va_loss_mnm, va_acc_mnm = validate_mnm(args, model, val_loader, loss_mnm, device)
        va_loss_ep,  va_acc_ep  = validate_ep (args, model, val_loader, loss_ep,  device)

        print(f"Epoch {epoch+1:03d} │ "
              f"Loss {tr_loss:.4f} │ "
              f"MNM Acc {tr_acc_mnm:.4f}/{va_acc_mnm:.4f} │ "
              f"EP Acc  {tr_acc_ep:.4f}/{va_acc_ep:.4f}")
        metrics = dict(epoch=epoch+1, train_loss=tr_loss,
                       train_mnm_accuracy=tr_acc_mnm, train_ep_accuracy=tr_acc_ep,
                       val_mnm_loss=va_loss_mnm, val_ep_loss=va_loss_ep,
                       val_mnm_accuracy=va_acc_mnm, val_ep_accuracy=va_acc_ep,
                       learning_rate=opt.param_groups[0]["lr"])
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("Non-finite epoch metrics")
        with (run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        sched.step()

        # simple score: sum of task accuracies
        score = va_acc_mnm + va_acc_ep
        improved = score > best_score
        best_score = max(best_score, score)
        checkpoint = dict(schema_version=2, epoch=epoch+1, model_config=model_config,
                          model_state_dict=model.state_dict(),
                          optimizer_state_dict=opt.state_dict(),
                          scheduler_state_dict=sched.state_dict(),
                          dataset_fingerprint=dataset_metadata["dataset_fingerprint"],
                          preprocessing_fingerprint=dataset_metadata["preprocessing_fingerprint"],
                          preprocessing=dataset_metadata["preprocessing"],
                          args=vars(args), metrics=metrics, best_score=best_score)
        torch.save(checkpoint, run_dir / "last.pt")
        if improved:
            torch.save(checkpoint, run_dir / "best.pt")
            print("  ↳ Best model saved.")


if __name__ == "__main__":
    main()
