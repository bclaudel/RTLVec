# ------------------------------------------------------------
# Train CdfgNN jointly on Masked-Node-Modeling + Edge-Prediction
# ------------------------------------------------------------
import os, json, torch, random
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader
from datetime import datetime

# ---------- project-specific helpers ----------
from args import parse_args
from utils import *
from models import CdfgNN
# ------------------------------------------------------------


# =====  Edge-prediction helper  =====
def generate_negative_edges(positive_edges, batch, edge_index, directed=True):
    """
    Vectorised negative-edge sampler (same as your EP file, trimmed).
    Returns negative_edges, positive_edges_trimmed with equal counts.
    """
    num_neg_needed = positive_edges.size(1)
    device = positive_edges.device

    # Fast adjacency lookup on CPU
    adj = set(map(tuple, edge_index.t().cpu().tolist()))
    if not directed:
        adj |= set((v, u) for u, v in adj)

    # nodes per graph
    g_nodes = {}
    for n, g in enumerate(batch.cpu().tolist()):
        g_nodes.setdefault(g, []).append(n)

    neg = []
    while len(neg) < num_neg_needed:
        g = random.choice(list(g_nodes))
        u, v = random.sample(g_nodes[g], 2)
        if (u, v) not in adj:
            neg.append([u, v])

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

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # ---------- shared graph data ----------
        x, edge_index = batch.x.clone(), batch.edge_index
        batch_id, pos_enc = batch.batch, batch.pos_enc       # [N], [N,k]
        x_trim = x[:, :-1]                                   # EP uses trimmed feats

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
        loss_mnm = loss_fn_mnm(logits_mnm[mask], target_type[mask])

        preds_mnm = logits_mnm[mask].argmax(1)
        tot_mnm_cor  += (preds_mnm == target_type[mask]).sum().item()
        tot_mnm_samp += mask.sum().item()

        # ---------- 2) Edge-Prediction (EP) ----------
        node_emb = model.gnn(x_trim, edge_index, batch_id)
        n_edges   = edge_index.size(1)
        k_mask    = int(n_edges * args.mask_ratio)
        mask_idx  = torch.randperm(n_edges, device=device)[:k_mask]
        updated   = model.transformer(node_emb, batch_id, pos_enc)

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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return (total_loss / len(loader),
            tot_mnm_cor / max(tot_mnm_samp, 1),
            tot_ep_cor  / max(tot_ep_samp,  1))


# =====  Validation helpers (task-specific, unchanged) =====
def validate_mnm(args, model, loader, loss_fn, device):
    model.eval(); tl, tc, ts = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x, ei, y, b, pe = batch.x, batch.edge_index, batch.y, batch.batch, batch.pos_enc
            tgt = y[:, 0].long()
            mask = (stratified_mask(tgt, args.mask_ratio).to(device)
                    if args.stratified_masking else
                    (torch.rand(x.size(0), device=device) < args.mask_ratio))
            if mask.sum() == 0: continue
            logits = model(x, ei, b, pe, mask)
            tl += loss_fn(logits[mask], tgt[mask]).item()
            preds = logits[mask].argmax(1)
            tc += (preds == tgt[mask]).sum().item(); ts += mask.sum().item()
    return tl/len(loader), tc/max(ts,1)


def validate_ep(args, model, loader, loss_fn, device):
    model.eval(); tl, tc, ts = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x, ei, b, pe = batch.x[:, :-1], batch.edge_index, batch.batch, batch.pos_enc
            node_emb = model.gnn(x, ei, b)
            n_e   = ei.size(1); k_m = int(n_e*args.mask_ratio)
            m_idx = torch.randperm(n_e, device=device)[:k_m]
            upd   = model.transformer(node_emb, b, pe)
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
    return tl/len(loader), tc/max(ts,1)


# =====================  MAIN  =====================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Data ---
    train_set = torch.load("data/train.pt", weights_only=False)
    val_set = torch.load("data/val.pt", weights_only=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # --- Model ---
    model = CdfgNN(gnn_type=args.gnn_type,
                   gnn_input_dim=64,
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
                   edge_predictor_input_dim=1024)
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

    # --- Logging / Saving ---
    # run_dir = f"saves/joint/{datetime.now():%Y-%m-%d_%H-%M-%S}"
    # os.makedirs(run_dir, exist_ok=True)
    # json.dump(vars(args), open(f"{run_dir}/args.json", "w"), indent=4)
    # logf = open(f"{run_dir}/log.txt", "w")

    best_score = 0.0
    for epoch in range(args.num_epochs):
        tr_loss, tr_acc_mnm, tr_acc_ep = train_joint_epoch(
            args, model, train_loader, opt, loss_mnm, loss_ep, device)

        va_loss_mnm, va_acc_mnm = validate_mnm(args, model, val_loader, loss_mnm, device)
        va_loss_ep,  va_acc_ep  = validate_ep (args, model, val_loader, loss_ep,  device)

        print(f"Epoch {epoch+1:03d} │ "
              f"Loss {tr_loss:.4f} │ "
              f"MNM Acc {tr_acc_mnm:.4f}/{va_acc_mnm:.4f} │ "
              f"EP Acc  {tr_acc_ep:.4f}/{va_acc_ep:.4f}")
        # logf.write(f"{epoch+1}\t{tr_loss:.4f}\t{tr_acc_mnm:.4f}\t{tr_acc_ep:.4f}\t"
        #            f"{va_acc_mnm:.4f}\t{va_acc_ep:.4f}\n"); logf.flush()

        sched.step()

        # simple score: sum of task accuracies
        if va_acc_mnm + va_acc_ep > best_score:
            best_score = va_acc_mnm + va_acc_ep
            # torch.save({'epoch': epoch+1,
            #             'model_state_dict': model.state_dict(),
            #             'optimizer_state_dict': opt.state_dict(),
            #             'va_acc_mnm': va_acc_mnm, 'va_acc_ep': va_acc_ep},
            #            f"{run_dir}/best_model.pth")
            print("  ↳ Best model updated.")

    # logf.close()


if __name__ == "__main__":
    main()
