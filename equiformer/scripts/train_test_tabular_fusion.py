# ---------- scripts/train_test_tabular_fusion.py ----------
from __future__ import annotations
import os, sys, json, argparse, subprocess, datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

class TabularStandardizer:
    def __init__(self): self.mean_, self.std_ = {}, {}
    def fit(self, df: pd.DataFrame, cols: List[str]):
        self.mean_ = {c: float(df[c].mean()) for c in cols}
        self.std_  = {c: float(df[c].std(ddof=0) or 1.0) for c in cols}
    def transform(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        X = df[cols].to_numpy(dtype=np.float32)
        mean = np.array([self.mean_[c] for c in cols], dtype=np.float32)
        std  = np.array([self.std_[c]  for c in cols], dtype=np.float32)
        std[std==0] = 1.0
        return (X - mean) / std
    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mean": self.mean_, "std": self.std_}, f, indent=2)
    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.mean_, self.std_ = obj["mean"], obj["std"]

class TabularMLP(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class FusionHead(nn.Module):
    def __init__(self, fusion="concat"):
        super().__init__()
        assert fusion in ("concat","gate")
        self.fusion = fusion
        self.struct_proj = nn.Linear(1, 64)
        if fusion == "concat":
            self.mlp = nn.Sequential(
                nn.Linear(64+64, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.1),
                nn.Linear(64, 1),
            )
        else:
            self.gate = nn.Linear(64, 1)
            self.mlp  = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(64,1))
    def forward(self, y_struct_scalar, h_feat):
        s = self.struct_proj(y_struct_scalar)  # (B,1)->(B,64)
        if self.fusion == "concat":
            x = torch.cat([s, h_feat], dim=-1)
            return self.mlp(x)
        g = torch.sigmoid(self.gate(h_feat))
        h = s * g + h_feat
        return self.mlp(h)

def mae(a,b): return float(np.mean(np.abs(a-b)))
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

def parse_cols(df: pd.DataFrame, user_cols: str) -> List[str]:
    if user_cols.strip():
        cols = [c.strip() for c in user_cols.split(",") if c.strip() in df.columns]
        assert cols, f"--tab_cols 无有效列：{user_cols}"
        return cols
    exclude = {"target","label","delta_g","delta_g_h","dg_h","Δg_h","dg","y","energy","e"}
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in num_cols if c.lower() not in exclude][:10]
    assert cols, "自动选择特征列失败，请用 --tab_cols 指定"
    return cols

def align_merge(struct_df: pd.DataFrame, tab_df: pd.DataFrame, id_col: Optional[str], tab_cols: List[str]) -> pd.DataFrame:
    assert "idx" in struct_df.columns or (id_col and id_col in struct_df.columns), \
        "结构CSV需要包含 idx 或与 --tab_id_col 同名的 id 列"
    if id_col and id_col in tab_df.columns and id_col in struct_df.columns:
        m = pd.merge(struct_df, tab_df[[id_col]+tab_cols], how="inner", on=id_col)
        m = m.sort_values(by="idx" if "idx" in m.columns else id_col).reset_index(drop=True)
        m["_align_mode"] = "id"
        return m
    else:
        assert "idx" in struct_df.columns, "未提供 --tab_id_col 时必须提供 idx 对齐"
        tab_df = tab_df.copy()
        # 如果表格数据已经有idx列，不要重复添加
        if "idx" not in tab_df.columns:
            tab_df["idx"] = np.arange(len(tab_df))
        m = pd.merge(struct_df, tab_df[["idx"]+tab_cols], how="inner", on="idx")
        m = m.sort_values(by="idx").reset_index(drop=True)
        m["_align_mode"] = "idx"
        return m

def train_fusion(train_df, val_df, tab_cols, tab_std, fusion, epochs, batch_size, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.tensor(tab_std.transform(train_df, tab_cols), dtype=torch.float32).to(device)
    ytr_struct = torch.tensor(train_df["y_pred"].to_numpy().reshape(-1,1), dtype=torch.float32).to(device)
    ytr = torch.tensor(train_df["y_true"].to_numpy().reshape(-1,1), dtype=torch.float32).to(device)

    Xval = torch.tensor(tab_std.transform(val_df, tab_cols), dtype=torch.float32).to(device)
    yval_struct = torch.tensor(val_df["y_pred"].to_numpy().reshape(-1,1), dtype=torch.float32).to(device)
    yval = torch.tensor(val_df["y_true"].to_numpy().reshape(-1,1), dtype=torch.float32).to(device)

    tab_mlp = TabularMLP(len(tab_cols)).to(device)
    head = FusionHead(fusion=fusion).to(device)
    opt = torch.optim.AdamW(list(tab_mlp.parameters())+list(head.parameters()), lr=1e-3, weight_decay=1e-3)
    best = {"mae": 1e9, "state": None}; patience, bad=20, 0

    for ep in range(1, epochs+1):
        tab_mlp.train(); head.train()
        idx = np.random.permutation(len(Xtr))
        for i in range(0, len(idx), batch_size):
            sl = idx[i:i+batch_size]
            h = tab_mlp(Xtr[sl])
            yhat = head(ytr_struct[sl], h)
            loss = F.l1_loss(yhat, ytr[sl])
            opt.zero_grad(); loss.backward(); opt.step()

        # val
        tab_mlp.eval(); head.eval()
        with torch.no_grad():
            hv = tab_mlp(Xval); yhatv = head(yval_struct, hv)
        mae_val = F.l1_loss(yhatv, yval).item()
        if mae_val < best["mae"]:
            best = {"mae": mae_val,
                    "state": {"tab_mlp": tab_mlp.state_dict(), "head": head.state_dict()}}
            bad = 0
        else:
            bad += 1
        print(f"[Fusion][{ep}/{epochs}] val MAE={mae_val:.4f} best={best['mae']:.4f} bad={bad}")
        if bad >= patience: break

    # load best
    tab_mlp.load_state_dict(best["state"]["tab_mlp"]); head.load_state_dict(best["state"]["head"])
    return tab_mlp, head, best["mae"]

def eval_fusion(tab_mlp, head, df, tab_cols, tab_std, device, mc_T=0):
    X = torch.tensor(tab_std.transform(df, tab_cols), dtype=torch.float32).to(device)
    ys = torch.tensor(df["y_pred"].to_numpy().reshape(-1,1), dtype=torch.float32).to(device)
    tab_mlp.eval(); head.eval()
    with torch.no_grad():
        h = tab_mlp(X)
        yhat = head(ys, h).cpu().numpy().reshape(-1)
    if mc_T and mc_T>0:
        preds = []
        tab_mlp.train(); head.train()
        for _ in range(mc_T):
            with torch.no_grad():
                h = tab_mlp(X)
                p = head(ys, h).cpu().numpy().reshape(-1)
            preds.append(p)
        preds = np.stack(preds, axis=1)  # [N,T]
        return yhat, preds.mean(1), preds.var(1)
    return yhat, None, None

def plot_basic(y_true, y_pred, outdir):
    ensure_dir(outdir)
    import matplotlib.pyplot as plt
    plt.figure(); plt.scatter(y_true, y_pred, s=6, alpha=0.5); plt.xlabel("True"); plt.ylabel("Pred"); plt.grid(True); plt.savefig(os.path.join(outdir,"predictions_vs_true.png"), dpi=200); plt.close()
    res = y_pred - y_true
    plt.figure(); plt.plot(res, ".", alpha=0.5); plt.axhline(0,color="r",ls="--"); plt.ylabel("Residual"); plt.savefig(os.path.join(outdir,"residuals.png"), dpi=200); plt.close()
    plt.figure(); plt.hist(res, bins=40, alpha=0.8); plt.xlabel("Residual"); plt.savefig(os.path.join(outdir,"residuals_dist.png"), dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","test","train_test"], default="train_test")
    ap.add_argument("--save_dir", type=str, default="")
    ap.add_argument("--struct_train_cmd", type=str, default="")
    ap.add_argument("--struct_pred_train", type=str, required=True)
    ap.add_argument("--struct_pred_val", type=str, required=True)
    ap.add_argument("--struct_pred_test", type=str, required=True)
    ap.add_argument("--tab_csv", type=str, default="data/processed/cleaned_data.csv")
    ap.add_argument("--tab_cols", type=str, default="")
    ap.add_argument("--tab_id_col", type=str, default="")
    ap.add_argument("--tab_norm_json", type=str, default="data/processed/tabular_norm.json")
    ap.add_argument("--fusion", type=str, choices=["concat","gate"], default="concat")
    ap.add_argument("--tab_dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mc_dropout_T", type=int, default=0)
    ap.add_argument("--extra_note", type=str, default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = f"{now_tag()}_tabfusion_run"
    if args.extra_note: run_name += f"_{args.extra_note}"
    save_dir = args.save_dir or os.path.join("experiments", run_name)
    ensure_dir(save_dir)

    # 1) 可选：训练结构模型（保留原脚本不动）
    if args.mode in ("train","train_test") and args.struct_train_cmd.strip():
        print(f"[STRUCT] run: {args.struct_train_cmd}")
        ret = subprocess.call(args.struct_train_cmd, shell=True)
        if ret != 0:
            print("[STRUCT] 子进程训练失败"); sys.exit(1)

    # 2) 读取结构预测 CSV
    tr = pd.read_csv(args.struct_pred_train); vl = pd.read_csv(args.struct_pred_val); te = pd.read_csv(args.struct_pred_test)
    for dfn, df in [("train",tr),("val",vl),("test",te)]:
        assert "y_pred" in df.columns and "y_true" in df.columns, f"{dfn} 缺列 y_pred/y_true"
        if "idx" not in df.columns and (args.tab_id_col and args.tab_id_col not in df.columns):
            raise ValueError(f"{dfn} 缺少 idx 且未提供 --tab_id_col 对齐依据")

    # 3) Tabular 读取与列选择
    tab_df = pd.read_csv(args.tab_csv)
    tab_cols = parse_cols(tab_df, args.tab_cols)

    # 4) 对齐（优先 id，对齐失败则 idx）
    tr_m = align_merge(tr, tab_df, args.tab_id_col, tab_cols)
    vl_m = align_merge(vl, tab_df, args.tab_id_col, tab_cols)
    te_m = align_merge(te, tab_df, args.tab_id_col, tab_cols)
    align_mode = tr_m["_align_mode"].iloc[0]
    print(f"[ALIGN] mode={align_mode} cols={tab_cols}")

    # 5) 标准化器
    std = TabularStandardizer()
    if os.path.exists(args.tab_norm_json):
        std.load(args.tab_norm_json)
    else:
        std.fit(tab_df, tab_cols); ensure_dir(os.path.dirname(args.tab_norm_json)); std.save(args.tab_norm_json)

    # 6) 训练融合 MLP
    if args.mode in ("train","train_test"):
        tab_mlp, head, best_mae = train_fusion(tr_m, vl_m, tab_cols, std, args.fusion, args.epochs, args.batch_size, args.seed, device)
        torch.save({"tab_mlp": tab_mlp.state_dict(), "head": head.state_dict(), "tab_cols": tab_cols, "align": align_mode}, os.path.join(save_dir,"fusion_ckpt.pt"))
    else:
        ck = torch.load(os.path.join(save_dir,"fusion_ckpt.pt"), map_location=device)
        tab_mlp, head = TabularMLP(len(ck["tab_cols"])).to(device), FusionHead(args.fusion).to(device)
        tab_mlp.load_state_dict(ck["tab_mlp"]); head.load_state_dict(ck["head"])
        tab_cols = ck["tab_cols"]

    # 7) 测试与落盘
    y_true = te_m["y_true"].to_numpy()
    y_struct = te_m["y_pred"].to_numpy()
    y_hat, mu, var = eval_fusion(tab_mlp, head, te_m, tab_cols, std, device, mc_T=args.mc_dropout_T)

    # 计算Test Loss (MSE)
    test_loss = float(np.mean((y_true - y_hat) ** 2))
    m = {"test_mae": mae(y_true, y_hat), "test_rmse": rmse(y_true, y_hat), "test_loss": test_loss, "align_mode": align_mode, "fusion": args.fusion, "mc_T": args.mc_dropout_T}
    with open(os.path.join(save_dir,"metrics.json"),"w") as f: json.dump(m,f,indent=2)
    pd.DataFrame([m]).to_csv(os.path.join(save_dir,"metrics.csv"), index=False)

    out_df = pd.DataFrame({"idx": te_m.get("idx", pd.Series(range(len(y_hat)))), "y_true": y_true, "y_struct": y_struct, "y_hat": y_hat})
    out_df.to_csv(os.path.join(save_dir,"predictions_test.csv"), index=False)

    if mu is not None:
        pd.DataFrame({"idx": out_df["idx"], "y_hat_mean": mu, "y_hat_var": var}).to_csv(os.path.join(save_dir,"predictions_with_uncertainty.csv"), index=False)

    if plt is not None:
        plot_basic(y_true, y_hat, save_dir)

    # 简单 Top-K 排序（按 |y_hat| 升序）
    topk = out_df.copy()
    topk["abs_y_hat"] = np.abs(topk["y_hat"])
    topk = topk.sort_values("abs_y_hat").head(200)
    topk.to_csv(os.path.join(save_dir,"topk_rank.csv"), index=False)

    print("[DONE] metrics:", m, " save_dir=", save_dir)

if __name__ == "__main__":
    main()
# ---------- end script ----------
