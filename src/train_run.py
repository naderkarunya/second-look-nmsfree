"""
train_run.py — one VG-YOLO training run from the command line (used to run two experiments in parallel,
one per GPU, on Kaggle's free "GPU T4 x2").

  python train_run.py --mode vg --base yolo26s-p2 --data /kaggle/working/ch_yolo/crowdhuman.yaml \
      --epochs 60 --seed 0 --name yolo26s-p2_vg_s0 --project /kaggle/working/runs --device 0
  python train_run.py --resume /path/to/last.pt --mode vg      (continue an interrupted run)
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import vgyolo  # noqa: E402
from ultralytics import YOLO  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["baseline", "vpb", "asff", "vis_off", "no_prior", "vg", "vg_oracle", "vg_sched"])
ap.add_argument("--base", default="yolo26s-p2")
ap.add_argument("--data", default=None)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--fraction", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--device", default="0")
ap.add_argument("--project", default="runs")
ap.add_argument("--name", default=None)
ap.add_argument("--pretrained", default=None, help="default: COCO weights of the base, e.g. yolo26s.pt")
ap.add_argument("--vis-weight", type=float, default=0.0, help="Option B: beta for occlusion-aware loss weighting")
ap.add_argument("--no-vw-norm", action="store_true", help="Option B: do not rescale weights to mean 1")
ap.add_argument("--vra", default="none", choices=["none", "o2o", "all"], help="visible-region assignment")
ap.add_argument("--crm", action="store_true", help="add the Crowd Relation Module to the one-to-one branch")
ap.add_argument("--hg", action="store_true", help="add the Head-Guided module (needs head boxes in the labels)")
ap.add_argument("--vba", action="store_true", help="add Visible-Body Attention to the neck outputs")
ap.add_argument("--max-hours", type=float, default=0.0, help="emergency brake: stop training (keeping best.pt) if it exceeds this many hours; epochs stay fixed")
ap.add_argument("--close-mosaic", type=int, default=10)
ap.add_argument("--resume", default=None, help="path to last.pt of an interrupted run")
a = ap.parse_args()

vgyolo.VG_CFG["mode"] = a.mode
vgyolo.VG_CFG["vis_weight"] = a.vis_weight
vgyolo.VG_CFG["vis_weight_norm"] = not a.no_vw_norm
vgyolo.VG_CFG["vra"] = a.vra
vgyolo.VG_CFG["crm"] = a.crm
vgyolo.VG_CFG["hg"] = a.hg
vgyolo.VG_CFG["vba"] = a.vba
t0 = time.time()
if a.resume:
    m = YOLO(a.resume)
    m.train(trainer=vgyolo.VGTrainer, resume=True)
else:
    name = a.name or f"{a.base}_{a.mode}_s{a.seed}_e{a.epochs}_f{a.fraction}"
    pre = a.pretrained or a.base.split("-")[0] + ".pt"
    m = YOLO(f"{a.base}.yaml")
    if a.max_hours > 0:
        def _brake(trainer):
            if (time.time() - trainer.train_time_start) > a.max_hours * 3600:
                print(f"MAX_HOURS reached ({a.max_hours} h) - stopping after epoch {trainer.epoch + 1}")
                trainer.stop = True
        m.add_callback("on_fit_epoch_end", _brake)
    m.train(
        trainer=vgyolo.VGTrainer, data=a.data, pretrained=pre, epochs=a.epochs, fraction=a.fraction,
        imgsz=a.imgsz, batch=a.batch, seed=a.seed, deterministic=True, max_det=500, workers=a.workers,
        device=a.device, project=a.project, name=name, exist_ok=True, save_period=10, plots=True,
        close_mosaic=a.close_mosaic)
save_dir = Path(m.trainer.save_dir)
hours = (time.time() - t0) / 3600
info = dict(mode=a.mode, max_hours=a.max_hours, epochs_done=int(getattr(m.trainer, "epoch", -1)) + 1, vis_weight=a.vis_weight, vis_weight_norm=not a.no_vw_norm, vra=a.vra, crm=a.crm, hg=a.hg, vba=a.vba, base=a.base, epochs=a.epochs, fraction=a.fraction, seed=a.seed, imgsz=a.imgsz,
            batch=a.batch, train_hours=round(hours, 3), save_dir=str(save_dir))
(save_dir / "run_info.json").write_text(json.dumps(info, indent=2))
print("RUN_INFO", json.dumps(info))
