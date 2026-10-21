# Auto-Settle AI Model Weights

All trained and fine-tuned model checkpoints used by the perception, tracking, and 3D pipelines are saved locally in `weights/` and mirrored in `backup/`:

| File | Size | Architecture | Classes / Purpose | Source |
|---|---|---|---|---|
| `parts_s_best.pt` | 23.4 MB | YOLO-seg | 33 vehicle part classes (Front/Back doors, Hood, Trunk, Bumpers, Fenders, Lights, Glass) | Fine-tuned on car-parts-segmentation |
| `dent_det_best.pt` | 124.9 MB | YOLOv11-seg | 6 damage classes: Dent, Scratch, Crack, Lamp broken, Glass shatter, Flat tire | Fine-tuned on CarDD |
| `damage_merged_s_best.pt`| 124.9 MB | YOLOv11-seg | 6 damage classes: multi-head polygon segmentation | Fine-tuned on CarDD |
| `cardd_yolov11_seg_best.pt` | 124.9 MB | YOLOv11-seg | Exact CarDD 1024px instance segmenter | Hugging Face mirror (`harpreetsahota/car-dd-segmentation-yolov11`) |
| `parts_segmentation.pt` | 23.4 MB | YOLO-seg | 33-class part segmenter with left/right panel separation | Hugging Face mirror (`mitbersh/car-parts-segmentation`) |

## Fallback & Discovery
`autosettle_vision/pipeline.py` automatically checks:
1. `backup/<name>`
2. `weights/<name>`
3. Root workspace directory
