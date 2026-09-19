# Consistency sweep -- input=384, gt=ours+official
canvas (384, 640), content 640x360, 10000 images

| model | da_mIoU_ours | da_mIoU_official | lane_fg_iou_ours | lane_fg_iou_official | lane_mIoU_ours | lane_mIoU_official | lane_line_acc_ours | lane_line_acc_official |
|---|---|---|---|---|---|---|---|---|
| OursStatic:~/ai_study/trac/experiments/phase6/final/B200/checkpoint.pt | 0.8573 | 0.8624 | 0.2182 | 0.1950 | 0.5975 | 0.5853 | 0.8774 | 0.8405 |
