# Consistency sweep -- input=384, gt=ours+official
canvas (384, 640), content 640x360, 10000 images

| model | da_mIoU_ours | da_mIoU_official | lane_fg_iou_ours | lane_fg_iou_official | lane_mIoU_ours | lane_mIoU_official | lane_line_acc_ours | lane_line_acc_official |
|---|---|---|---|---|---|---|---|---|
| OursStatic:~/ai_study/trac/experiments/phase6/final/B100/checkpoint.pt | 0.8594 | 0.8649 | 0.2173 | 0.1941 | 0.5970 | 0.5848 | 0.8773 | 0.8401 |
| OursStatic:~/ai_study/trac/experiments/phase6/final/B100_s1/checkpoint.pt | 0.8631 | 0.8683 | 0.2104 | 0.1887 | 0.5927 | 0.5813 | 0.8868 | 0.8497 |
| OursStatic:~/ai_study/trac/experiments/phase6/final/B100_s2/checkpoint.pt | 0.8626 | 0.8677 | 0.2203 | 0.1954 | 0.5989 | 0.5858 | 0.8715 | 0.8330 |
