
# EPIC-KITCHENS
# Pre-train
wget https://utexas.box.com/shared/static/yp1krj3dsmr8wj0sz01t10bwa9fgq3zy.pt -O avion_pretrain_baseline_vitb_best.pt
wget https://utexas.box.com/shared/static/e681nrxivc9makufvrumrfuaopk57h4n.pt -O avion_pretrain_lavila_vitb_best.pt
wget https://utexas.box.com/shared/static/1iatmrs7ufdeooce09a61t1n6wsouf4l.pt -O avion_pretrain_lavila_vitl_best.pt

# Fine-tune (CLS)
wget https://utexas.box.com/shared/static/2fkvtc67m0f82wmm5cnqfo7wg951lobv.pt -O avion_finetune_cls_baseline_vitb_best.pt
wget https://utexas.box.com/shared/static/2fkvtc67m0f82wmm5cnqfo7wg951lobv.pt -O avion_finetune_cls_lavila_vitb_best.pt
wget https://utexas.box.com/shared/static/crnqo9bu0owtfz4yc1yqf8hz6g0ze39b.pt -O avion_finetune_cls_lavila_vitl_best.pt

# Fine-tune (MIR)
wget https://utexas.box.com/shared/static/ke5kwfixttb4t7uxdbs9gmiiuu1582dg.pt -O avion_finetune_mir_lavila_vitb_best.pt
wget https://utexas.box.com/shared/static/m7f65hg9eonz34g0l2x5r0t92ouh0u4w.pt -O avion_finetune_mir_lavila_vitl_best.pt

# Kinetics (VideoMAE)
wget https://utexas.box.com/shared/static/61vjh8k4q3ia8wlns0rmkbnazzxipua9.pt -O avion_videomae_pretrain_vitb.pt
wget https://utexas.box.com/shared/static/p9tigkrop86f60ae6o85nbxfwh53dghm.pt -O avion_videomae_finetune_vitb_best.pt
