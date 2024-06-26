#!/bin/sh
fb_status=$(wget --spider -S https://filebox.ece.vt.edu/ 2>&1 | grep  "HTTP/1.1 200 OK")

mkdir checkpoints

echo "downloading from filebox ..."
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/color-model.pth
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/depth-model.pth
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/edge-model.pth
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/model.pt
# echo "downloading from other link ..."
# wget https://github.com/LemonATsu/3d-photo-model-weights/raw/master/color-model.pth
# wget https://github.com/LemonATsu/3d-photo-model-weights/raw/master/depth-model.pth
# wget https://github.com/LemonATsu/3d-photo-model-weights/raw/master/edge-model.pth
# wget https://github.com/LemonATsu/3d-photo-model-weights/raw/master/model.pt

mv color-model.pth checkpoints/.
mv depth-model.pth checkpoints/.
mv edge-model.pth checkpoints/.
mv model.pt MiDaS/.

echo "cloning from BoostingMonocularDepth ..."
git clone https://github.com/compphoto/BoostingMonocularDepth.git
mkdir -p BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/

echo "downloading mergenet weights ..."
# wget https://sfu.ca/~yagiz/CVPR21/latest_net_G.pth
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/latest_net_G.pth
mv latest_net_G.pth BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/
wget https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt
mv model-f46da743.pt BoostingMonocularDepth/midas/model.pt


echo "cloning from Depth Anything"
git clone https://github.com/GengLUO/Depth-Anything.git