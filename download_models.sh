#bin/sh
mkdir models
cd models
wget deep-temporal-seg.informatik.uni-freiburg.de/models/squeeze_seg_models.zip
unzip squeeze_seg_models.zip
cp models/* .
rm -r models
rm squeeze_seg_models.zip
