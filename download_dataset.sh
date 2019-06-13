#bin/sh
mkdir datasets
cd datasets
wget deep-temporal-seg.informatik.uni-freiburg.de/datasets/squeeze_seg.zip
unzip squeeze_seg.zip
mv squeeze_seg/* .
rm -r squeeze_seg 
rm squeeze_seg.zip
