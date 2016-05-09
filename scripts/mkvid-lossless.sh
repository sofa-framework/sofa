#!/bin/bash
if [ "$*" == "" ]; then
echo "Usage: $0 output.avi fps step input_prefix [codec] [\"options\"]"
echo "Possible codecs are: ffv1 huffyuv mjpeg h264[default] h264fast h264slow"
exit 1
fi

shopt -s nullglob
let i=0;
let n=${3:-1};
(
for f in ${4:-seq}?.png ${4:-seq}??.png ${4:-seq}???.png ${4:-seq}????.png ${4:-seq}?????.png ${4:-seq}??????.png ${4:-seq}???????.png ${4:-seq}????????.png; do
#for f in capture*.png; do
  if [ $(($i%$n)) -eq 0 ]; then
    echo $f
  fi
  let i+=1
done
) > seqlist
#find . -name "seq*.png" > seqlist
FIRST=`head -1 seqlist`
TAIL=`tail -1 seqlist`
let i=0; while [ $i -lt 30 ]; do echo $TAIL >> seqlist; let i+=1; done
RES=`identify -format %wx%h $FIRST`
if [ -d seqframes ]; then rm -rf seqframes; fi
mkdir seqframes
let n=100000
for f in `cat < seqlist`; do ln $f seqframes/frame-${n:1:5}.png || ln -s ../$f seqframes/frame-${n:1:5}.png; let n+=1; done

OPTS="-y -r ${2:-30}"
#OPTS="$OPTS -s $RES"
OPTS="$OPTS -i seqframes/frame-%05d.png"

CODEC=${5:-h264}
case "$CODEC" in
ffv1)
  OPTS="$OPTS -pix_fmt yuv420p -flags +bitexact -vcodec ffv1"
  ;;
huffyuv)
  OPTS="$OPTS -pix_fmt yuv420p -flags +bitexact -vcodec huffyuv"
  ;;
mjpeg)
  OPTS="$OPTS -flags +bitexact -vcodec mjpeg -qmin 1 -qscale 1"
  ;;
h264)
#  OPTS="$OPTS -pix_fmt yuv420p -vcodec libx264 -cqp 0 -me_method dia -subq 1 -partitions -parti4x4-parti8x8-partp4x4-partp8x8-partb8x8"
#  OPTS="$OPTS -flags +bitexact+gray+umv+aic+aiv -vcodec libx264 -cqp 0"
  OPTS="$OPTS -flags +bitexact+gray+umv+aic+aiv -vcodec libx264 -vpre lossless_medium -threads 0"
  ;;
h264fast)
  OPTS="$OPTS -flags +bitexact+gray+umv+aic+aiv -vcodec libx264 -vpre lossless_fast -threads 0"
  ;;
h264slow)
  OPTS="$OPTS -flags +bitexact+gray+umv+aic+aiv -vcodec libx264 -vpre lossless_max -threads 0"
  ;;
*)
  echo "Unknown or unsupported codec $CODEC"
  exit 1
  ;;
esac



OPTS="$OPTS ${6} ${1:-video-lossless.avi}"

echo ffmpeg $OPTS
ffmpeg $OPTS
ls -l ${1:-video-lossless.avi}
