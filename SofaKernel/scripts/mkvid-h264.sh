#!/bin/bash
if [ "$*" == "" ]; then
echo "Usage: $0 output.avi fps step input_prefix bitrate [\"options\"]"
echo "If the result is flipped vertically, add -z in options"
echo "If the red/blue channels are swapped, add -k in options"
exit 1
fi

shopt -s nullglob
let i=0;
let n=${3:-1};
(
for f in ${4:-seq}?.png ${4:-seq}??.png ${4:-seq}???.png ${4:-seq}????.png ${4:-seq}?????.png ${4:-seq}??????.png ${4:-seq}???????.png ${4:-seq}????????.png ${4:-seq}?.bmp ${4:-seq}??.bmp ${4:-seq}???.bmp ${4:-seq}????.bmp ${4:-seq}?????.bmp ${4:-seq}??????.bmp ${4:-seq}???????.bmp ${4:-seq}????????.bmp; do
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
OPTS="-i seqlist -x imlist,null -H 0 -g $RES  -w ${5:-1000} -y ffmpeg,null -F h264 -f ${2:-30} -w ${5:-1000}"
TRANSVER=`transcode --version 2>&1 | awk ' { print $2; } '`
if [ "${TRANSVER:0:4}" == "v1.0" ]; then
  echo Identified transcode v1.0.x
  OPTS="$OPTS --use_rgb"
elif [ "${TRANSVER:0:4}" == "v1.1" ]; then
  echo Identified transcode v1.1.x
  OPTS="$OPTS -V rgb24"
else
  echo Unknown transcode version $TRANSVER
fi
echo transcode $OPTS -o ${1:-video-h264.avi} $6
transcode $OPTS -o ${1:-video-h264.avi} $6
ls -l ${1:-video-h264.avi}
