#!/bin/bash
for f in "$@"; do
pdfinfo $f | awk 'BEGIN { done=0 } $1 == "Title:" && NF>1 { $1=""; print "'$f':", $0; done=1 } END { if (!done) print "'$f'" }'
done
(
echo '<html><head><title>Sofa Documentation</title></head><body><h1>Sofa Documentation</h1>The most complete documentation can be found on the <a href="http://wiki.sofa-framework.org/">Sofa Wiki</a><br><br>Other materials:<ul>'
for f in "$@"; do
  FDIR=${f%/*}
  FPDF=${f##*/}
  FNAME=${FPDF%.pdf}
  FTITLE=$(pdfinfo $f | awk 'BEGIN { done=0 } $1 == "Title:" && NF>1 { $1=""; print "'$FNAME':", $0; done=1 } END { if (!done) print "'$FNAME'" }')
  echo '<li><a href="'$f'">'$FTITLE'</a></li>'
done
echo '</ul></body></html>'
) > index.html
