#!/bin/bash

gen-index () {
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
}

LATEX=1

if [ "$1" == "--nolatex" ]; then
  LATEX=0
  shift
fi

export TEXINPUTS=..:../..:${TEXINPUTS}
LATEX_OPTIONS="-no-shell-escape -no-parse-first-line -interaction=batchmode"


FLIST="sofaDocumentation/sofadocumentation.tex"
for f in */*.tex
do
  if [ "$f" != "sofaDocumentation/sofadocumentation.tex" ]; then
    if head -1 $f | grep -q '\documentclass'; then
      FLIST="$FLIST $f"
    fi
  fi
done
echo $FLIST
PDFLIST=""
for f in $FLIST
do
  FDIR=${f%/*}
  FTEX=${f##*/}
  FNAME=${FTEX%.tex}
  echo '<<>>'$FNAME
  (
  cd $FDIR
  if [ $LATEX -ne 0 ]; then
    # commands inspired by http://www.acoustics.hut.fi/u/mairas/UltimateLatexMakefile/Makefile
    if test -r ${FNAME}.toc; then cp -f ${FNAME}.toc ${FNAME}.toc.bak; fi; pdflatex $FNAME $LATEX_OPTIONS < /dev/null
    egrep -c "No file.*\.bbl|Citation.*undefined" ${FNAME}.log && (bibtex $FNAME; if test -r ${FNAME}.toc; then cp -f ${FNAME}.toc ${FNAME}.toc.bak; fi; pdflatex $FNAME $LATEX_OPTIONS < /dev/null)
    egrep "(There were undefined references|Rerun to get (cross-references|the bars) right)" ${FNAME}.log && (if test -r ${FNAME}.toc; then cp -f ${FNAME}.toc ${FNAME}.toc.bak; fi; pdflatex $FNAME $LATEX_OPTIONS < /dev/null)
    egrep "(There were undefined references|Rerun to get (cross-references|the bars) right)" ${FNAME}.log && (if test -r ${FNAME}.toc; then cp -f ${FNAME}.toc ${FNAME}.toc.bak; fi; pdflatex $FNAME $LATEX_OPTIONS < /dev/null)
    if cmp -s ${FNAME}.toc ${FNAME}.toc.bak; then true; else pdflatex $FNAME $LATEX_OPTIONS < /dev/null; fi
    rm -f ${FNAME}.toc.bak
    # Display relevant warnings
    egrep -i "(Reference|Citation).*undefined" ${FNAME}.log 1>&2
    rm -f ${FNAME}.log ${FNAME}.aux ${FNAME}.dvi ${FNAME}.bbl ${FNAME}.blg ${FNAME}.ilg ${FNAME}.toc ${FNAME}.lof ${FNAME}.lot ${FNAME}.idx ${FNAME}.ind ${FNAME}.out ${FNAME}.log ${FNAME}.vrb ${FNAME}.snm ${FNAME}.nav
  fi
  )
  if [ -f ${FDIR}/${FNAME}.pdf ]; then
    if [ $LATEX -ne 0 ]; then
      echo ${FNAME}.pdf generation complete
    fi
    PDFLIST="${PDFLIST} ${FDIR}/${FNAME}.pdf"
    # Upload if given a SCP destination
    if [ "$1" != "" ]; then
      ssh ${1%%:*} mkdir -p ${1#*:}/$FDIR
      scp -B ${FDIR}/${FNAME}.pdf $1/$FDIR/${FNAME}.pdf
      echo ${FNAME}.pdf uploaded
    fi
  fi
done
echo '<<>>Index'
echo 'PDFs:' $PDFLIST
gen-index $PDFLIST
if [ "$1" != "" ]; then
  scp -B index.html $1/index.html
  echo Index uploaded
fi
