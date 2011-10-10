#!/bin/bash

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
for f in $FLIST
do
  FDIR=${f%/*}
  FTEX=${f##*/}
  FNAME=${FTEX%.tex}
  echo '<<>>'$FNAME
  (
  cd $FDIR
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
  # Upload if given a SCP destination
  if [ -f ${FNAME}.pdf ]; then
    echo ${FNAME}.pdf generation complete
    if [ "$1" != "" ]; then
      scp -B ${FNAME}.pdf $1/$FDIR/${FNAME}.pdf
      echo ${FNAME}.pdf uploaded
    fi
  fi
  )
done
