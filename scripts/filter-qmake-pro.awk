#!/usr/bin/awk -f
# This script removes parts flagged by a given FILTER_TAG inside qmake project files.
# The file with the relevant parts removed is output to the standard output,
# while files mentionned inside removed parts are output to the standard error.
# This can be used to remove then from the repository by svn rm'ing them.

# If you never used awk, a good introduction is available at http://www.cs.hmc.edu/qref/awk.html

# init
BEGIN {
    infilelist=0;
    blk_before=0;
    blk_after=0;
    blk_filtered=0;
    if (plugins == "")
        plugins = "applications/plugins";
    if (FILTER_TAG == "")
        FILTER_TAG = "SOFA_RELEASE";
}

{ blk_before=blk_after; }

/[{}]/ {
  split($0,a,"#")
  if (a[1]!="") {
    l = a[1];
    beg[1]=""; nbeg = split(l,beg,"{")
    end[1]=""; nend = split(l,end,"}")
    blk_after = blk_before + nbeg - nend
  }
}

END {
    if (blk_after != 0) print "# ERROR: unmatched brackets"
#    print "# Filtered with " FILTER_TAG
}

# Match blocks conditionnally included depending on the SOFA_DEV flag
# Note that we now count the brackets in the source file to find the
# SOFA_DEV closing one instead of relying on the presence of a comment

blk_filtered==0 && $0 ~ "contains[ \t]*\\([ \t]*DEFINES[ \t]*,[ \t]*"FILTER_TAG"[ \t]*\\)[ \t]*{" {
    blk_filtered=blk_after;
    infilelist=0;
}

blk_filtered>0 && blk_after < blk_filtered {
    blk_filtered=0;
    next;
}

/^[ \t#]*declare/ {
  name=$0;
  gsub(/.*\(/,"",name); gsub(/,.*$/,"",name); gsub(/ /,"",name); gsub(/\t/,"",name);
  path=$0;
  gsub(/.*\([^,]*,/,"",path); gsub(/,.*$/,"",path); gsub(/\).*$/,"",path); gsub(/ /,"",path); gsub(/\t/,"",path);
#  gsub(/\).*$/,"",path); gsub(/,$/,"",path); gsub(/ /,"",path);
  #print "#PROJECT <<<",name,"|",path,">>>";
  projects[name] = path;
}
blk_filtered>0 && /^[ \t#]*enable/ {
  name=$0;
  gsub(/.*\(/,"",name); gsub(/,.*$/,"",name); gsub(/\).*$/,"",name); gsub(/ /,"",name); gsub(/\t/,"",name);
  path = projects[name];
  #print "#DEV PROJECT <<<",name,"|",path,">>>";
  print path > "/dev/stderr";
  next;
}
blk_filtered>0 && /^[ \t#]*usePlugin/ {
  name=$0;
  gsub(/.*\(/,"",name); gsub(/,.*$/,"",name); gsub(/\).*$/,"",name); gsub(/ /,"",name); gsub(/\t/,"",name);
  #print "#DEV PLUGIN <<<",name,">>>";
  print plugins "/" name > "/dev/stderr";
  next;
}

blk_filtered>0 {
    #print "# " $0
    # look for a filename
    # we assume files and directories are listed inside variables containing SUBDIRS, SOURCES or HEADERS
    f0=1;
    nf=NF;
    if ($nf~/\r/) {
      #print "#CR <" $nf ">"
      gsub(/[\r\n]/,"",$nf);
      #print "#CRF <" $nf ">"
      if ($nf=="") nf--;
    }
    if (infilelist==0) {
	if (($1~"SUBDIRS" || $1~"SOURCES" || $1~"HEADERS") && ($2=="=" || $2=="+=")) {
	    #print "#infilelist " $1
	    infilelist=1;
	    f0=3; # first file is in 3rd field
	}
	else if ($1~"###") {
	    #print "#infilelist " $1
	    infilelist=1;
	    f0=2; # first file is in 3rd field
	}
    }
    if (infilelist) {
	for(f=f0;f<=nf && !(f==nf && $f~/^[:space:]*\\[:space:]*$/);f++) {
	    fname=$f
	    gsub(/[\r\n]/,"",fname);
	    gsub(/\\[:space:]*$/,"",fname);
	    gsub(/^##*/,"",fname);
	    if (fname != "") print fname > "/dev/stderr";
	}
	if (f>nf && (nf==0 || $nf!~/\\[:space:]*$/)) { # no "\\" is put at the end of the line -> end of list
	    #print "#end"
	    infilelist=0;
	}
    }
    next;
}

# Match lines mentionning FILTER_TAG, such as the point where it is defined
$0 ~ FILTER_TAG { next; }

# other: simply print the line
#{ print blk_before "<" $0 ">" blk_after; }
{ print }
