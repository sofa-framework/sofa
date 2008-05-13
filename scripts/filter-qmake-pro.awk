#!/usr/bin/awk -f
# This script removes parts flagged by SOFA_DEV inside qmake project files
# The file with the relevant parts removed is output to the standard output,
# while files mentionned inside removed parts are output to the standard error.
# This can be used to remove then from the repository by svn rm'ing them.

# If you never used awk, a good introduction is available at http://www.cs.hmc.edu/qref/awk.html

# init
BEGIN {
    infilelist=0;
}

# Match blocks conditionnally included depending on the SOFA_DEV flag

/contains[ \t]*\([ \t]*DEFINES[ \t]*,[ \t]*SOFA_DEV[ \t]*\)[ \t]*{/,/[ \t]*}.*SOFA_DEV/ {
#    for(f=1;f<=NF && !(f==NF && $f~/\\[:space:]*/);f++)
#        print f " -> \"" $f "\""
    # look for a filename
    # we assume files and directories are listed inside variables containing SUBDIRS, SOURCES or HEADERS
    f0=1;
    if (infilelist==0) {
	if (($1~"SUBDIRS" || $1~"SOURCES" || $1~"HEADERS") && ($2=="=" || $2=="+=")) {
#	    print "infilelist " $1
	    infilelist=1;
	    f0=3; # first file is in 3rd field
	}
    }
    if (infilelist) {
	for(f=f0;f<=NF && !(f==NF && $f~/\\[:space:]*/);f++) {
	    fname=$f
	    gsub(/[\r\n]/,"",fname);
	    if (fname != "#") print fname > "/dev/stderr";
	}
	if (f>NF) { # no "\\" is put at the end of the line -> end of list
#	    print "end"
	    infilelist=0;
	}
    }
    next;
}

# Match lines mentionning SOFA_DEV, such as the point where it is defined
/SOFA_DEV/ { next; }

# other: simply print the line
{ print; }
