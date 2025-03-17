#!/usr/bin/awk -f
# This script removes parts flagged by SOFA_DEV inside C source code files
# The file with the relevant parts removed is output to the standard output.

# If you never used awk, a good introduction is available at http://www.cs.hmc.edu/qref/awk.html

# init
BEGIN {
    # indicate whether the current code is filtered of not
    filtered=0;
    # indicate the number if impricated if seen so far.
    # The if conditions influence on the filtering will be stored in the ifcond array
    # i.e. if the outmost if is 'ifdef SOFA_DEV', it
    nbif=0;
}

# match preprocessor condition activating filtering
$1~/^#ifdef/ && $2=="SOFA_DEV" {
    filtered=1;
    nbif++;
    ifcond[nbif]=1;
    next;
}

# match preprocessor condition desactivating filtering
$1~/^#ifndef/ && $2=="SOFA_DEV" {
    nbif++;
    if (filtered==0)
	ifcond[nbif]=-1;
    else
	ifcond[nbif]=0; # if we are already filtered, this if has no influence
    next;
}

# match other preprocessor condition
$1~/^#if/ {
    nbif++;
    ifcond[nbif]=0;
}

# match else deactivating filtering
$1~/^#else/ && ifcond[nbif]==1 {
    filtered=0;
    ifcond[nbif]=-1;
    next;
}

# match else activating filtering
$1~/^#else/ && ifcond[nbif]==-1 {
    filtered=1;
    ifcond[nbif]=1;
    next;
}

# match other else
$1~/^#else/ && ifcond[nbif]==0 {
}

# match other else
$1~/^#elif/ && ifcond[nbif]==0 {
}

# match endif deactivating filtering
$1~/^#endif/ && ifcond[nbif]==1 {
    filtered=0;
    delete ifcond[nbif];
    nbif--;
    next;
}

# match endif activating filtering
$1~/^#endif/ && ifcond[nbif]==-1 {
    delete ifcond[nbif];
    nbif--;
    next;
}

# match other endif
$1~/^#endif/ && ifcond[nbif]==0 {
    delete ifcond[nbif];
    nbif--;
}

# debug print
#{ print filtered "(" nbif "): " $0; }

# non-filtered line: simply print the line
filtered==0 { print; }
