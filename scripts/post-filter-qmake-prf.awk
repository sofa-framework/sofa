#!/usr/bin/awk -f
# This script removes sub-projects in a dependencies-based qmake project file
# corresponding to files that are no longer present (removed by other scripts)

# If you never used awk, a good introduction is available at http://www.cs.hmc.edu/qref/awk.html

# Usage: post-filter-qmake-prf.awk -v features=features pass=1 sofa-dependencies.prf pass=2 sofa-dependencies.prf

# init
BEGIN {
    FS=","
    OFS=","
}

END {
}

/^[ \t#]*declare/ {
  name=$1;
  gsub(/.*\(/,"",name); gsub(/ /,"",name); gsub(/\t/,"",name);
  path=$2;
  gsub(/\).*$/,"",path); gsub(/ /,"",path); gsub(/\t/,"",path);
  deps=$3;
  gsub(/\).*$/,"",deps); gsub("\t"," ",deps); gsub(/[ ]*$/,"",deps);
  postfix=$3;
  gsub(/^[^\)]*\)/,")",postfix);
  #print "#PROJECT <<<" name "|" path "|" deps ">>>" > "/dev/stderr";
  projects[name] = path;
  found = 0;
  if (system("test -f " path) == 0)
      found = path;
  else if (features && system("test -f " features "/sofa/" path ) == 0)
      found = features "/sofa/" path;
  else if (system("test -d " path) == 0)
  {
      pro = path;
      sub(/^.*\//,"",pro)
      if (system("test -f " path "/" pro ".pro" ) == 0)
          found = path "/" pro ".pro";
  }
  if (!found) #system("test -e " path) != 0 && (!features || system("test -e " features "/sofa/" path ) != 0) )
  {
      if (pass != 2) print "# " name ": PATH " path " NOT FOUND" > "/dev/stderr";
      projMissing[name] = path;
      next;
  }
  else
  {
      #if (pass != 2) print "# " name ": PATH " path " FOUND IN " found > "/dev/stderr";
      projFound[name] = path;
  }
  if (pass != 1)
  {
      n = split(deps,deparray, " ");
      removed=0
      newdeps=""
      for (d in deparray)
      {
          dep = deparray[d];
          #print "# " d " / " n ": " dep > "/dev/stderr";
          if (!(dep in projFound))
          {
              print "# " name ": DEPENDENCY " dep " NOT FOUND" > "/dev/stderr";
#              sub(dep,"",$3);
              removed = removed+1
          }
          else
              newdeps = newdeps " " dep
      }
      if (removed > 0)
          $3 = newdeps postfix
  }
}

pass!=1 && /^[ \t#]*enable/ {
  name=$0;
  gsub(/.*\(/,"",name); gsub(/,.*$/,"",name); gsub(/\).*$/,"",name); gsub(/ /,"",name);
  if (!(name in projFound)) next;
  path = projFound[name];
}

# other: simply print the line
pass!=1 { print }
