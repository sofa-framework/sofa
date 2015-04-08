#!/usr/bin/sed -f 


#---------------------
# Pattern of the script
#---------------------
# before: <AnyComponent ... object1="../path1" ... object2="../path2" ...
# after:  <AnyComponent ... object1="@path1"   ... object2="@path2"   ...



s|\(.*\) object1="../\([^"]*\)"|\1 object1="@\2"|g

s|\(.*\) object1="\([^"@.][^"]*\)"|\1 object1="@\2"|g

s|\(.*\) object1=".."|\1 object1="@."|g


s|\(.*\) object2="../\([^"]*\)"|\1 object2="@\2"|g

s|\(.*\) object2="\([^"@.][^"]*\)"|\1 object2="@\2"|g

s|\(.*\) object2=".."|\1 object2="@."|g
