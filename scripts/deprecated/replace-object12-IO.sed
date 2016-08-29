#!/usr/bin/sed -f 


#---------------------
# Pattern of the script
#---------------------
# before: <...Mapping ... object1="object1" ... object2="object2" ...
# after:  <...Mapping ... input="@object1" ... output="@object2" ...



s|<\(.*Mapping .*\)object1="../\([^"]*\)"|<\1input="@\2"|g

s|<\(.*Mapping .*\)object1="\([^"@.][^"]*\)"|<\1input="@\2"|g

s|<\(.*Mapping .*\)object1=".."|<\1input="@."|g


s|<\(.*Mapping .*\)object2="../\([^"]*\)"|<\1output="@\2"|g

s|<\(.*Mapping .*\)object2="\([^"@.][^"]*\)"|<\1output="@\2"|g

s|<\(.*Mapping .*\)object2=".."|<\1output="@."|g
