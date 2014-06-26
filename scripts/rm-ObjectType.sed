#!/bin/sed -f 


#---------------------
# Pattern of the script
#---------------------
# before: <Object ... type="MyComponent" ...
# after:  <MyComponent ... ...



s/<Object\(.*\) type="\([^"]*\)"/<\2\1/g