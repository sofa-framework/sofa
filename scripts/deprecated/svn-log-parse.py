# extract imformations of a svn log (in xml format)
# uses only the first logentry
# usage: svn-log-parse.py filename [revision|author|msg|date]
# use "svn log -l 1 --xml > log.xml" to have the infos about the last checkouted revision


import os
import os.path
import sys
from xml.dom import minidom

if len(sys.argv) == 3:
    in_filename = str(sys.argv[1])
    command = str(sys.argv[2])
    if os.path.isfile (in_filename):
        xmldoc = minidom.parse(in_filename)
        #for node in xmldoc.getElementsByTagName('logentry'):
        node=xmldoc.getElementsByTagName('logentry')[0]
        if node:
            revision = str(node.getAttribute('revision'))
            msg=''
            author=''
            date=''
            for childNode in node.childNodes:
                if childNode.nodeType==minidom.Node.ELEMENT_NODE:
                    if childNode.nodeName=='msg':
                        msg=str(childNode.firstChild.nodeValue)[:200]
                    if childNode.nodeName=='author':
                        author=str(childNode.firstChild.nodeValue)
                    if childNode.nodeName=='date':
                        date=str(childNode.firstChild.nodeValue)[:16].replace('T',' ')


            #print 'revision='+revision
            #print 'author='+author
            #print 'date='+date
            #print 'msg='+msg
            if (command=='revision'): print revision
            if (command=='msg'): print msg
            if (command=='author'): print author
            if (command=='date'): print date
