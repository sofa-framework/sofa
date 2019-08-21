#!/usr/bin/python

# This tool parses all C++ files and replaces pathes in #include directives following matches described in an input file.
#
# Arguments
# -i <pathes_file>: a file with matching between pathes
#                   Each line must be of the form:
#
#                           <path1> : <path2>
#
#-d <directory_to_parse>: the path of the parent directory that will be browsed recursively for files to process

import sys, getopt
import string, re
import fileinput
from os import walk

def usage():
    print 'usage: python convert_include_path.py -i <pathes_file> -d <directory_to_parse>'


def main(argv):
    matchfile, path = readOpt(argv)
    print 'Parsing directory \'' + path + '\' with matches from file \'' + matchfile + '\'\n'

    matches = loadMatches(matchfile)

    processedExtensions = { 'h', 'inl', 'cpp' }
    for (dirpath, subdirnames, files) in walk(path):
        for f in files:
            for ext in processedExtensions:
                if f.endswith(ext):
                    processFile(dirpath + '/' + f, matches)

# Read command line arguments
def readOpt(argv):
    if len(argv) != 4:
        usage()
        sys.exit(1)

    try:
        opts, args = getopt.getopt(argv, "hi: d:")
    except getopt.GetoptError:
        print 'Error while reading the arguments'
        sys.exit(1)

    matchfile = ''
    path = ''    
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit(0)
        elif opt == '-i':
            matchfile = arg
        elif opt == '-d':
            path = arg

    return matchfile, path

# Load the file containing matchings between pathes
def loadMatches(matchfile):
    f = open(matchfile, 'r')

    matches = {} 

    for line in f:
        line = line.rstrip('\n');
        pathes = string.split(line, ' : ')
        if len(pathes) == 2:
            matches[pathes[0]] = pathes[1]

    f.close()

    return matches

# Modify the pathes of the file #include directives if necessary
def processFile(file, matches):
    print '*** Processing file \'' + file + '\'\n'

    filecontent = []
    with open(file, 'r') as f:
        filecontent = f.readlines();
        f.close()

    pattern = '#include [<"](.*)[>"]'
    i = 0
    for line in filecontent:
        match = re.search(pattern, line)
        if match and match.groups() > 0:
            includePath = match.groups()[0]
            if includePath in matches:
                newline = line[:match.start(1)] + matches[includePath] + line[match.end(1):]
                filecontent[i] = newline
                print '      ' + line + '   => ' +  newline
        i += 1

    with open(file, 'w') as f:
        f.writelines(filecontent);
        f.close()

    return


if __name__ == '__main__':
    main(sys.argv[1:])
