/***************************************************************************
                          TargetList.cpp  -  description
                             -------------------
    begin                : Sun Mar 30 2003
    copyright            : (C) 2003 by Emmanuel Promayon
    email                : Emmanuel.Promayon@imag.fr
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "TargetList.h"
#include <sstream>

// --------------- constructor ---------------
TargetList::TargetList() {}

TargetList::TargetList(const std::string l) {
    add(l);
}

TargetList::TargetList(const TargetList & t) {
    add(t.toString()); // not very efficient but particularly easy todo!
}

// --------------- add ---------------
void TargetList::add(const unsigned int index) {
    indexedTargetList.push_back(index);
}

// --------------- add ---------------
// from a list using the list syntax (eg: 1-4,7,9-11) stored in s
// insert all the corresponding target to currentL target list (above ex insert targets : 1 2 3 4 7 9 10 11)
#define isDigit(c)  ((c)>='0' && (c)<='9')

void TargetList::add(const std::string s) {
    int begin, end, len, i,j;
    
    // inits
    len = s.size();
    begin = end = -1;
    i = 0;

    // examine all the string
    while (i<len) {
        if (isDigit(s[i])) {
            // the current char is a digit
            if (begin == -1) {
                // begin is to be set now
                begin = s[i++] - '0';
                // Loop to construct the target nr
                while (i<len && isDigit(s[i]))
                    begin = begin*10 + s[i++] - '0';
                // here begin is equal to the target index to begin with
            } else {
                // it is end's turn (meaning we already get the target index for the beginning)
                end = s[i++] - '0';
                while (i<len && isDigit(s[i]))
                    end = end*10 + s[i++] - '0';
                // here end is equal to the target index to end with
            }
        }

        // here something (that was not a digit) was encoutered
        while (i<len && !isDigit(s[i])) {
            if (s[i]==',') {
                // if it was a comma, save from begin to end
                if (end==-1) // end was not find, only target index 'begin'
                    add(begin);
                else {
                    // target index from begin..save are to be inserted
                    for (j=begin;j<=end;j++)
                        add(j);
                }
                // init to start again
                begin=end=-1;
            }
            else {
                if (s[i]!='-') {
                    // this is not a comma, read till the end or the next comma
                    // because this is a new component name
                    std::string name;
                    while (i<len) {
                        name = "";
                        while (i<len && s[i]!=',') {
                            name += s[i];
                            i++;
                        }
                        namedTargetList.push_back(name);
                        i++;
                    }
                }
            }
            // if it was something else than a ","... well, do nothing
            i++;
        }
    }

    // Here the target list ended, see if there was not something in begin and end
    if (begin != -1) { // smth to save
        if (end==-1) // end was not find
            add(begin);
        else {
            // regions from begin..save are to be saved
            for (j=begin;j<=end;j++)
                add(j);
        }
    }

}

// --------------- getNumberOfTargets ---------------
unsigned int TargetList::getNumberOfTargets() const {
    if (indexedTargets()) {
        return indexedTargetList.size();
    } else {
        return namedTargetList.size();
    }
}

// --------------- getIndexedTarget ---------------
int TargetList::getIndexedTarget(const unsigned int targetIndex) const {
    if (targetIndex < indexedTargetList.size())
        return indexedTargetList[targetIndex];
    else
        return -1;
}

// --------------- getNamedTarget ---------------
std::string TargetList::getNamedTarget(const unsigned int targetIndexInList) const {
    if (targetIndexInList < namedTargetList.size())
        return namedTargetList[targetIndexInList];
    else
        return "";
}

// --------------- clear ---------------
void TargetList::clear() {
    indexedTargetList.clear();
    namedTargetList.clear();
}

// --------------- toString ---------------
std::string TargetList::toString() const {
    std::stringstream s(std::stringstream::out);

    if (indexedTargets()) {
        unsigned int i=0;
        unsigned int beginId, beginIndex;
        unsigned int endId, endIndex;
        while (i<indexedTargetList.size()) {
            // init begin of the possible interval
            beginIndex = i;
            beginId = indexedTargetList[beginIndex];

            // find an interval
            do {
                endIndex = i;
                endId = indexedTargetList[endIndex];
                i++;
            } while (i<indexedTargetList.size() && (indexedTargetList[i]==(endId+1)));

            // add a comma for separation
            if (beginIndex!=0)
                s << ",";

            // if there is not interval
            if (endIndex==beginIndex) {
                s << beginId;
            } else {
                s << beginId << "-" << endId;
            }
        }
    } else {
        // targets are named
        for (unsigned int i=0;i<namedTargetList.size();i++) {
            s << namedTargetList[i];
            if (i!=namedTargetList.size()-1)
                s << ",";
        }
    }

    return s.str();
}

// --------------- toAnsys ---------------
std::string TargetList::toAnsys() const {

    std::stringstream s(std::stringstream::out);
    s << "! --- selection of target list: " << toString() << std::endl;

    if (indexedTargets()) {
        unsigned int i=0;
        unsigned int beginId, beginIndex;
        unsigned int endId, endIndex;

        while (i<indexedTargetList.size()) {
            // init begin of the possible interval
            beginIndex = i;
            beginId = indexedTargetList[beginIndex];

            // find an interval
            do {
                endIndex = i;
                endId = indexedTargetList[endIndex];
                i++;
            } while (i<indexedTargetList.size() && (indexedTargetList[i]==(endId+1)));

            // add a comma for separation
            if (beginIndex!=0)
                // add to the selection
                s << "NSEL, A,,, ";
            else
                // define a new selection
                s << "NSEL, S,,, ";

            // if there is not interval
            if (endIndex==beginIndex) {
                s << beginId;
            } else {
                s << beginId << "," << endId << ",1" << std::endl;
            }
        }
    }

    return s.str();
}

// --------------- toAnsys ---------------
bool TargetList::indexedTargets() const {
    return (indexedTargetList.size()>0);
}

// --------------- isIn ---------------
bool TargetList::isIn(unsigned int id) const {
  if (indexedTargetList.size()==0 && namedTargetList.size()==1 && namedTargetList[0]=="*")
    return true;

  bool found=false;
  unsigned int i=0;
  while(i<indexedTargetList.size() && !found) {
    found = (indexedTargetList[i]==id);
    i++;
  }
  return found;
}

