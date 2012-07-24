/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_KDTREE_INL
#define SOFA_HELPER_KDTREE_INL

#include "kdTree.h"

#include <map>
#include <limits>
#include <set>
#include <iterator>

namespace sofa
{
namespace helper
{

template<class Coord>
void kdTree<Coord>::build(const VecCoord& p)
{
    position=&p;
    unsigned int nbp=position->size();
    UIlist list;   for(unsigned int i=0; i<nbp; i++) list.push_back(i);
    tree.resize(nbp);
    firstNode=build(list,(unsigned char)0);
}


template<class Coord>
unsigned int kdTree<Coord>::build(UIlist &list, unsigned char direction)
{
    // detect leaf
    if(list.size()==(unsigned int)1)
    {
        unsigned int index=list.front(); list.pop_front();
        tree[index].left=tree[index].right=index;
        tree[index].splitdir=dim;
        return index;
    }
    // ordered list of 1D coord
    distanceSet q;
    UIlist::iterator it;
    for(it=list.begin(); it!=list.end(); it++) q.insert( distanceToPoint((*position)[*it][direction],*it));
    it=list.begin();    for(distanceSetIt sit=q.begin(); sit!=q.end(); sit++) {*it=sit->second; it++;}
    // split list in 2
    it=list.begin(); advance (it,list.size()/2);
    UIlist newlist;
    newlist.splice(newlist.begin(),list,list.begin(),it);
    // add node
    unsigned int index=list.front(); list.pop_front();
    tree[index].splitdir=direction;
    tree[index].left=tree[index].right=index;
    // split children recursively
    unsigned char newdirection=direction+1; if(newdirection==dim) newdirection=0;
    if(newlist.size()!=0) tree[index].left=build(newlist,newdirection);
    if(list.size()!=0) tree[index].right=build(list,newdirection);
    // return child index to parent
    return index;
}


template<class Coord>
void kdTree<Coord>::closest(distanceSet &cl,const Coord &x, const unsigned int &currentnode)
// [zhang94] algorithm
{
    Real Dmax=std::numeric_limits<Real>::max();
    distanceSetIt it=cl.end();
    if(cl.size()!=0) { it--; Dmax=it->first; }
    Real c1=x[tree[currentnode].splitdir],c2=(*position)[currentnode][tree[currentnode].splitdir];
    if(abs(c1-c2)<=Dmax)
    {
        Real d=(x-(*position)[currentnode]).norm();
        if(d<Dmax)
        {
            cl.insert(distanceToPoint(d,currentnode));
            if(cl.size()>N) {it=cl.end(); it--; cl.erase(it);}
        }
    }
    if(tree[currentnode].splitdir==dim) return; // leaf
    if(tree[currentnode].left!=currentnode)     if(c1-Dmax<c2)  closest(cl,x,tree[currentnode].left);
    if(tree[currentnode].right!=currentnode)    if(c2-Dmax<c1)  closest(cl,x,tree[currentnode].right);
}


// slightly improved version of the above, for one point
template<class Coord>
void kdTree<Coord>::closest(distanceToPoint &cl,const Coord &x, const unsigned int &currentnode)
{
    Real Dmax=cl.first;
    Real c1=x[tree[currentnode].splitdir],c2=(*position)[currentnode][tree[currentnode].splitdir];
    if(abs(c1-c2)<=Dmax)
    {
        Real d=(x-(*position)[currentnode]).norm();
        if(d<Dmax)
        {
            cl.first=d;
            cl.second=currentnode;
        }
    }
    if(tree[currentnode].splitdir==dim) return; // leaf
    if(tree[currentnode].left!=currentnode)     if(c1-Dmax<c2)  closest(cl,x,tree[currentnode].left);
    if(tree[currentnode].right!=currentnode)    if(c2-Dmax<c1)  closest(cl,x,tree[currentnode].right);
}


template<class Coord>
void kdTree<Coord>::getNClosest(distanceSet &cl, const Coord &x, const unsigned int n)
{
    N=n;
    cl.clear();
    closest(cl,x,firstNode);
}

template<class Coord>
unsigned int kdTree<Coord>::getClosest(const Coord &x)
{
    distanceToPoint cl(std::numeric_limits<Real>::max(),firstNode);
    closest(cl,x,firstNode);
    return cl.second;
}

template<class Coord>
void kdTree<Coord>::updateCachedDistances(distanceSet &cl, const Coord &x)
{
    distanceSet newset;
    distanceSetIt it,itback=cl.end(); itback--;
    for(it=cl.begin(); it!=itback; it++)        newset.insert(distanceToPoint(((*position)[it->second]-x).norm(),it->second));
    newset.insert(distanceToPoint(std::numeric_limits<Real>::max(),itback->second));
    cl.swap(newset);
}



}
}

#endif
