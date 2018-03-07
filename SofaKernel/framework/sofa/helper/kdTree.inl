/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_KDTREE_INL
#define SOFA_HELPER_KDTREE_INL

#include "kdTree.h"

#include <map>
#include <limits>
#include <iterator>
#include <cmath>

namespace sofa
{
namespace helper
{

template<class Coord>
void kdTree<Coord>::build(const VecCoord& positions)
{
    const unsigned int nbp=positions.size();
    UIlist list;   for(unsigned int i=0; i<nbp; i++) list.push_back(i);
    tree.resize(nbp);
    firstNode=build(list,(unsigned char)0, positions);
}

template<class Coord>
void kdTree<Coord>::build(const VecCoord& positions, const vector<unsigned int> &ROI)
{
    const size_t nbp = ROI.size();
    UIlist list;
    for(size_t i=0; i<nbp; i++)
        list.push_back(ROI[i]);
    tree.resize(positions.size());
    firstNode=build(list,(unsigned char)0, positions);
}

template<class Coord>
void kdTree<Coord>::print(const unsigned int index)
{
    dmsg_info("KDTree") << index<<"["<<(int)tree[index].splitdir<<"] "<<tree[index].left<<" "<<tree[index].right ;
    if(tree[index].left!=index) print(tree[index].left);
    if(tree[index].right!=index) print(tree[index].right);
}

template<class Coord>
unsigned int kdTree<Coord>::build(UIlist &list, unsigned char direction, const VecCoord& positions)
{
    // detect leaf
    if(list.size()==(unsigned int)1)
    {
        unsigned int index=list.front(); list.pop_front();
        tree[index].left=tree[index].right=index;
        tree[index].splitdir=direction;
        return index;
    }
    // ordered list of 1D coord
    distanceSet q;
    UIlist::iterator it;
    for(it=list.begin(); it!=list.end(); it++) q.insert( distanceToPoint(positions[*it][direction],*it));
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
    if(newlist.size()!=0) tree[index].left=build(newlist,newdirection,positions);
    if(list.size()!=0) tree[index].right=build(list,newdirection,positions);
    // return child index to parent
    return index;
}


template<class Coord>
void kdTree<Coord>::closest(distanceSet &cl,const Coord &x, const unsigned int &currentnode, const VecCoord& positions, unsigned N) const
// [zhang94] algorithm
{
    Real Dmax;
    distanceSetIt it=cl.end();
    if(cl.size()==N) { it--; Dmax=it->first; } else Dmax=std::numeric_limits<Real>::max();
    unsigned int splitdir=tree[currentnode].splitdir;
    Coord pos=positions[currentnode];
    Real c1=x[splitdir],c2=pos[splitdir];
    if(std::abs(c1-c2)<=Dmax)
    {
        Real d=(x-pos).norm();
        if(d<Dmax)
        {
            Dmax=d;
            cl.insert(distanceToPoint(d,currentnode));
            if(cl.size()>N) {it=cl.end(); it--; cl.erase(it);}
        }
    }
    if(tree[currentnode].left!=currentnode)     if(c1-Dmax<c2)  closest(cl,x,tree[currentnode].left,positions,N);
    if(tree[currentnode].right!=currentnode)    if(c2-Dmax<c1)  closest(cl,x,tree[currentnode].right,positions,N);
}


// slightly improved version of the above, for one point
template<class Coord>
void kdTree<Coord>::closest(distanceToPoint &cl,const Coord &x, const unsigned int &currentnode, const VecCoord& positions) const
{
    Real Dmax=cl.first;
    unsigned int splitdir=tree[currentnode].splitdir;
    Coord pos=positions[currentnode];
    Real c1=x[splitdir],c2=pos[splitdir];
    if(std::abs(c1-c2)<=Dmax)
    {
        Real d=(x-pos).norm();
        if(d<Dmax)
        {
            Dmax=d;
            cl.first=d;
            cl.second=currentnode;
        }
    }
    if(tree[currentnode].left!=currentnode)     if(c1-Dmax<c2)  closest(cl,x,tree[currentnode].left,positions);
    if(tree[currentnode].right!=currentnode)    if(c2-Dmax<c1)  closest(cl,x,tree[currentnode].right,positions);
}


template<class Coord>
void kdTree<Coord>::getNClosest(distanceSet &cl, const Coord &x, const VecCoord& positions, const unsigned int n) const
{
    cl.clear();
    closest(cl,x,firstNode,positions,n);
}

template<class Coord>
unsigned int kdTree<Coord>::getClosest(const Coord &x, const VecCoord& positions) const
{
    distanceToPoint cl(std::numeric_limits<Real>::max(),firstNode);
    closest(cl,x,firstNode,positions);
    return cl.second;
}

template<class Coord>
bool kdTree<Coord>::getNClosestCached(distanceSet &cl,  distanceToPoint &cacheThresh_max, distanceToPoint &cacheThresh_min, Coord &previous_x, const Coord &x, const VecCoord& positions, const unsigned int n) const
{
    Real dx=(previous_x-x).norm();
    if(dx>=cacheThresh_max.first || cl.size()<2)
    {
        getNClosest(cl,x,positions,n);
        distanceSetIt it0=cl.begin(), it1=it0; it1++;
        typename distanceSet::reverse_iterator itn=cl.rbegin();
        cacheThresh_max.first=((itn->first)-(it0->first))*(Real)0.5; // half distance between first and last closest points
        cacheThresh_max.second=itn->second;
        cacheThresh_min.first=((it1->first)-(it0->first))*(Real)0.5; // half distance between first and second closest points
        cacheThresh_min.second=it0->second;
        previous_x=x;
        return false;
    }
    else if(dx>=cacheThresh_min.first) // in the cache -> update N-1 distances
    {
        distanceSet newset;
        for(distanceSetIt it=cl.begin(); it!=cl.end(); it++)
            if(it->second!=cacheThresh_max.second)
                newset.insert(distanceToPoint((positions[it->second]-x).norm(),it->second));
            else
                newset.insert(distanceToPoint(std::numeric_limits<Real>::max(),it->second));
        cl.swap(newset);
        return true;
    }
    else // still the same closest point
    {
        distanceSet newset;
        for(distanceSetIt it=cl.begin(); it!=cl.end(); it++)
            if(it->second==cacheThresh_min.second)
                newset.insert(distanceToPoint((positions[it->second]-x).norm(),it->second));
            else
                newset.insert(distanceToPoint(std::numeric_limits<Real>::max(),it->second));
        cl.swap(newset);
        return true;
    }
}



}
}

#endif
