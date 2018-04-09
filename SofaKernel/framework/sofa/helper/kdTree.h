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
#ifndef SOFA_HELPER_KDTREE_H
#define SOFA_HELPER_KDTREE_H

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>
#include <set>
#include <list>


namespace sofa
{

namespace helper
{

/**
*  This class implements classical kd tree for nearest neighbors search
*  - the tree is rebuild from points by calling build(p)
*  - N nearest points from point x (in terms of euclidean distance) are retrieved with getNClosest(distance/index_List , x , N)
*  - Caching may be used to speed up retrieval: if dx< (d(n)-d(0))/2, then the closest point is in the n-1 cached points (updateCachedDistances is used to update the n-1 distances)
*  see for instance: [zhang92] report and [simon96] thesis for more details
*
*  @author Benjamin Gilles
**/


template<class Coord>
class kdTree
{
public:
    typedef typename Coord::value_type Real;
    enum { dim=Coord::total_size };
    typedef vector<Coord> VecCoord;

    typedef std::pair<Real,unsigned int> distanceToPoint;
    typedef std::set<distanceToPoint> distanceSet;
    typedef typename distanceSet::iterator distanceSetIt;
    typedef std::list<unsigned int> UIlist;

    typedef struct
    {
        unsigned char splitdir; // 0/1/2 -> x/y/z
        unsigned int left;	// index of the left node
        unsigned int right; // index of the right node
    } TREENODE;

    bool isEmpty() const {return tree.size()==0;}
    void build(const VecCoord& positions);       ///< update tree (to be used whenever positions have changed)
    void build(const VecCoord& positions, const vector<unsigned int> &ROI);       ///< update tree based on positions subset (to be used whenever points p have changed)
    void getNClosest(distanceSet &cl, const Coord &x, const VecCoord& positions, const unsigned int n) const;  ///< get an ordered set of n distance/index pairs between positions and x
    unsigned int getClosest(const Coord &x, const VecCoord& positions) const; ///< get the index of the closest point between positions and x
    bool getNClosestCached(distanceSet &cl, distanceToPoint &cacheThresh_max, distanceToPoint &cacheThresh_min, Coord &previous_x, const Coord &x, const VecCoord& positions, const unsigned int n) const;  ///< use distance caching to accelerate closest point computation when positions are fixed (see simon96 thesis)


    /// @name To be Data-zable
    /// @{
        inline friend std::ostream& operator<< ( std::ostream& os, const kdTree<Coord>& ) {return os;}
        inline friend std::istream& operator>> ( std::istream& is, kdTree<Coord>& ) {return is;}
    /// @}

protected :
    void print(const unsigned int index);

    vector< TREENODE > tree; unsigned int firstNode;

    unsigned int build(UIlist &list, unsigned char direction, const VecCoord& positions); // recursive function to build the kdtree
    void closest(distanceSet &cl, const Coord &x, const unsigned int &currentnode, const VecCoord& positions, unsigned N) const;     // recursive function to get closest points
    void closest(distanceToPoint &cl,const Coord &x, const unsigned int &currentnode, const VecCoord& positions) const;  // recursive function to get closest point
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_HELPER_KDTREE_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_HELPER_API kdTree<sofa::defaulttype::Vec2d>;
extern template class SOFA_HELPER_API kdTree<sofa::defaulttype::Vec3d>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_HELPER_API kdTree<sofa::defaulttype::Vec2f>;
extern template class SOFA_HELPER_API kdTree<sofa::defaulttype::Vec3f>;
#endif
#endif

}
}

#endif
