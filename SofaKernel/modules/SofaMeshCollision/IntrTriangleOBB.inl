/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaMeshCollision/IntrTriangleOBB.h>

namespace sofa{
namespace component{
namespace collision{
//----------------------------------------------------------------------------

template <typename TDataTypes,typename TDataTypes2>
TIntrTriangleOBB<TDataTypes,TDataTypes2>::TIntrTriangleOBB (const IntrTri& tri, const Box & box): _tri(&tri),mBox(&box){
    _is_colliding = false;
}

template <typename TDataTypes,typename TDataTypes2>
bool TIntrTriangleOBB<TDataTypes,TDataTypes2>::Find (Real dmax,int tri_flg)
{
    bool config_modified;

    mContactTime = -std::numeric_limits<Real>::max();

    //int tri_flg = _tri->getCollisionModel()->getTriangleFlags(_tri->getIndex());
    //int tri_flg = _tri->getCollisionModel()->getTriangleFlags(_tri->getIndex());
    //((TTriangleModel<TDataTypes> *)(_tri->getCollisionModel()))->bidon();

    int side = IntrConfiguration<Real>::NONE;
    IntrConfiguration<Real> triContact, boxContact;

    // Test tri-normal.
    Vec3 edge[3] =
    {
        _tri->p2() - _tri->p1(),
        _tri->p3() - _tri->p2(),
        _tri->p1() - _tri->p3()
    };

    if(!IntrAxis<IntrTri,Box>::Find(_tri->n(), *_tri, *mBox, dmax,
        mContactTime, side, triContact, boxContact,config_modified))
        return false;

    if(config_modified)
        _sep_axis = _tri->n();

    Vec3 axis;
    int coplanar = -1; // triangle coplanar to none of the box normals
    int i0;
    for (i0 = 0; i0 < 3; ++i0)
    {
        axis = mBox->axis(i0);
        if(!IntrAxis<IntrTri,Box>::Find(axis, *_tri, *mBox,
            dmax, mContactTime,side, triContact, boxContact,config_modified))
            return false;

        if(config_modified)
            _sep_axis = axis;

        // Test if axis is parallel to triangle normal.  The test is:
        // sin(Angle(normal,axis)) < epsilon
        Real NdA = (_tri->n()) * axis;
//        Real NdN = (_tri->n()).SquaredLength();
//        Real AdA = axis.SquaredLength();
//        if (NdA < IntrUtil<Real>::ZERO_TOLERANCE())
//        {
//            coplanar = i0;
//        }        Real NdA = triNorm.Dot(axis);
        Real sn = fabs((Real)1 -NdA*NdA);
        if (sn < IntrUtil<Real>::ZERO_TOLERANCE())
        {
            coplanar = i0;
        }



    }

    if (coplanar == -1)
    {
        // Test triedges cross boxfaces.
        for (i0 = 0; i0 < 3; ++i0)
        {

            if(tri_flg&TriangleModel::FLAG_E12){
                axis = edge[0].cross(mBox->axis(i0));
                if(axis.norm2() > IntrUtil<Real>::ZERO_TOLERANCE()){
                    IntrUtil<Real>::normalize(axis);

                    if(!IntrAxis<IntrTri,Box>::Find(axis, *_tri, *mBox,
                        dmax, mContactTime,side, triContact, boxContact,config_modified))
                        return false;

                    if(config_modified)
                        _sep_axis = axis;
                }
            }

            if(tri_flg&TriangleModel::FLAG_E23){
                axis = edge[1].cross(mBox->axis(i0));
                if(axis.norm2() > IntrUtil<Real>::ZERO_TOLERANCE()){
                    IntrUtil<Real>::normalize(axis);

                    if(!IntrAxis<IntrTri,Box>::Find(axis, *_tri, *mBox,
                        dmax, mContactTime,side, triContact, boxContact,config_modified))
                        return false;

                    if(config_modified)
                        _sep_axis = axis;
                }
            }

            if(tri_flg&TriangleModel::FLAG_E31){
                axis = edge[2].cross(mBox->axis(i0));
                if(axis.norm2() > IntrUtil<Real>::ZERO_TOLERANCE()){
                    IntrUtil<Real>::normalize(axis);

                    if(!IntrAxis<IntrTri,Box>::Find(axis, *_tri, *mBox,
                        dmax, mContactTime,side, triContact, boxContact,config_modified))
                        return false;

                    if(config_modified)
                        _sep_axis = axis;
                }
            }
        }
    }
      else
        //if()
    {
        // Test triedges cross coplanar box axis.
        if(tri_flg&TriangleModel::FLAG_E12){
            axis = edge[0].cross(_tri->n());
            IntrUtil<Real>::normalize(axis);

            if(!IntrAxis<IntrTri,Box>::Find(axis, *_tri, *mBox,
                dmax, mContactTime,side, triContact, boxContact,config_modified))
                return false;

            if(config_modified)
                _sep_axis = axis;
        }

        if(tri_flg&TriangleModel::FLAG_E23){
            axis = edge[1].cross(_tri->n());
            IntrUtil<Real>::normalize(axis);

            if(!IntrAxis<IntrTri,Box>::Find(axis, *_tri, *mBox,
                dmax, mContactTime,side, triContact, boxContact,config_modified))
                return false;

            if(config_modified)
                _sep_axis = axis;
        }

        if(tri_flg&TriangleModel::FLAG_E31){
            axis = edge[2].cross(_tri->n());
            IntrUtil<Real>::normalize(axis);

            if(!IntrAxis<IntrTri,Box>::Find(axis, *_tri, *mBox,
                dmax, mContactTime,side, triContact, boxContact,config_modified))
                return false;

            if(config_modified)
                _sep_axis = axis;
        }
    }

    if (side == IntrConfiguration<Real>::NONE)
    {
        return false;
    }


    FindContactSet<IntrTri,Box>(*_tri, *mBox, _sep_axis,side, triContact, boxContact,
         mContactTime, _pt_on_first,_pt_on_second);

    if((!(tri_flg&TriangleModel::FLAG_P1)) && IntrUtil<Real>::equal(_pt_on_first,_tri->p(0)))
        return false;
    else if((!(tri_flg&TriangleModel::FLAG_P2)) && IntrUtil<Real>::equal(_pt_on_first,_tri->p(1)))
        return false;
    else if((!(tri_flg&TriangleModel::FLAG_P3)) && IntrUtil<Real>::equal(_pt_on_first,_tri->p(2)))
        return false;

    if(side == IntrConfiguration<Real>::LEFT)
        _sep_axis *= -1.0;

//    assert(mBox->onSurface(_pt_on_second));

    return true;
}


template <typename TDataTypes,typename TDataTypes2>
inline bool TIntrTriangleOBB<TDataTypes,TDataTypes2>::Find(Real tmax){return Find(tmax,_tri->flags());}

}
}
}
