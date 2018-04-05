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
#ifndef INTERSECTOR_H
#define INTERSECTOR_H
#include "config.h"

#include <sofa/defaulttype/Vec.h>

namespace sofa{
namespace component{
namespace collision{


template <class Real>
class Intersector{
public:
    typedef sofa::defaulttype::Vec<3,Real> Vec3;

    inline const Vec3 & separatingAxis()const{
        return _sep_axis;
    }

    inline const Vec3 & pointOnFirst()const{
        return _pt_on_first;
    }

    inline const Vec3 & pointOnSecond()const{
        return _pt_on_second;
    }

    inline bool colliding()const{
        return _is_colliding;
    }
protected:
    Vec3 _sep_axis;
    Vec3 _pt_on_first;
    Vec3 _pt_on_second;
    Real mContactTime;
    bool _is_colliding;
};

}
}
}
#endif // INTERSECTOR_H
