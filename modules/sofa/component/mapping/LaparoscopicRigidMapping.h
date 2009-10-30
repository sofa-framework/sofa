/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_LAPAROSCOPICRIGIDMAPPING_H
#define SOFA_COMPONENT_MAPPING_LAPAROSCOPICRIGIDMAPPING_H


#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/component/topology/PointSubset.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa
{

namespace component
{

namespace mapping
{

template <class BasicMapping>
class LaparoscopicRigidMapping : public BasicMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(LaparoscopicRigidMapping,BasicMapping), BasicMapping);
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::VecCoord VecCoord;
    typedef typename In::VecCoord InVecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    //typedef typename Coord::value_type Real;

public:
    Data<defaulttype::Vector3> pivot;
    Data<defaulttype::Quat> rotation;

    LaparoscopicRigidMapping(In* from, Out* to)
        : Inherit(from, to)
        , pivot(initData(&pivot, defaulttype::Vector3(0,0,0), "pivot","Pivot point position"))
        , rotation(initData(&rotation, defaulttype::Quat(0,0,0,1), "rotation", "TODO-rotation"))
    {
    }

    virtual ~LaparoscopicRigidMapping()
    {
    }


    //void setPivot(const defaulttype::Vector3& val) { this->pivot = val; }
    //void setRotation(const defaulttype::Quat& val) { this->rotation = val; this->rotation.normalize(); }

    void init();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    void draw();

protected:
    defaulttype::Quat currentRotation;
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
