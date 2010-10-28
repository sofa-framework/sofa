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

#include <sofa/core/Mapping.h>

#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

template <class TIn, class TOut>
class LaparoscopicRigidMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(LaparoscopicRigidMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;

public:
    Data< Vector3 > pivot;
    Data< Quat > rotation;

    LaparoscopicRigidMapping(core::State<In>* from, core::State<Out>* to)
        : Inherit(from, to)
        , pivot(initData(&pivot, Vector3(0,0,0), "pivot","Pivot point position"))
        , rotation(initData(&rotation, Quat(0,0,0,1), "rotation", "TODO-rotation"))
    {
    }

    virtual ~LaparoscopicRigidMapping()
    {
    }


    //void setPivot(const defaulttype::Vector3& val) { this->pivot = val; }
    //void setRotation(const defaulttype::Quat& val) { this->rotation = val; this->rotation.normalize(); }

    void init();

    void apply(Data<OutVecCoord>& out, const Data<InVecCoord>& in, const core::MechanicalParams *mparams);

    void applyJ(Data<OutVecDeriv>& out, const Data<InVecDeriv>& in, const core::MechanicalParams *mparams);

    void applyJT(Data<InVecDeriv>& out, const Data<OutVecDeriv>& in, const core::MechanicalParams *mparams);

    void draw();

protected:
    Quat currentRotation;
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_LAPAROSCOPICRIGIDMAPPING_CPP)
#pragma warning(disable : 4231)
extern template class LaparoscopicRigidMapping< LaparoscopicRigidTypes, RigidTypes >;
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
