/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_MAPPING_LAPAROSCOPICRIGIDMAPPING_H
#define SOFA_COMPONENT_MAPPING_LAPAROSCOPICRIGIDMAPPING_H
#include "config.h"

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
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;

public:
    Data< defaulttype::Vector3 > pivot;
    Data< defaulttype::Quat > rotation;
protected:
    LaparoscopicRigidMapping()
        : Inherit()
        , pivot(initData(&pivot, defaulttype::Vector3(0,0,0), "pivot","Pivot point position"))
        , rotation(initData(&rotation, defaulttype::Quat(0,0,0,1), "rotation", "TODO-rotation"))
    {
    }

    virtual ~LaparoscopicRigidMapping()
    {
    }
public:

    //void setPivot(const defaulttype::Vector3& val) { this->pivot = val; }
    //void setRotation(const defaulttype::Quat& val) { this->rotation = val; this->rotation.normalize(); }

    void init();

    void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& out, const Data<InVecCoord>& in);

    void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in);

    void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in);

    void draw(const core::visual::VisualParams* vparams);


protected:
    sofa::defaulttype::Quat currentRotation;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_LAPAROSCOPICRIGIDMAPPING_CPP)
extern template class SOFA_GENERAL_RIGID_API LaparoscopicRigidMapping< LaparoscopicRigidTypes, RigidTypes >;
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
