/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/mapping/linear/config.h>

#include <sofa/core/Mapping.h>

#include <sofa/defaulttype/VecTypes.h>

#include <sofa/component/mapping/linear/Mesh2PointTopologicalMapping.h>

namespace sofa::core::topology { class BaseMeshTopology; }

namespace sofa::component::mapping::linear
{

template <class TIn, class TOut>
class Mesh2PointMechanicalMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(Mesh2PointMechanicalMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;

    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename InCoord::value_type Real;

    using Index = sofa::Index;

protected:
    Mesh2PointMechanicalMapping(core::State<In>* from = nullptr, core::State<Out>* to = nullptr);

    virtual ~Mesh2PointMechanicalMapping();

public:

    void init() override;

    void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& out, const Data<InVecCoord>& in) override;

    void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in) override;

    void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in) override;

    void applyJT(const core::ConstraintParams *cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in) override;

protected:
    Mesh2PointTopologicalMapping* topoMap;
    core::topology::BaseMeshTopology* inputTopo;
    core::topology::BaseMeshTopology* outputTopo;
};



#if !defined(SOFA_COMPONENT_MAPPING_MESH2POINTMECHANICALMAPPING_CPP)
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API Mesh2PointMechanicalMapping< defaulttype::Vec3Types, defaulttype::Vec3Types >;
#endif

} //namespace sofa::component::mapping::linear
