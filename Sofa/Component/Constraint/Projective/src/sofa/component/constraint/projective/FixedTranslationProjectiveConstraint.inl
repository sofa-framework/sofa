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

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/component/constraint/projective/FixedTranslationProjectiveConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/type/vector_algorithm.h>

namespace sofa::component::constraint::projective
{

template< class DataTypes>
FixedTranslationProjectiveConstraint<DataTypes>::FixedTranslationProjectiveConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(nullptr)
    , d_indices(initData(&d_indices, "indices", "Indices of the fixed points") )
    , d_fixAll(initData(&d_fixAll, false, "fixAll", "filter all the DOF to implement a fixed object") )
    , d_drawSize(initData(&d_drawSize, (SReal)0.0, "drawSize", "0 -> point based rendering, >0 -> radius of spheres") )
    , d_coordinates(initData(&d_coordinates, "coordinates", "Coordinates of the fixed points") )
    , l_topology(initLink("topology", "link to the topology container"))
{
    // default to indice 0
    d_indices.beginEdit()->push_back(0);
    d_indices.endEdit();

    f_indices.setParent(&d_indices);
    f_fixAll.setParent(&d_fixAll);
    _drawSize.setParent(&d_drawSize);
    f_coordinates.setParent(&d_coordinates);
}


template <class DataTypes>
FixedTranslationProjectiveConstraint<DataTypes>::~FixedTranslationProjectiveConstraint()
{

}

template <class DataTypes>
void FixedTranslationProjectiveConstraint<DataTypes>::clearIndices()
{
    d_indices.beginEdit()->clear();
    d_indices.endEdit();
}

template <class DataTypes>
void FixedTranslationProjectiveConstraint<DataTypes>::addIndex(Index index)
{
    d_indices.beginEdit()->push_back(index);
    d_indices.endEdit();
}

template <class DataTypes>
void FixedTranslationProjectiveConstraint<DataTypes>::removeIndex(Index index)
{
    sofa::type::removeValue(*d_indices.beginEdit(), index);
    d_indices.endEdit();
}

// -- Constraint interface
template <class DataTypes>
void FixedTranslationProjectiveConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    if (sofa::core::topology::BaseMeshTopology* _topology = l_topology.get())
    {
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

        // Initialize topological changes support
        d_indices.createTopologyHandler(_topology);
        d_coordinates.createTopologyHandler(_topology);
    }
    else
    {
        msg_info() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
    }
}


template<Size N, class T>
static inline void clearPos(defaulttype::RigidDeriv<N,T>& v)
{
    getVCenter(v).clear();
}

template<class T>
static inline void clearPos(type::Vec<6,T>& v)
{
    for (unsigned int i=0; i<3; ++i)
        v[i] = 0;
}

template <class DataTypes> template <class DataDeriv>
void FixedTranslationProjectiveConstraint<DataTypes>::projectResponseT(DataDeriv& dx,
    const std::function<void(DataDeriv&, const unsigned int)>& clear)
{
    if (d_fixAll.getValue())
    {
        for (std::size_t i = 0; i < dx.size(); i++)
        {
            clear(dx, i);
        }
    }
    else
    {
        const SetIndexArray & indices = d_indices.getValue();
        for (const auto index : indices)
        {
            clear(dx, index);
        }
    }
}

template <class DataTypes>
void FixedTranslationProjectiveConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT<VecDeriv>(res.wref(), [](auto& dx, const unsigned int index) {dx[index].clear(); });
}

template <class DataTypes>
void FixedTranslationProjectiveConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& /*vData*/)
{

}

template <class DataTypes>
void FixedTranslationProjectiveConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& /*xData*/)
{

}

template <class DataTypes>
void FixedTranslationProjectiveConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataMatrixDeriv> c = cData;
    projectResponseT<MatrixDeriv>(c.wref(), [](MatrixDeriv& res, const unsigned int index) { res.clearColBlock(index); });
}


template <class DataTypes>
void FixedTranslationProjectiveConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    const SetIndexArray & indices = d_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    std::vector<sofa::type::Vec3> vertices;
    constexpr sofa::type::RGBAColor color(1, 0.5, 0.5, 1);

    if (d_fixAll.getValue() == true)
    {
        for (unsigned i = 0; i < x.size(); i++)
        {
            sofa::type::Vec3 v;
            const typename DataTypes::CPos& cpos = DataTypes::getCPos(x[i]);
            for(Size j=0 ; j<cpos.size() && j<3; j++)
                v[j] = cpos[j];

            vertices.push_back(v);
        }
    }
    else
    {
        for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            sofa::type::Vec3 v;
            const typename DataTypes::CPos& cpos = DataTypes::getCPos(x[*it]);
            for(Size j=0 ; j<cpos.size() && j<3; j++)
                v[j] = cpos[j];

            vertices.push_back(v);
        }
    }
    vparams->drawTool()->drawPoints(vertices, 10, color);


}

} // namespace sofa::component::constraint::projective
