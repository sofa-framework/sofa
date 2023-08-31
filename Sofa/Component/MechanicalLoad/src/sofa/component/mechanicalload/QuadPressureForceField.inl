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

#include <sofa/component/mechanicalload/QuadPressureForceField.h>
#include <sofa/core/topology/TopologySubsetData.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <vector>

namespace sofa::component::mechanicalload
{


template <class DataTypes> QuadPressureForceField<DataTypes>::~QuadPressureForceField()
{
}

template <class DataTypes>
QuadPressureForceField<DataTypes>::QuadPressureForceField()
    : pressure(initData(&pressure, "pressure", "Pressure force per unit area"))
    , quadList(initData(&quadList,"quadList", "Indices of quads separated with commas where a pressure is applied"))
    , normal(initData(&normal,"normal", "Normal direction for the plane selection of quads"))
    , dmin(initData(&dmin,(Real)0.0, "dmin", "Minimum distance from the origin along the normal direction"))
    , dmax(initData(&dmax,(Real)0.0, "dmax", "Maximum distance from the origin along the normal direction"))
    , p_showForces(initData(&p_showForces, (bool)false, "showForces", "draw quads which have a given pressure"))
    , l_topology(initLink("topology", "link to the topology container"))
    , quadPressureMap(initData(&quadPressureMap, "quadPressureMap", "map between edge indices and their pressure"))
    , m_topology(nullptr)
{
}

template <class DataTypes>
void QuadPressureForceField<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (m_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (dmin.getValue()!=dmax.getValue())
    {
        selectQuadsAlongPlane();
    }
    if (quadList.getValue().size()>0)
    {
        selectQuadsFromString();
    }

    quadPressureMap.createTopologyHandler(m_topology);

    initQuadInformation();
}

template <class DataTypes>
void QuadPressureForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& /* d_x */, const DataVecDeriv& /* d_v */)
{
    VecDeriv& f = *d_f.beginEdit();
    Deriv force;

    const sofa::type::vector<Index>& my_map = quadPressureMap.getMap2Elements();
    const sofa::type::vector<QuadPressureInformation>& my_subset = quadPressureMap.getValue();

    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        force=my_subset[i].force/4;
        f[m_topology->getQuad(my_map[i])[0]]+=force;
        f[m_topology->getQuad(my_map[i])[1]]+=force;
        f[m_topology->getQuad(my_map[i])[2]]+=force;
        f[m_topology->getQuad(my_map[i])[3]]+=force;

    }
    d_f.endEdit();
    updateQuadInformation();
}


template<class DataTypes>
void QuadPressureForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */)
{
    //Todo

    //Remove warning
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
    (void)kFactor;

    return;
}


template<class DataTypes>
void QuadPressureForceField<DataTypes>::initQuadInformation()
{
    const sofa::type::vector<Index>& my_map = quadPressureMap.getMap2Elements();
    auto my_subset = sofa::helper::getWriteOnlyAccessor(quadPressureMap);

    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        const auto& q = this->m_topology->getQuad(my_map[i]);

        const auto& n0 = DataTypes::getCPos(x0[q[0]]);
        const auto& n1 = DataTypes::getCPos(x0[q[1]]);
        const auto& n2 = DataTypes::getCPos(x0[q[2]]);
        const auto& n3 = DataTypes::getCPos(x0[q[3]]);

        my_subset[i].area = sofa::geometry::Quad::area(n0, n1, n2, n3);
        my_subset[i].force=pressure.getValue()*my_subset[i].area;
    }
}


template<class DataTypes>
void QuadPressureForceField<DataTypes>::updateQuadInformation()
{
    sofa::type::vector<QuadPressureInformation>& my_subset = *(quadPressureMap).beginEdit();

    for (unsigned int i=0; i<my_subset.size(); ++i)
        my_subset[i].force=(pressure.getValue()*my_subset[i].area);

    quadPressureMap.endEdit();
}


template <class DataTypes>
void QuadPressureForceField<DataTypes>::selectQuadsAlongPlane()
{
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    std::vector<bool> vArray;

    vArray.resize(x.size());

    for( unsigned int i=0; i<x.size(); ++i)
    {
        vArray[i]=isPointInPlane(x[i]);
    }

    sofa::type::vector<QuadPressureInformation>& my_subset = *(quadPressureMap).beginEdit();
    type::vector<Index> inputQuads;

    for (size_t n=0; n<m_topology->getNbQuads(); ++n)
    {
        if ((vArray[m_topology->getQuad(n)[0]]) && (vArray[m_topology->getQuad(n)[1]])&& (vArray[m_topology->getQuad(n)[2]])&& (vArray[m_topology->getQuad(n)[3]]) )
        {
            // insert a dummy element : computation of pressure done later
            QuadPressureInformation q;
            q.area = 0;
            my_subset.push_back(q);
            inputQuads.push_back(n);
        }
    }
    quadPressureMap.endEdit();
    quadPressureMap.setMap2Elements(inputQuads);

    return;
}


template <class DataTypes>
void QuadPressureForceField<DataTypes>::selectQuadsFromString()
{
    sofa::type::vector<QuadPressureInformation>& my_subset = *(quadPressureMap).beginEdit();
    type::vector<Index> _quadList = quadList.getValue();

    quadPressureMap.setMap2Elements(_quadList);

    for (unsigned int i = 0; i < _quadList.size(); ++i)
    {
        QuadPressureInformation q;
        q.area = 0;
        my_subset.push_back(q);
    }

    quadPressureMap.endEdit();

    return;
}

template <class DataTypes>
bool QuadPressureForceField<DataTypes>::isPointInPlane(Coord p)
{
    Real d=dot(p,normal.getValue());
    if ((d>dmin.getValue())&& (d<dmax.getValue()))
        return true;
    else
        return false;
}

template <class DataTypes>
void QuadPressureForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* )
{
    // force does not depend on the position, so the derivative with respect
    // to position is null => stiffness matrix is null
}

template <class DataTypes>
void QuadPressureForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
void QuadPressureForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    if (!p_showForces.getValue())
        return;

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, true);

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    vparams->drawTool()->disableLighting();
    std::vector<sofa::type::Vec3> vertices;
    const sofa::type::RGBAColor color = sofa::type::RGBAColor::green();

    const sofa::type::vector<Index>& my_map = quadPressureMap.getMap2Elements();

    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        for(unsigned int j=0 ; j<4 ; j++)
            vertices.push_back(x[m_topology->getQuad(my_map[i])[j]]);
    }
    vparams->drawTool()->drawQuads(vertices, color);


    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, false);

}

} // namespace sofa::component::mechanicalload
