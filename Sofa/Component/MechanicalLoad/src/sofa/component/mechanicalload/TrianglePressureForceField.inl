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

#include <sofa/component/mechanicalload/TrianglePressureForceField.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/core/topology/TopologySubsetData.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/type/RGBAColor.h>
#include <vector>
#include <set>

namespace sofa::component::mechanicalload
{

template <class DataTypes> TrianglePressureForceField<DataTypes>::~TrianglePressureForceField()
{
}

template <class DataTypes>  TrianglePressureForceField<DataTypes>::TrianglePressureForceField()
    : pressure(initData(&pressure, "pressure", "Pressure force per unit area"))
    , cauchyStress(initData(&cauchyStress, MatSym3(),"cauchyStress", "Cauchy Stress applied on the normal of each triangle"))
    , triangleList(initData(&triangleList,"triangleList", "Indices of triangles separated with commas where a pressure is applied"))
    , normal(initData(&normal,"normal", "Normal direction for the plane selection of triangles"))
    , dmin(initData(&dmin,(Real)0.0, "dmin", "Minimum distance from the origin along the normal direction"))
    , dmax(initData(&dmax,(Real)0.0, "dmax", "Maximum distance from the origin along the normal direction"))
    , p_showForces(initData(&p_showForces, (bool)false, "showForces", "draw triangles which have a given pressure"))
    , p_useConstantForce(initData(&p_useConstantForce, (bool)true, "useConstantForce", "applied force is computed as the pressure vector times the area at rest"))
    , l_topology(initLink("topology", "link to the topology container"))
    , trianglePressureMap(initData(&trianglePressureMap, "trianglePressureMap", "map between edge indices and their pressure"))
    , m_topology(nullptr)
{}

template <class DataTypes> void TrianglePressureForceField<DataTypes>::init()
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
        selectTrianglesAlongPlane();
    }
    if (triangleList.getValue().size()>0)
    {
        selectTrianglesFromString();
    }

    trianglePressureMap.createTopologyHandler(m_topology);

    initTriangleInformation();

    if(p_useConstantForce.m_isSet() || p_useConstantForce.getValue() == false)
    {
        msg_deprecated() << "Non constant pressure force field has been removed.";
    }
}

template <class DataTypes>
void TrianglePressureForceField<DataTypes>::addForce(const core::MechanicalParams*  /*mparams*/, DataVecDeriv& d_f, const DataVecCoord&  /*d_x*/ , const DataVecDeriv& /*d_v*/)
{
    if (!p_useConstantForce.getValue())
    {
        msg_deprecated() << "Non constant force field has been deprecated because of an invalid computation. Please contact sofa-dev to request for integration of a valid version.";
        return;
    }

    auto f =sofa::helper::getWriteAccessor(d_f);
    const sofa::type::vector<Index>& my_map = trianglePressureMap.getMap2Elements();

    const sofa::type::vector<TrianglePressureInformation>& my_subset = trianglePressureMap.getValue();
    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        const auto force=my_subset[i].force/3;
        f[m_topology->getTriangle(my_map[i])[0]]+=force;
        f[m_topology->getTriangle(my_map[i])[1]]+=force;
        f[m_topology->getTriangle(my_map[i])[2]]+=force;
    }
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::addDForce(const core::MechanicalParams* /* mparams */, DataVecDeriv&  /*d_df*/ , const DataVecDeriv&  /*d_dx*/ )
{
    if (!p_useConstantForce.getValue())
    {
        msg_error() << "Implementation for non constant pressure is missing, please contact sofa dev if you see this message";
        return;
    }

    return;
}

template<class DataTypes>
void TrianglePressureForceField<DataTypes>::initTriangleInformation()
{
    const sofa::type::vector<Index>& my_map = trianglePressureMap.getMap2Elements();
    auto my_subset = sofa::helper::getWriteOnlyAccessor(trianglePressureMap);

    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        const auto& t = this->m_topology->getTriangle(my_map[i]);

        const auto& n0 = DataTypes::getCPos(x0[t[0]]);
        const auto& n1 = DataTypes::getCPos(x0[t[1]]);
        const auto& n2 = DataTypes::getCPos(x0[t[2]]);

        my_subset[i].area = sofa::geometry::Triangle::area(n0, n1, n2);
        my_subset[i].force=pressure.getValue()*my_subset[i].area;
    }
}

template<class DataTypes>
bool TrianglePressureForceField<DataTypes>::isPointInPlane(Coord p)
{
    Real d=dot(p,normal.getValue());
    if ((d>dmin.getValue())&& (d<dmax.getValue()))
        return true;
    else
        return false;
}

template<class DataTypes>
void TrianglePressureForceField<DataTypes>::updateTriangleInformation()
{
    sofa::type::vector<TrianglePressureInformation>& my_subset = *(trianglePressureMap).beginEdit();

    for (unsigned int i=0; i<my_subset.size(); ++i)
        my_subset[i].force=(pressure.getValue()*my_subset[i].area);

    trianglePressureMap.endEdit();
}


template <class DataTypes>
void TrianglePressureForceField<DataTypes>::selectTrianglesAlongPlane()
{
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    std::vector<bool> vArray;

    vArray.resize(x.size());

    for( unsigned int i=0; i<x.size(); ++i)
    {
        vArray[i]=isPointInPlane(x[i]);
    }

    auto my_subset = sofa::helper::getWriteAccessor(trianglePressureMap);
    type::vector<Index> inputTriangles;

    for (size_t n=0; n<m_topology->getNbTriangles(); ++n)
    {
        if ((vArray[m_topology->getTriangle(n)[0]]) && (vArray[m_topology->getTriangle(n)[1]])&& (vArray[m_topology->getTriangle(n)[2]]) )
        {
            // insert a dummy element : computation of pressure done later
            TrianglePressureInformation t;
            t.area = 0;
            my_subset.push_back(t);
            inputTriangles.push_back(n);
        }
    }
    trianglePressureMap.setMap2Elements(inputTriangles);

    return;
}


template <class DataTypes>
void TrianglePressureForceField<DataTypes>::selectTrianglesFromString()
{
    auto my_subset = sofa::helper::getWriteAccessor(trianglePressureMap);

    type::vector<Index> _triangleList = triangleList.getValue();

    trianglePressureMap.setMap2Elements(_triangleList);
    for (unsigned int i = 0; i < _triangleList.size(); ++i)
    {
        TrianglePressureInformation t;
        t.area = 0;
        my_subset.push_back(t);
    }

    return;
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!p_showForces.getValue())
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, true);

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    vparams->drawTool()->disableLighting();

    const sofa::type::RGBAColor&  color = sofa::type::RGBAColor::green();
    std::vector< sofa::type::Vec3 > vertices;

    const sofa::type::vector<Index>& my_map = trianglePressureMap.getMap2Elements();
    const sofa::type::vector<TrianglePressureInformation>& my_subset = trianglePressureMap.getValue();
    std::vector< sofa::type::Vec3 > forceVectors;
    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        Deriv force = my_subset[i].force / 3;
        for (unsigned int j = 0; j < 3; j++)
        {
            sofa::type::Vec3 p = x[m_topology->getTriangle(my_map[i])[j]];
            vertices.push_back(p);
            forceVectors.push_back(p);
            forceVectors.push_back(p + force);
        }
    }
    vparams->drawTool()->drawTriangles(vertices, color);
    vparams->drawTool()->drawLines(forceVectors, 1, sofa::type::RGBAColor::red());

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, false);


}

template <class DataTypes>
void TrianglePressureForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template <class DataTypes>
void TrianglePressureForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* /*matrix*/)
{
    if (!p_useConstantForce.getValue())
    {
        msg_error() << "The non constant version of this force field is missing the derivative computation";
        return;
    }

    // force does not depend on the position, so the derivative with respect
    // to position is null => stiffness matrix is null
}

template<class DataTypes>
SReal TrianglePressureForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
{
    msg_warning() << "Method getPotentialEnergy not implemented yet.";
    return 0.0;
}

} // namespace sofa::component::mechanicalload
