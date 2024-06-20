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
    : d_pressure(initData(&d_pressure, "pressure", "Pressure force per unit area"))
    , d_cauchyStress(initData(&d_cauchyStress, MatSym3(), "cauchyStress", "Cauchy Stress applied on the normal of each triangle"))
    , d_triangleList(initData(&d_triangleList, "triangleList", "Indices of triangles separated with commas where a pressure is applied"))
    , d_showForces(initData(&d_showForces, (bool)false, "showForces", "draw triangles which have a given pressure"))
    , d_useConstantForce(initData(&d_useConstantForce, (bool)true, "useConstantForce", "applied force is computed as the pressure vector times the area at rest"))
    , l_topology(initLink("topology", "link to the topology container"))
    , d_trianglePressureMap(initData(&d_trianglePressureMap, "trianglePressureMap", "Map between triangle indices and their pressure"))
    , m_topology(nullptr)
{
    pressure.setParent(&d_pressure);
    cauchyStress.setParent(&d_cauchyStress);
    triangleList.setParent(&d_triangleList);
    p_showForces.setParent(&d_showForces);
    p_useConstantForce.setParent(&d_useConstantForce);


    this->addUpdateCallback("pressure_change", { &pressure }, [this](const core::DataTracker& t)
    {
        SOFA_UNUSED(t);
        updateTriangleInformation();
        return sofa::core::objectmodel::ComponentState::Valid;
    }, {});


    this->addUpdateCallback("triangles_change", { &triangleList }, [this](const core::DataTracker& t)
    {
        SOFA_UNUSED(t);
        initTriangleInformation();
        return sofa::core::objectmodel::ComponentState::Valid;
    }, {});

}


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

    d_trianglePressureMap.createTopologyHandler(m_topology);

    initTriangleInformation();

}

template <class DataTypes>
void TrianglePressureForceField<DataTypes>::addForce(const core::MechanicalParams*  /*mparams*/, DataVecDeriv& d_f, const DataVecCoord&  d_x , const DataVecDeriv& /* d_v */)
{

    VecDeriv& f = *d_f.beginEdit();

    const sofa::type::vector<Index>& my_map = d_trianglePressureMap.getMap2Elements();

	if (d_useConstantForce.getValue()) {
		const sofa::type::vector<TrianglePressureInformation>& my_subset = d_trianglePressureMap.getValue();


		for (unsigned int i=0; i<my_map.size(); ++i)
		{
			const auto force=my_subset[i].force/3;
			f[m_topology->getTriangle(my_map[i])[0]]+=force;
			f[m_topology->getTriangle(my_map[i])[1]]+=force;
			f[m_topology->getTriangle(my_map[i])[2]]+=force;

		}
	} else {
        typedef core::topology::BaseMeshTopology::Triangle Triangle;
		const sofa::type::vector<Triangle> &ta = m_topology->getTriangles();
		const  VecDeriv p = d_x.getValue();
		MatSym3 cauchy=d_cauchyStress.getValue();
		Deriv areaVector;

		for (unsigned int i=0; i<my_map.size(); ++i)
		{
			const Triangle &t=ta[my_map[i]];
			areaVector=cross(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])/6.0f;
			const auto force=cauchy*areaVector;
			for (size_t j=0;j<3;++j) {
				f[t[j]]+=force;
			}
		}

	}
    d_f.endEdit();
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv&  /*d_df*/ , const DataVecDeriv&  /*d_dx*/ )
{
	mparams->kFactor();
	return;
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::initTriangleInformation()
{
    if (triangleList.getValue().empty())
        return;

    // Get list of input triangle indices
    type::vector<Index> _triangleList = triangleList.getValue();

    // Get write access to TopologySubset Data storing pressure information per triangle
    auto my_subset = sofa::helper::getWriteOnlyAccessor(d_trianglePressureMap);

    // Set the list of triangles indices as map of this TopologySubset Data
    d_trianglePressureMap.setMap2Elements(_triangleList);

    // Fill pressure data
    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    const Deriv& my_pressure = pressure.getValue();

    for (unsigned int i = 0; i < _triangleList.size(); ++i)
    {
        const auto& t = this->m_topology->getTriangle(_triangleList[i]);

        const auto& n0 = DataTypes::getCPos(x0[t[0]]);
        const auto& n1 = DataTypes::getCPos(x0[t[1]]);
        const auto& n2 = DataTypes::getCPos(x0[t[2]]);

        TrianglePressureInformation tInfo;
        tInfo.area = sofa::geometry::Triangle::area(n0, n1, n2);
        tInfo.force = my_pressure * tInfo.area;
        my_subset.push_back(tInfo);
    }
}


template<class DataTypes>
void TrianglePressureForceField<DataTypes>::updateTriangleInformation()
{
    sofa::type::vector<TrianglePressureInformation>& my_subset = *(d_trianglePressureMap).beginEdit();

    const Deriv& my_pressure = pressure.getValue();
    for (unsigned int i = 0; i < my_subset.size(); ++i)
        my_subset[i].force = (my_pressure * my_subset[i].area);

    d_trianglePressureMap.endEdit();
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

    const sofa::type::vector<Index>& my_map = d_trianglePressureMap.getMap2Elements();
    const sofa::type::vector<TrianglePressureInformation>& my_subset = d_trianglePressureMap.getValue();
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

template<class DataTypes>
SReal TrianglePressureForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
{
    msg_warning() << "Method getPotentialEnergy not implemented yet.";
    return 0.0;
}

} // namespace sofa::component::mechanicalload
