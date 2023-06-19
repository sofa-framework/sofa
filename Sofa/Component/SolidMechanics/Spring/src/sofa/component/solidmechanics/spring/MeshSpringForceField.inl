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
#include <sofa/component/solidmechanics/spring/MeshSpringForceField.h>
#include <sofa/component/solidmechanics/spring/StiffSpringForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/RGBAColor.h>
#include <iostream>

namespace sofa::component::solidmechanics::spring
{

template <class DataTypes>
MeshSpringForceField<DataTypes>::MeshSpringForceField()
    : d_linesStiffness(initData(&d_linesStiffness,Real(0),"linesStiffness","Stiffness for the Lines",true))
    , d_linesDamping(initData(&d_linesDamping,Real(0),"linesDamping","Damping for the Lines",true))
    , d_trianglesStiffness(initData(&d_trianglesStiffness,Real(0),"trianglesStiffness","Stiffness for the Triangles",true))
    , d_trianglesDamping(initData(&d_trianglesDamping,Real(0),"trianglesDamping","Damping for the Triangles",true))
    , d_quadsStiffness(initData(&d_quadsStiffness,Real(0),"quadsStiffness","Stiffness for the Quads",true))
    , d_quadsDamping(initData(&d_quadsDamping,Real(0),"quadsDamping","Damping for the Quads",true))
    , d_tetrahedraStiffness(initData(&d_tetrahedraStiffness,Real(0),"tetrahedraStiffness","Stiffness for the Tetrahedra",true))
    , d_tetrahedraDamping(initData(&d_tetrahedraDamping,Real(0),"tetrahedraDamping","Damping for the Tetrahedra",true))
    , d_cubesStiffness(initData(&d_cubesStiffness,Real(0),"cubesStiffness","Stiffness for the Cubes",true))
    , d_cubesDamping(initData(&d_cubesDamping,Real(0),"cubesDamping","Damping for the Cubes",true))
    , d_noCompression( initData(&d_noCompression, false, "noCompression", "Only consider elongation", false))
    , d_drawMinElongationRange(initData(&d_drawMinElongationRange, Real(8.), "drawMinElongationRange","Min range of elongation (red eongation - blue neutral - green compression)"))
    , d_drawMaxElongationRange(initData(&d_drawMaxElongationRange, Real(15.), "drawMaxElongationRange","Max range of elongation (red eongation - blue neutral - green compression)"))
    , d_drawSpringSize(initData(&d_drawSpringSize, Real(8.), "drawSpringSize","Size of drawed lines"))
    , d_localRange( initData(&d_localRange, type::Vec<2, sofa::Index>(sofa::InvalidID, sofa::InvalidID), "localRange", "optional range of local DOF indices. Any computation involving only indices outside of this range are discarded (useful for parallelization using mesh partitionning)" ) )
    , l_topology(initLink("topology", "link to the topology container"))
{
	this->ks.setDisplayed(false);
    this->kd.setDisplayed(false);
    this->addAlias(&d_linesStiffness,     "stiffness"); this->addAlias(&d_linesDamping,     "damping");
    this->addAlias(&d_trianglesStiffness, "stiffness"); this->addAlias(&d_trianglesDamping, "damping");
    this->addAlias(&d_quadsStiffness,     "stiffness"); this->addAlias(&d_quadsDamping,     "damping");
    this->addAlias(&d_tetrahedraStiffness,"stiffness"); this->addAlias(&d_tetrahedraDamping, "damping");
    this->addAlias(&d_cubesStiffness,     "stiffness"); this->addAlias(&d_cubesDamping,      "damping");
    //Name changes: keep compatibility with old version
    this->addAlias(&d_tetrahedraStiffness,"tetrasStiffness"); this->addAlias(&d_tetrahedraDamping, "tetrasDamping");
}

template <class DataTypes>
MeshSpringForceField<DataTypes>::~MeshSpringForceField()
{
}

template<class DataTypes>
void MeshSpringForceField<DataTypes>::addSpring(std::set<std::pair<sofa::Index, sofa::Index> >& sset, sofa::Index m1, sofa::Index m2, Real stiffness, Real damping)
{
    const auto& localRange = d_localRange.getValue();
    if (localRange[0] != sofa::InvalidID)
    {
        if (m1 < localRange[0] || m2 < localRange[0]) return;
    }
    if (localRange[1] != sofa::InvalidID)
    {
        if (m1 > localRange[1] && m2 > localRange[1]) return;
    }

    if (m1<m2)
    {
        if (sset.count(std::make_pair(m1,m2))>0) return;
        sset.insert(std::make_pair(m1,m2));
    }
    else
    {
        if (sset.count(std::make_pair(m2,m1))>0) return;
        sset.insert(std::make_pair(m2,m1));
    }
    const Real l = ((mstate2->read(core::ConstVecCoordId::restPosition())->getValue())[m2] - (mstate1->read(core::ConstVecCoordId::restPosition())->getValue())[m1]).norm();
    if (l > std::numeric_limits<Real>::epsilon())
    {
        sofa::helper::getWriteAccessor(springs)->emplace_back(m1,m2,stiffness/l, damping/l, l, d_noCompression.getValue());
    }
    else
    {
        sofa::helper::getWriteAccessor(springs)->emplace_back(m1,m2,stiffness, damping, l, d_noCompression.getValue());
    }
}

template<class DataTypes>
void MeshSpringForceField<DataTypes>::init()
{
    StiffSpringForceField<DataTypes>::clear();
    if(!(mstate1) || !(mstate2))
        mstate2 = mstate1 = dynamic_cast<sofa::core::behavior::MechanicalState<DataTypes> *>(this->getContext()->getMechanicalState());

    if (mstate1==mstate2)
    {

        if (l_topology.empty())
        {
            msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
            l_topology.set(this->getContext()->getMeshTopologyLink());
        }

        sofa::core::topology::BaseMeshTopology* _topology = l_topology.get();
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

        if (_topology == nullptr)
        {
            msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
        
        std::set< std::pair<sofa::Index, sofa::Index> > sset;
        sofa::Size n;
        Real s, d;
        if (d_linesStiffness.getValue() != 0.0 || d_linesDamping.getValue() != 0.0)
        {
            s = d_linesStiffness.getValue();
            d = d_linesDamping.getValue();
            n = _topology->getNbLines();
            for (sofa::Index i=0; i<n; ++i)
            {
                sofa::core::topology::BaseMeshTopology::Line e = _topology->getLine(i);
                addSpring(sset, e[0], e[1], s, d);
            }
        }
        if (d_trianglesStiffness.getValue() != 0.0 || d_trianglesDamping.getValue() != 0.0)
        {
            s = d_trianglesStiffness.getValue();
            d = d_trianglesDamping.getValue();
            n = _topology->getNbTriangles();
            for (sofa::Index i=0; i<n; ++i)
            {
                sofa::core::topology::BaseMeshTopology::Triangle e = _topology->getTriangle(i);
                addSpring(sset, e[0], e[1], s, d);
                addSpring(sset, e[0], e[2], s, d);
                addSpring(sset, e[1], e[2], s, d);
            }
        }
        if (d_quadsStiffness.getValue() != 0.0 || d_quadsDamping.getValue() != 0.0)
        {
            s = d_quadsStiffness.getValue();
            d = d_quadsDamping.getValue();
            n = _topology->getNbQuads();
            for (sofa::Index i=0; i<n; ++i)
            {
                sofa::core::topology::BaseMeshTopology::Quad e = _topology->getQuad(i);
                addSpring(sset, e[0], e[1], s, d);
                addSpring(sset, e[0], e[2], s, d);
                addSpring(sset, e[0], e[3], s, d);
                addSpring(sset, e[1], e[2], s, d);
                addSpring(sset, e[1], e[3], s, d);
                addSpring(sset, e[2], e[3], s, d);
            }
        }
        if (d_tetrahedraStiffness.getValue() != 0.0 || d_tetrahedraDamping.getValue() != 0.0)
        {
            s = d_tetrahedraStiffness.getValue();
            d = d_tetrahedraDamping.getValue();
            n = _topology->getNbTetrahedra();
            for (sofa::Index i=0; i<n; ++i)
            {
                sofa::core::topology::BaseMeshTopology::Tetra e = _topology->getTetrahedron(i);
                addSpring(sset, e[0], e[1], s, d);
                addSpring(sset, e[0], e[2], s, d);
                addSpring(sset, e[0], e[3], s, d);
                addSpring(sset, e[1], e[2], s, d);
                addSpring(sset, e[1], e[3], s, d);
                addSpring(sset, e[2], e[3], s, d);
            }
        }

        if (d_cubesStiffness.getValue() != 0.0 || d_cubesDamping.getValue() != 0.0)
        {
            s = d_cubesStiffness.getValue();
            d = d_cubesDamping.getValue();

            n = _topology->getNbHexahedra();
            for (sofa::Index i=0; i<n; ++i)
            {
                sofa::core::topology::BaseMeshTopology::Hexa e = _topology->getHexahedron(i);

                for (sofa::Index k=0; k<8; k++)
                {
                    for (sofa::Index j=k+1; j<8; j++)
                    {
                        addSpring(sset, e[k], e[j], s, d);
                    }
                }
            }
        }
    }

    StiffSpringForceField<DataTypes>::init();
}


template<class DataTypes>
void MeshSpringForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if(this->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid || !mstate1 || !mstate2)
        return ;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    if(vparams->displayFlags().getShowForceFields())
    {
        typedef typename Inherit1::Spring  Spring;
        const sofa::type::vector<Spring> &ss = springs.getValue();
        
        const VecCoord& p1 = mstate1->read(core::ConstVecCoordId::position())->getValue();
        const VecCoord& p2 = mstate2->read(core::ConstVecCoordId::position())->getValue();
        
        Real minElongation = std::numeric_limits<Real>::max();
        Real maxElongation = 0.;
        for (sofa::Index i=0; i<ss.size(); ++i)
        {
            const Spring& s = ss[i];
            Deriv v = p1[s.m1] - p2[s.m2];
            Real elongation = (s.initpos - v.norm()) / s.initpos;
            maxElongation = std::max(maxElongation, elongation);
            minElongation = std::min(minElongation, elongation);
        }
        
        const Real minElongationRange = d_drawMinElongationRange.getValue();
        const Real maxElongationRange = d_drawMaxElongationRange.getValue();
        Real range = std::min(std::max(maxElongation, std::abs(minElongation)), maxElongationRange) - minElongationRange;
        range = (range < 0.) ? 1. : range;
        const Real drawSpringSize = d_drawSpringSize.getValue();

        for (sofa::Index i=0; i<ss.size(); ++i)
        {
            const Spring& s = ss[i];
            const Coord pa[2] = {p1[s.m1], p2[s.m2]};
            const std::vector<sofa::type::Vec3> points(pa, pa+2);
            Deriv v = pa[0] - pa[1];
            Real elongation = (s.initpos - v.norm()) / s.initpos;
            Real R = 0.;
            Real G = 0.;
            Real B = 1.;
            if(elongation < 0.)
            {
                elongation = std::abs(elongation);
                B = (range-std::min(elongation - minElongationRange, range))/range;
                B = (B < 0.) ? 0. : B;
                R = 1. - B;
            }
            else
            {
                B = (range-std::min(elongation - minElongationRange, range))/range;
                B = (B < 0.) ? 0. : B;
                G = 1. - B;
            }

            vparams->drawTool()->drawLines(points, float(drawSpringSize), sofa::type::RGBAColor{ float(R), float(G), float(B), 1.f });
        }


    }
}

} // namespace sofa::component::solidmechanics::spring
