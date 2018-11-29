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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_MESHSPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_MESHSPRINGFORCEFIELD_INL

#include <SofaDeformable/MeshSpringForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace interactionforcefield
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
    , d_localRange( initData(&d_localRange, defaulttype::Vec<2,int>(-1,-1), "localRange", "optional range of local DOF indices. Any computation involving only indices outside of this range are discarded (useful for parallelization using mesh partitionning)" ) )
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
void MeshSpringForceField<DataTypes>::addSpring(std::set<std::pair<int,int> >& sset, int m1, int m2, Real stiffness, Real damping)
{
    if (d_localRange.getValue()[0] >= 0)
    {
        if (m1 < d_localRange.getValue()[0] || m2 < d_localRange.getValue()[0]) return;
    }
    if (d_localRange.getValue()[1] >= 0)
    {
        if (m1 > d_localRange.getValue()[1] && m2 > d_localRange.getValue()[1]) return;
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
    Real l = ((mstate2->read(core::ConstVecCoordId::restPosition())->getValue())[m2] - (mstate1->read(core::ConstVecCoordId::restPosition())->getValue())[m1]).norm();
    springs.beginEdit()->push_back(typename SpringForceField<DataTypes>::Spring(m1,m2,stiffness/l, damping/l, l, d_noCompression.getValue()));
    springs.endEdit();
}

template<class DataTypes>
void MeshSpringForceField<DataTypes>::init()
{
    StiffSpringForceField<DataTypes>::clear();
    if(!(mstate1) || !(mstate2))
        mstate2 = mstate1 = dynamic_cast<sofa::core::behavior::MechanicalState<DataTypes> *>(this->getContext()->getMechanicalState());

    if (mstate1==mstate2)
    {
        sofa::core::topology::BaseMeshTopology* topology = this->getContext()->getMeshTopology();

        if (topology != NULL)
        {
            std::set< std::pair<int,int> > sset;
            size_t n;
            Real s, d;
            if (d_linesStiffness.getValue() != 0.0 || d_linesDamping.getValue() != 0.0)
            {
                s = d_linesStiffness.getValue();
                d = d_linesDamping.getValue();
                n = topology->getNbLines();
                for (size_t i=0; i<n; ++i)
                {
                    sofa::core::topology::BaseMeshTopology::Line e = topology->getLine(i);
                    addSpring(sset, e[0], e[1], s, d);
                }
            }
            if (d_trianglesStiffness.getValue() != 0.0 || d_trianglesDamping.getValue() != 0.0)
            {
                s = d_trianglesStiffness.getValue();
                d = d_trianglesDamping.getValue();
                n = topology->getNbTriangles();
                for (size_t i=0; i<n; ++i)
                {
                    sofa::core::topology::BaseMeshTopology::Triangle e = topology->getTriangle(i);
                    addSpring(sset, e[0], e[1], s, d);
                    addSpring(sset, e[0], e[2], s, d);
                    addSpring(sset, e[1], e[2], s, d);
                }
            }
            if (d_quadsStiffness.getValue() != 0.0 || d_quadsDamping.getValue() != 0.0)
            {
                s = d_quadsStiffness.getValue();
                d = d_quadsDamping.getValue();
                n = topology->getNbQuads();
                for (size_t i=0; i<n; ++i)
                {
                    sofa::core::topology::BaseMeshTopology::Quad e = topology->getQuad(i);
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
                n = topology->getNbTetrahedra();
                for (size_t i=0; i<n; ++i)
                {
                    sofa::core::topology::BaseMeshTopology::Tetra e = topology->getTetrahedron(i);
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

                n = topology->getNbHexahedra();
                for (size_t i=0; i<n; ++i)
                {
                    sofa::core::topology::BaseMeshTopology::Hexa e = topology->getHexahedron(i);

                    for (int k=0; k<8; k++)
                    {
                        for (int j=k+1; j<8; j++)
                        {
                            addSpring(sset, e[k], e[j], s, d);
                        }
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
    if(vparams->displayFlags().getShowForceFields())
    {
        typedef typename Inherit1::Spring  Spring;
        const sofa::helper::vector<Spring> &ss = springs.getValue();
        
        const VecCoord& p1 = mstate1->read(core::ConstVecCoordId::position())->getValue();
        const VecCoord& p2 = mstate2->read(core::ConstVecCoordId::position())->getValue();
        
        Real minElongation = std::numeric_limits<Real>::max();
        Real maxElongation = 0.;
        for (size_t i=0; i<ss.size(); ++i)
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

        for (size_t i=0; i<ss.size(); ++i)
        {
            const Spring& s = ss[i];
            const Coord pa[2] = {p1[s.m1], p2[s.m2]};
            const std::vector<sofa::defaulttype::Vector3> points(pa, pa+2);
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

            vparams->drawTool()->drawLines(points, drawSpringSize, sofa::defaulttype::Vec4f(R, G, B, 1.f));
        }
    }
}

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_MESHSPRINGFORCEFIELD_INL */
