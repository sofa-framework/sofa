/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
MeshSpringForceField<DataTypes>::~MeshSpringForceField()
{
}

template<class DataTypes>
void MeshSpringForceField<DataTypes>::addSpring(std::set<std::pair<int,int> >& sset, int m1, int m2, Real stiffness, Real damping)
{
    if (localRange.getValue()[0] >= 0)
    {
        if (m1 < localRange.getValue()[0] || m2 < localRange.getValue()[0]) return;
    }
    if (localRange.getValue()[1] >= 0)
    {
        if (m1 > localRange.getValue()[1] && m2 > localRange.getValue()[1]) return;
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
    Real l = ((this->mstate2->read(core::ConstVecCoordId::restPosition())->getValue())[m2] - (this->mstate1->read(core::ConstVecCoordId::restPosition())->getValue())[m1]).norm();
     this->springs.beginEdit()->push_back(typename SpringForceField<DataTypes>::Spring(m1,m2,stiffness/l, damping/l, l, noCompression.getValue()));
      this->springs.endEdit();
}

template<class DataTypes>
void MeshSpringForceField<DataTypes>::init()
{
    this->StiffSpringForceField<DataTypes>::clear();
    if(!(this->mstate1) || !(this->mstate2))
        this->mstate2 = this->mstate1 = dynamic_cast<sofa::core::behavior::MechanicalState<DataTypes> *>(this->getContext()->getMechanicalState());

    if (this->mstate1==this->mstate2)
    {
        sofa::core::topology::BaseMeshTopology* topology = this->getContext()->getMeshTopology();

        if (topology != NULL)
        {
            std::set< std::pair<int,int> > sset;
            int n;
            Real s, d;
            if (this->linesStiffness.getValue() != 0.0 || this->linesDamping.getValue() != 0.0)
            {
                s = this->linesStiffness.getValue();
                d = this->linesDamping.getValue();
                n = topology->getNbLines();
                for (int i=0; i<n; ++i)
                {
                    sofa::core::topology::BaseMeshTopology::Line e = topology->getLine(i);
                    this->addSpring(sset, e[0], e[1], s, d);
                }
            }
            if (this->trianglesStiffness.getValue() != 0.0 || this->trianglesDamping.getValue() != 0.0)
            {
                s = this->trianglesStiffness.getValue();
                d = this->trianglesDamping.getValue();
                n = topology->getNbTriangles();
                for (int i=0; i<n; ++i)
                {
                    sofa::core::topology::BaseMeshTopology::Triangle e = topology->getTriangle(i);
                    this->addSpring(sset, e[0], e[1], s, d);
                    this->addSpring(sset, e[0], e[2], s, d);
                    this->addSpring(sset, e[1], e[2], s, d);
                }
            }
            if (this->quadsStiffness.getValue() != 0.0 || this->quadsDamping.getValue() != 0.0)
            {
                s = this->quadsStiffness.getValue();
                d = this->quadsDamping.getValue();
                n = topology->getNbQuads();
                for (int i=0; i<n; ++i)
                {
                    sofa::core::topology::BaseMeshTopology::Quad e = topology->getQuad(i);
                    this->addSpring(sset, e[0], e[1], s, d);
                    this->addSpring(sset, e[0], e[2], s, d);
                    this->addSpring(sset, e[0], e[3], s, d);
                    this->addSpring(sset, e[1], e[2], s, d);
                    this->addSpring(sset, e[1], e[3], s, d);
                    this->addSpring(sset, e[2], e[3], s, d);
                }
            }
            if (this->tetrahedraStiffness.getValue() != 0.0 || this->tetrahedraDamping.getValue() != 0.0)
            {
                s = this->tetrahedraStiffness.getValue();
                d = this->tetrahedraDamping.getValue();
                n = topology->getNbTetrahedra();
                for (int i=0; i<n; ++i)
                {
                    sofa::core::topology::BaseMeshTopology::Tetra e = topology->getTetrahedron(i);
                    this->addSpring(sset, e[0], e[1], s, d);
                    this->addSpring(sset, e[0], e[2], s, d);
                    this->addSpring(sset, e[0], e[3], s, d);
                    this->addSpring(sset, e[1], e[2], s, d);
                    this->addSpring(sset, e[1], e[3], s, d);
                    this->addSpring(sset, e[2], e[3], s, d);
                }
            }

            if (this->cubesStiffness.getValue() != 0.0 || this->cubesDamping.getValue() != 0.0)
            {
                s = this->cubesStiffness.getValue();
                d = this->cubesDamping.getValue();
#ifdef SOFA_NEW_HEXA
                n = topology->getNbHexahedra();
                for (int i=0; i<n; ++i)
                {
                    sofa::core::topology::BaseMeshTopology::Hexa e = topology->getHexahedron(i);
#else
                n = topology->getNbCubes();
                for (int i=0; i<n; ++i)
                {
                    sofa::core::topology::BaseMeshTopology::Cube e = topology->getCube(i);
#endif
                    for (int i=0; i<8; i++)
                        for (int j=i+1; j<8; j++)
                        {
                            this->addSpring(sset, e[i], e[j], s, d);
                        }
                }
            }
        }
    }
    this->StiffSpringForceField<DataTypes>::init();
}


template<class DataTypes>
void MeshSpringForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if( d_draw.getValue() )
    {
        typedef typename Inherit1::Spring  Spring;
        sofa::helper::vector<Spring >& ss = *this->springs.beginEdit();
        
        const VecCoord& p1 = this->mstate1->read(core::ConstVecCoordId::position())->getValue();
        const VecCoord& p2 = this->mstate2->read(core::ConstVecCoordId::position())->getValue();
        
        Real minElongation = std::numeric_limits<Real>::max();
        Real maxElongation = 0.;
        for (unsigned int i=0; i<ss.size(); ++i)
        {
            Spring& s = ss[i];
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

        for (unsigned int i=0; i<ss.size(); ++i)
        {
            Spring& s = ss[i];
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
        this->springs.endEdit();
    }
}

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_MESHSPRINGFORCEFIELD_INL */
