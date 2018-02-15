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


} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_MESHSPRINGFORCEFIELD_INL */
