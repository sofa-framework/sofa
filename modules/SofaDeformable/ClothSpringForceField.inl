/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_CLOTHSPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_CLOTHSPRINGFORCEFIELD_INL

#include <SofaDeformable/ClothSpringForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template<class DataTypes>
void ClothSpringForceField<DataTypes>::addSpring( unsigned a, unsigned b, std::set<IndexPair>& springSet )
{
    IndexPair ab(a<b?a:b, a<b?b:a);
    if (springSet.find(ab) != springSet.end()) return;
    springSet.insert(ab);
    const VecCoord& x =this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    Real s = (Real)this->ks.getValue();
    Real d = (Real)this->kd.getValue();
    Real l = (x[a]-x[b]).norm();
    Spring spring(a,b,s,d,l,true); // elongation only spring
    this->SpringForceField<DataTypes>::addSpring(spring);
}


template<class DataTypes>
void ClothSpringForceField<DataTypes>::init()
{
    // A point is connected by a spring to all others which share a common face

    std::map< IndexPair, IndexPair > edgeMap;
    std::set< IndexPair > springSet;
    sofa::core::topology::BaseMeshTopology* topology = this->getContext()->getMeshTopology();
    assert( topology );

    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = topology->getQuads();
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = topology->getTriangles();

    for(int i=0; i<topology->getNbPoints(); ++i)
    {
        // quads
        sofa::core::topology::BaseMeshTopology::QuadsAroundVertex quadsAroundVertex = topology->getQuadsAroundVertex(i);
        for( unsigned j= 0; j<quadsAroundVertex.size(); ++j )
        {
            const sofa::core::topology::BaseMeshTopology::Quad& face = quads[quadsAroundVertex[j]];
            for(unsigned k=0; k<face.size();k++)
            {
                if(face[k]!=i)
                    addSpring(i, face[k],springSet);
            }
        }
        
        //triangles
        sofa::core::topology::BaseMeshTopology::TrianglesAroundVertex trianglesAroundVertex = topology->getTrianglesAroundVertex(i);
        for( unsigned j= 0; j<trianglesAroundVertex.size(); ++j )
        {
            const sofa::core::topology::BaseMeshTopology::Triangle& face = triangles[trianglesAroundVertex[j]];
            for(unsigned k=0; k<face.size();k++)
            {
                if(face[k]!=i)
                    addSpring(i, face[k],springSet);
            }
        }
    }

    // init the parent class
    StiffSpringForceField<DataTypes>::init();
}


} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_QUADBENDINGSPRINGS_INL */
