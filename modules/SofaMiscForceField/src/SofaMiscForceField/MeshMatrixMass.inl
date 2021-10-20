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

#include <SofaMiscForceField/MeshMatrixMass.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaBaseMechanics/AddMToMatrixFunctor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/vector.h>
#include <SofaBaseTopology/CommonAlgorithms.h>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.h>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <numeric>

namespace sofa::component::mass
{
using namespace sofa::core::topology;

template <class DataTypes, class MassType>
MeshMatrixMass<DataTypes, MassType>::MeshMatrixMass()
    : d_massDensity( initData(&d_massDensity, "massDensity", "Specify real and strictly positive value(s) for the mass density. \n"
                                                             "If unspecified or wrongly set, the totalMass information is used.") )
    , d_totalMass( initData(&d_totalMass, Real(1.0), "totalMass", "Specify the total mass resulting from all particles. \n"
                                                                  "If unspecified or wrongly set, the default value is used: totalMass = 1.0") )
    , d_vertexMass( initData(&d_vertexMass, "vertexMass", "internal values of the particles masses on vertices, supporting topological changes") )
    , d_edgeMass( initData(&d_edgeMass, "edgeMass", "internal values of the particles masses on edges, supporting topological changes") )
    , d_computeMassOnRest(initData(&d_computeMassOnRest, false, "computeMassOnRest", "If true, the mass of every element is computed based on the rest position rather than the position"))
    , d_showCenterOfGravity( initData(&d_showCenterOfGravity, false, "showGravityCenter", "display the center of gravity of the system" ) )
    , d_showAxisSize( initData(&d_showAxisSize, Real(1.0), "showAxisSizeFactor", "factor length of the axis displayed (only used for rigids)" ) )
    , d_lumping( initData(&d_lumping, false, "lumping","boolean if you need to use a lumped mass matrix") )
    , d_printMass( initData(&d_printMass, false, "printMass","boolean if you want to check the mass conservation") )
    , f_graph( initData(&f_graph,"graph","Graph of the controlled potential") )
    , l_topology(initLink("topology", "link to the topology container"))
    , m_massTopologyType(TopologyElementType::UNKNOWN)
    , m_topology(nullptr)
{
    f_graph.setWidget("graph");

    /// Internal data, not supposed to be accessed by the user
}

template <class DataTypes, class MassType>
MeshMatrixMass<DataTypes, MassType>::~MeshMatrixMass()
{

}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyVertexMassCreation(Index, MassType & VertexMass,
    const core::topology::BaseMeshTopology::Point&,
    const sofa::type::vector< Index > &,
    const sofa::type::vector< double >&)
{
    VertexMass = 0;
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyEdgeMassCreation(Index, MassType & EdgeMass,
    const core::topology::BaseMeshTopology::Edge&,
    const sofa::type::vector< Index > &,
    const sofa::type::vector< double >&)
{
    EdgeMass = 0;
}


template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyVertexMassDestruction(Index id, MassType& VertexMass)
{
    SOFA_UNUSED(id);
    helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);
    totalMass -= VertexMass;
}


template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyEdgeMassDestruction(Index id, MassType& EdgeMass)
{
    SOFA_UNUSED(id);
    if(!isLumped())
    {
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);
        totalMass -= EdgeMass;
    }
}



// -------------------------------------------------------
// ------- Triangle Creation/Destruction functions -------
// -------------------------------------------------------
//{

/// Creation fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyVertexMassTriangleCreation(const sofa::type::vector< Index >& triangleAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Triangle >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< double > >& coefs)
{
    SOFA_UNUSED(elems);
    if (this->getMassTopologyType() == TopologyElementType::TRIANGLE)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses (d_vertexMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);        

        // update mass density vector
        sofa::Size nbMass = getMassDensity().size();
        sofa::Size nbTri = this->m_topology->getNbTriangles();
        if (nbMass < nbTri)
            addMassDensity(triangleAdded, ancestors, coefs);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<triangleAdded.size(); ++i)
        {
            // Get the triangle to be added
            const core::topology::BaseMeshTopology::Triangle &t = this->m_topology->getTriangle(triangleAdded[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(this->triangleGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[triangleAdded[i]] * this->triangleGeo->computeRestTriangleArea(triangleAdded[i]))/(typename DataTypes::Real(6.0));
                }
                else
                {
                    mass=(densityM[triangleAdded[i]] * this->triangleGeo->computeTriangleArea(triangleAdded[i]))/(typename DataTypes::Real(6.0));
                }
            }

            // Adding mass
            for (unsigned int j=0; j<3; ++j)
                VertexMasses[ t[j] ] += mass;

            // update total mass: 
            if(!this->isLumped())
            {
                totalMass += 3.0 * mass;
            }
            else
            {
                totalMass += 3.0 * mass * m_massLumpingCoeff;
            }
        }
    }
}

/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyEdgeMassTriangleCreation(const sofa::type::vector< Index >& triangleAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Triangle >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< double > >& coefs)
{
    SOFA_UNUSED(elems);
    if (this->getMassTopologyType() == TopologyElementType::TRIANGLE)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses ( d_edgeMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        // update mass density vector
        sofa::Size nbMass = getMassDensity().size();
        sofa::Size nbTri = this->m_topology->getNbTriangles();
        if (nbMass < nbTri)
            addMassDensity(triangleAdded, ancestors, coefs);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<triangleAdded.size(); ++i)
        {
            // Get the edgesInTriangle to be added
            const core::topology::BaseMeshTopology::EdgesInTriangle &te = this->m_topology->getEdgesInTriangle(triangleAdded[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(this->triangleGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[triangleAdded[i]] * this->triangleGeo->computeRestTriangleArea(triangleAdded[i]))/(typename DataTypes::Real(12.0));
                }
                else
                {
                    mass=(densityM[triangleAdded[i]] * this->triangleGeo->computeTriangleArea(triangleAdded[i]))/(typename DataTypes::Real(12.0));
                }
            }

            // Adding mass edges of concerne triangle
            for (unsigned int j=0; j<3; ++j)
                EdgeMasses[ te[j] ] += mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass += 3.0 * mass * 2.0; // x 2 because mass is actually splitted over half-edges
            }
        }
    }
}

/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyVertexMassTriangleDestruction(const sofa::type::vector< Index >& triangleRemoved)
{
    if (this->getMassTopologyType() == TopologyElementType::TRIANGLE)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses (d_vertexMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<triangleRemoved.size(); ++i)
        {
            // Get the triangle to be removed
            const core::topology::BaseMeshTopology::Triangle &t = this->m_topology->getTriangle(triangleRemoved[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(this->triangleGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[triangleRemoved[i]] * this->triangleGeo->computeRestTriangleArea(triangleRemoved[i]))/(typename DataTypes::Real(6.0));
                }
                else
                {
                    mass=(densityM[triangleRemoved[i]] * this->triangleGeo->computeTriangleArea(triangleRemoved[i]))/(typename DataTypes::Real(6.0));
                }
            }

            // Removing mass
            for (unsigned int j=0; j<3; ++j)
                VertexMasses[ t[j] ] -= mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass -= 3.0 * mass;
            }
            else
            {
                totalMass -= 3.0 * mass * m_massLumpingCoeff;
            }
        }
    }
}

/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyEdgeMassTriangleDestruction(const sofa::type::vector< Index >& triangleRemoved)
{
    if (this->getMassTopologyType() == TopologyElementType::TRIANGLE)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses (d_edgeMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<triangleRemoved.size(); ++i)
        {
            // Get the triangle to be removed
            const core::topology::BaseMeshTopology::EdgesInTriangle &te = this->m_topology->getEdgesInTriangle(triangleRemoved[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(this->triangleGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[triangleRemoved[i]] * this->triangleGeo->computeRestTriangleArea(triangleRemoved[i]))/(typename DataTypes::Real(12.0));
                }
                else
                {
                    mass=(densityM[triangleRemoved[i]] * this->triangleGeo->computeTriangleArea(triangleRemoved[i]))/(typename DataTypes::Real(12.0));
                }
            }

            // Removing mass edges of concerne triangle
            for (unsigned int j=0; j<3; ++j)
                EdgeMasses[ te[j] ] -= mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass -= 3.0 * mass * 2.0; // x 2 because mass is actually splitted over half-edges
            }
        }
    }
}

// }

// ---------------------------------------------------
// ------- Quad Creation/Destruction functions -------
// ---------------------------------------------------
//{

/// Creation fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyVertexMassQuadCreation(const sofa::type::vector< Index >& quadAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Quad >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< double > >& coefs)
{
    SOFA_UNUSED(elems);
    if (this->getMassTopologyType() == TopologyElementType::QUAD)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses ( d_vertexMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        // update mass density vector
        sofa::Size nbMass = getMassDensity().size();
        sofa::Size nbQ = this->m_topology->getNbQuads();
        if (nbMass < nbQ)
            addMassDensity(quadAdded, ancestors, coefs);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<quadAdded.size(); ++i)
        {
            // Get the quad to be added
            const core::topology::BaseMeshTopology::Quad &q = this->m_topology->getQuad(quadAdded[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(this->quadGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[quadAdded[i]] * this->quadGeo->computeRestQuadArea(quadAdded[i]))/(typename DataTypes::Real(8.0));
                }
                else
                {
                    mass=(densityM[quadAdded[i]] * this->quadGeo->computeQuadArea(quadAdded[i]))/(typename DataTypes::Real(8.0));
                }
            }

            // Adding mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ q[j] ] += mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass += 4.0 * mass;
            }
            else
            {
                totalMass += 4.0 * mass * m_massLumpingCoeff;
            }
        }
    }
}

/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyEdgeMassQuadCreation(const sofa::type::vector< Index >& quadAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Quad >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< double > >& coefs)
{
    SOFA_UNUSED(elems);

    if (this->getMassTopologyType() == TopologyElementType::QUAD)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses ( d_edgeMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);
        
        // update mass density vector
        sofa::Size nbMass = getMassDensity().size();
        sofa::Size nbQ = this->m_topology->getNbQuads();
        if (nbMass < nbQ)
            addMassDensity(quadAdded, ancestors, coefs);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<quadAdded.size(); ++i)
        {
            // Get the EdgesInQuad to be added
            const core::topology::BaseMeshTopology::EdgesInQuad &qe = this->m_topology->getEdgesInQuad(quadAdded[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(this->quadGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[quadAdded[i]] * this->quadGeo->computeRestQuadArea(quadAdded[i]))/(typename DataTypes::Real(16.0));
                }
                else
                {
                    mass=(densityM[quadAdded[i]] * this->quadGeo->computeQuadArea(quadAdded[i]))/(typename DataTypes::Real(16.0));
                }
            }

            // Adding mass edges of concerne quad
            for (unsigned int j=0; j<4; ++j)
                EdgeMasses[ qe[j] ] += mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass += 4.0 * mass * 2.0; // x 2 because mass is actually splitted over half-edges
            }
        }
    }
}

/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyVertexMassQuadDestruction(const sofa::type::vector< Index >& quadRemoved)
{
    if (this->getMassTopologyType() == TopologyElementType::QUAD)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses ( d_vertexMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<quadRemoved.size(); ++i)
        {
            // Get the quad to be removed
            const core::topology::BaseMeshTopology::Quad &q = this->m_topology->getQuad(quadRemoved[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(this->quadGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[quadRemoved[i]] * this->quadGeo->computeRestQuadArea(quadRemoved[i]))/(typename DataTypes::Real(8.0));
                }
                else
                {
                    mass=(densityM[quadRemoved[i]] * this->quadGeo->computeQuadArea(quadRemoved[i]))/(typename DataTypes::Real(8.0));
                }
            }

            // Removing mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ q[j] ] -= mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass -= 4.0 * mass;
            }
            else
            {
                totalMass -= 4.0 * mass * m_massLumpingCoeff;
            }
        }
    }
}

/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyEdgeMassQuadDestruction(const sofa::type::vector< Index >& quadRemoved)
{
    if (this->getMassTopologyType() == TopologyElementType::QUAD)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses ( d_edgeMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<quadRemoved.size(); ++i)
        {
            // Get the EdgesInQuad to be removed
            const core::topology::BaseMeshTopology::EdgesInQuad &qe = this->m_topology->getEdgesInQuad(quadRemoved[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(this->quadGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[quadRemoved[i]] * this->quadGeo->computeRestQuadArea(quadRemoved[i]))/(typename DataTypes::Real(16.0));
                }
                else
                {
                    mass=(densityM[quadRemoved[i]] * this->quadGeo->computeQuadArea(quadRemoved[i]))/(typename DataTypes::Real(16.0));
                }
            }

            // Removing mass edges of concerne quad
            for (unsigned int j=0; j<4; ++j)
                EdgeMasses[ qe[j] ] -= mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass -= 4.0 * mass * 2.0; // x 2 because mass is actually splitted over half-edges
            }
        }
    }
}

// }



// ----------------------------------------------------------
// ------- Tetrahedron Creation/Destruction functions -------
// ----------------------------------------------------------
//{

/// Creation fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyVertexMassTetrahedronCreation(const sofa::type::vector< Index >& tetrahedronAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Tetrahedron >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< double > >& coefs)
{
    SOFA_UNUSED(elems);
    if (this->getMassTopologyType() == TopologyElementType::TETRAHEDRON)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses ( d_vertexMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        // update mass density vector
        sofa::Size nbMass = getMassDensity().size();
        sofa::Size nbT = this->m_topology->getNbTetrahedra();
        if (nbMass < nbT)
            addMassDensity(tetrahedronAdded, ancestors, coefs);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<tetrahedronAdded.size(); ++i)
        {
            // Get the tetrahedron to be added
            const core::topology::BaseMeshTopology::Tetrahedron &t = this->m_topology->getTetrahedron(tetrahedronAdded[i]);

            // Compute rest mass of conserne tetrahedron = density * tetrahedron volume.
            if(this->tetraGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[tetrahedronAdded[i]] * this->tetraGeo->computeRestTetrahedronVolume(tetrahedronAdded[i]))/(typename DataTypes::Real(10.0));
                }
                else
                {
                    mass=(densityM[tetrahedronAdded[i]] * this->tetraGeo->computeTetrahedronVolume(tetrahedronAdded[i]))/(typename DataTypes::Real(10.0));
                }
            }

            // Adding mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ t[j] ] += mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass += 4.0 * mass;
            }
            else
            {
                totalMass += 4.0 * mass * m_massLumpingCoeff;
            }
        }
    }
}

/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyEdgeMassTetrahedronCreation(const sofa::type::vector< Index >& tetrahedronAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Tetrahedron >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< double > >& coefs)
{
    SOFA_UNUSED(elems);
    if (this->getMassTopologyType() == TopologyElementType::TETRAHEDRON)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses ( d_edgeMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);
        
        // update mass density vector
        sofa::Size nbMass = getMassDensity().size();
        sofa::Size nbT = this->m_topology->getNbTetrahedra();
        if (nbMass < nbT)
            addMassDensity(tetrahedronAdded, ancestors, coefs);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<tetrahedronAdded.size(); ++i)
        {
            // Get the edgesInTetrahedron to be added
            const core::topology::BaseMeshTopology::EdgesInTetrahedron &te = this->m_topology->getEdgesInTetrahedron(tetrahedronAdded[i]);

            // Compute rest mass of conserne triangle = density * tetrahedron volume.
            if(this->tetraGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[tetrahedronAdded[i]] * this->tetraGeo->computeRestTetrahedronVolume(tetrahedronAdded[i]))/(typename DataTypes::Real(20.0));
                }
                else
                {
                    mass=(densityM[tetrahedronAdded[i]] * this->tetraGeo->computeTetrahedronVolume(tetrahedronAdded[i]))/(typename DataTypes::Real(20.0));
                }
            }

            // Adding mass edges of concerne triangle
            for (unsigned int j=0; j<6; ++j)
                EdgeMasses[ te[j] ] += mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass += 6.0 * mass * 2.0; // x 2 because mass is actually splitted over half-edges
            }
        }
    }
}

/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyVertexMassTetrahedronDestruction(const sofa::type::vector< Index >& tetrahedronRemoved)
{
    if (this->getMassTopologyType() == TopologyElementType::TETRAHEDRON)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses ( d_vertexMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<tetrahedronRemoved.size(); ++i)
        {
            // Get the tetrahedron to be removed
            const core::topology::BaseMeshTopology::Tetrahedron &t = this->m_topology->getTetrahedron(tetrahedronRemoved[i]);

            // Compute rest mass of conserne tetrahedron = density * tetrahedron volume.
            if(this->tetraGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[tetrahedronRemoved[i]] * this->tetraGeo->computeRestTetrahedronVolume(tetrahedronRemoved[i]))/(typename DataTypes::Real(10.0));
                }
                else
                {
                    mass=(densityM[tetrahedronRemoved[i]] * this->tetraGeo->computeTetrahedronVolume(tetrahedronRemoved[i]))/(typename DataTypes::Real(10.0));
                }
            }

            // Removing mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ t[j] ] -= mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass -= 4.0 * mass;
            }
            else
            {
                totalMass -= 4.0 * mass * m_massLumpingCoeff;
            }
        }
    }
}

/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyEdgeMassTetrahedronDestruction(const sofa::type::vector< Index >& tetrahedronRemoved)
{
    if (this->getMassTopologyType() == TopologyElementType::TETRAHEDRON)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses ( d_edgeMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<tetrahedronRemoved.size(); ++i)
        {
            // Get the edgesInTetrahedron to be removed
            const core::topology::BaseMeshTopology::EdgesInTetrahedron &te = this->m_topology->getEdgesInTetrahedron(tetrahedronRemoved[i]);

            // Compute rest mass of conserne triangle = density * tetrahedron volume.
            if(this->tetraGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[tetrahedronRemoved[i]] * this->tetraGeo->computeRestTetrahedronVolume(tetrahedronRemoved[i]))/(typename DataTypes::Real(20.0));
                }
                else
                {
                    mass=(densityM[tetrahedronRemoved[i]] * this->tetraGeo->computeTetrahedronVolume(tetrahedronRemoved[i]))/(typename DataTypes::Real(20.0));
                }
            }

            // Removing mass edges of concerne triangle
            for (unsigned int j=0; j<6; ++j)
                EdgeMasses[ te[j] ] -= mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass -= 6.0 * mass * 2.0; // x 2 because mass is actually splitted over half-edges
            }
        }
    }
}

// }


// ---------------------------------------------------------
// ------- Hexahedron Creation/Destruction functions -------
// ---------------------------------------------------------
//{

/// Creation fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyVertexMassHexahedronCreation(const sofa::type::vector< Index >& hexahedronAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Hexahedron >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< double > >& coefs)
{
    SOFA_UNUSED(elems);
    if (this->getMassTopologyType() == TopologyElementType::HEXAHEDRON)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses ( d_vertexMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);
        
        // update mass density vector
        sofa::Size nbMass = getMassDensity().size();
        sofa::Size nbT = this->m_topology->getNbHexahedra();
        if (nbMass < nbT)
            addMassDensity(hexahedronAdded, ancestors, coefs);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<hexahedronAdded.size(); ++i)
        {
            // Get the hexahedron to be added
            const core::topology::BaseMeshTopology::Hexahedron &h = this->m_topology->getHexahedron(hexahedronAdded[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(this->hexaGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[hexahedronAdded[i]] * this->hexaGeo->computeRestHexahedronVolume(hexahedronAdded[i]))/(typename DataTypes::Real(20.0));
                }
                else
                {
                    mass=(densityM[hexahedronAdded[i]] * this->hexaGeo->computeHexahedronVolume(hexahedronAdded[i]))/(typename DataTypes::Real(20.0));
                }
            }

            // Adding mass
            for (unsigned int j=0; j<8; ++j)
                VertexMasses[ h[j] ] += mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass += 8.0 * mass;
            }
            else
            {
                totalMass += 8.0 * mass * m_massLumpingCoeff;
            }
        }
    }
}

/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyEdgeMassHexahedronCreation(const sofa::type::vector< Index >& hexahedronAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Hexahedron >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< double > >& coefs)
{
    SOFA_UNUSED(elems);
    if (this->getMassTopologyType() == TopologyElementType::HEXAHEDRON)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses ( d_edgeMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        // update mass density vector
        sofa::Size nbMass = getMassDensity().size();
        sofa::Size nbT = this->m_topology->getNbHexahedra();
        if (nbMass < nbT)
            addMassDensity(hexahedronAdded, ancestors, coefs);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<hexahedronAdded.size(); ++i)
        {
            // Get the EdgesInHexahedron to be added
            const core::topology::BaseMeshTopology::EdgesInHexahedron &he = this->m_topology->getEdgesInHexahedron(hexahedronAdded[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(this->hexaGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[hexahedronAdded[i]] * this->hexaGeo->computeRestHexahedronVolume(hexahedronAdded[i]))/(typename DataTypes::Real(40.0));
                }
                else
                {
                    mass=(densityM[hexahedronAdded[i]] * this->hexaGeo->computeHexahedronVolume(hexahedronAdded[i]))/(typename DataTypes::Real(40.0));
                }
            }

            // Adding mass edges of concerne triangle
            for (unsigned int j=0; j<12; ++j)
                EdgeMasses[ he[j] ] += mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass += 12.0 * mass * 2.0; // x 2 because mass is actually splitted over half-edges
            }
        }
    }
}

/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyVertexMassHexahedronDestruction(const sofa::type::vector< Index >& hexahedronRemoved)
{
    if (this->getMassTopologyType() == TopologyElementType::HEXAHEDRON)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses ( d_vertexMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<hexahedronRemoved.size(); ++i)
        {
            // Get the hexahedron to be removed
            const core::topology::BaseMeshTopology::Hexahedron &h = this->m_topology->getHexahedron(hexahedronRemoved[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(this->hexaGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[hexahedronRemoved[i]] * this->hexaGeo->computeRestHexahedronVolume(hexahedronRemoved[i]))/(typename DataTypes::Real(20.0));
                }
                else
                {
                    mass=(densityM[hexahedronRemoved[i]] * this->hexaGeo->computeHexahedronVolume(hexahedronRemoved[i]))/(typename DataTypes::Real(20.0));
                }
            }

            // Removing mass
            for (unsigned int j=0; j<8; ++j)
                VertexMasses[ h[j] ] -= mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass -= 8.0 * mass;
            }
            else
            {
                totalMass -= 8.0 * mass * m_massLumpingCoeff;
            }
        }
    }
}

/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::applyEdgeMassHexahedronDestruction(const sofa::type::vector< Index >& hexahedronRemoved)
{
    if (this->getMassTopologyType() == TopologyElementType::HEXAHEDRON)
    {
        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses ( d_edgeMass );
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        // Initialisation
        const type::vector<Real> densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<hexahedronRemoved.size(); ++i)
        {
            // Get the EdgesInHexahedron to be removed
            const core::topology::BaseMeshTopology::EdgesInHexahedron &he = this->m_topology->getEdgesInHexahedron(hexahedronRemoved[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(this->hexaGeo)
            {
                if(d_computeMassOnRest.getValue())
                {
                    mass=(densityM[hexahedronRemoved[i]] * this->hexaGeo->computeRestHexahedronVolume(hexahedronRemoved[i]))/(typename DataTypes::Real(40.0));
                }
                else
                {
                    mass=(densityM[hexahedronRemoved[i]] * this->hexaGeo->computeHexahedronVolume(hexahedronRemoved[i]))/(typename DataTypes::Real(40.0));
                }
            }

            // Removing mass edges of concerne triangle
            for (unsigned int j=0; j<12; ++j)
                EdgeMasses[ he[j] ] -= mass;

            // update total mass
            if(!this->isLumped())
            {
                totalMass -= 12.0 * mass * 2.0; // x 2 because mass is actually splitted over half-edges
            }
        }
    }
}


// }

using sofa::core::topology::TopologyElementType;

template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::init()
{
    m_massLumpingCoeff = 0.0;

    TopologyElementType topoType = checkTopology();
    if(topoType == TopologyElementType::POINT)
    {
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    Inherited::init();
    initTopologyHandlers(topoType);

    massInitialization();

    //Reset the graph
    f_graph.beginEdit()->clear();
    f_graph.endEdit();

    // add data to tracker
    this->trackInternalData(d_vertexMass);
    this->trackInternalData(d_edgeMass);
    this->trackInternalData(d_massDensity);
    this->trackInternalData(d_totalMass);

    //Function for GPU-CUDA version only
    this->copyVertexMass();
}


template <class DataTypes, class MassType>
sofa::core::topology::TopologyElementType MeshMatrixMass<DataTypes, MassType>::checkTopology()
{
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
        return sofa::core::topology::TopologyElementType::POINT;
    }

    this->getContext()->get(edgeGeo);
    this->getContext()->get(triangleGeo);
    this->getContext()->get(quadGeo);
    this->getContext()->get(tetraGeo);
    this->getContext()->get(hexaGeo);
    
    if (m_topology->getNbHexahedra() > 0)
    {
        if(!hexaGeo)
        {
            msg_error() << "Hexahedron topology but no geometry algorithms found. Add the component HexahedronSetGeometryAlgorithms.";
            return TopologyElementType::POINT;
        }
        else
        {
            msg_info() << "Hexahedral topology found.";
            return TopologyElementType::HEXAHEDRON;
        }
    }
    else if (m_topology->getNbTetrahedra() > 0)
    {
        if(!tetraGeo)
        {
            msg_error() << "Tetrahedron topology but no geometry algorithms found. Add the component TetrahedronSetGeometryAlgorithms.";
            return TopologyElementType::POINT;
        }
        else
        {
            msg_info() << "Tetrahedral topology found.";
            return TopologyElementType::TETRAHEDRON;
        }
    }
    else if (m_topology->getNbQuads() > 0)
    {
        if(!quadGeo)
        {
            msg_error() << "Quad topology but no geometry algorithms found. Add the component QuadSetGeometryAlgorithms.";
            return TopologyElementType::POINT;
        }
        else
        {
            msg_info() << "Quad topology found.";
            return TopologyElementType::QUAD;
        }
    }
    else if (m_topology->getNbTriangles() > 0)
    {
        if(!triangleGeo)
        {
            msg_error() << "Triangle topology but no geometry algorithms found. Add the component TriangleSetGeometryAlgorithms.";
            return TopologyElementType::POINT;
        }
        else
        {
            msg_info() << "Triangular topology found.";
            return TopologyElementType::TRIANGLE;
        }
    }
    else if (m_topology->getNbEdges() > 0)
    {
        if(!edgeGeo)
        {
            msg_error() << "Edge topology but no geometry algorithms found. Add the component EdgeSetGeometryAlgorithms.";
            return TopologyElementType::POINT;
        }
        else
        {
            msg_info() << "Edge topology found.";
            return TopologyElementType::EDGE;
        }
    }
    else
    {
        msg_error() << "Topology empty.";
        return TopologyElementType::POINT;
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::initTopologyHandlers(sofa::core::topology::TopologyElementType topologyType)
{
    // add the functions to handle topology changes for Vertex informations
    d_vertexMass.createTopologyHandler(m_topology);
    d_vertexMass.setCreationCallback([this](Index pointIndex, MassType& m,
        const core::topology::BaseMeshTopology::Point& point,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< double >& coefs)
    {
        applyVertexMassCreation(pointIndex, m, point, ancestors, coefs);
    });
    d_vertexMass.setDestructionCallback([this](Index pointIndex, MassType& m)
    {
        applyVertexMassDestruction(pointIndex, m);
    });

    // add the functions to handle topology changes for Edge informations
    d_edgeMass.createTopologyHandler(m_topology);
    d_edgeMass.setCreationCallback([this](Index edgeIndex, MassType& EdgeMass,
        const core::topology::BaseMeshTopology::Edge& edge,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< double >& coefs)
    {
        applyEdgeMassCreation(edgeIndex, EdgeMass, edge, ancestors, coefs);
    });
    d_edgeMass.setDestructionCallback([this](Index edgeIndex, MassType& m)
    {
        applyEdgeMassDestruction(edgeIndex, m);
    });

    // register engines to the corresponding toplogy containers depending on current topology type
    bool hasTriangles = false;
    bool hasQuads = false;
    if (topologyType == TopologyElementType::HEXAHEDRON)
    {
        d_vertexMass.linkToHexahedronDataArray();
        d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::HEXAHEDRAADDED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::HexahedraAdded* hAdd = static_cast<const core::topology::HexahedraAdded*>(eventTopo);
            applyVertexMassHexahedronCreation(hAdd->getIndexArray(), hAdd->getElementArray(), hAdd->ancestorsList, hAdd->coefs);
        });
        d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::HEXAHEDRAREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::HexahedraRemoved* hRemove = static_cast<const core::topology::HexahedraRemoved*>(eventTopo);
            applyVertexMassHexahedronDestruction(hRemove->getArray());
        });

        d_edgeMass.linkToHexahedronDataArray();
        d_edgeMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::HEXAHEDRAADDED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::HexahedraAdded* hAdd = static_cast<const core::topology::HexahedraAdded*>(eventTopo);
            applyEdgeMassHexahedronCreation(hAdd->getIndexArray(), hAdd->getElementArray(), hAdd->ancestorsList, hAdd->coefs);
        });
        d_edgeMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::HEXAHEDRAREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::HexahedraRemoved* hRemove = static_cast<const core::topology::HexahedraRemoved*>(eventTopo);
            applyEdgeMassHexahedronDestruction(hRemove->getArray());
        });

        hasQuads = true; // hexahedron imply quads
    }
    else if (topologyType == TopologyElementType::TETRAHEDRON)
    {
        d_vertexMass.linkToTetrahedronDataArray();
        d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TETRAHEDRAADDED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::TetrahedraAdded* tAdd = static_cast<const core::topology::TetrahedraAdded*>(eventTopo);
            applyVertexMassTetrahedronCreation(tAdd->getIndexArray(), tAdd->getElementArray(), tAdd->ancestorsList, tAdd->coefs);
        });
        d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TETRAHEDRAREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::TetrahedraRemoved* tRemove = static_cast<const core::topology::TetrahedraRemoved*>(eventTopo);
            applyVertexMassTetrahedronDestruction(tRemove->getArray());
        });

        d_edgeMass.linkToTetrahedronDataArray();
        d_edgeMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TETRAHEDRAADDED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::TetrahedraAdded* tAdd = static_cast<const core::topology::TetrahedraAdded*>(eventTopo);
            applyEdgeMassTetrahedronCreation(tAdd->getIndexArray(), tAdd->getElementArray(), tAdd->ancestorsList, tAdd->coefs);
        });
        d_edgeMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TETRAHEDRAREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::TetrahedraRemoved* tRemove = static_cast<const core::topology::TetrahedraRemoved*>(eventTopo);
            applyEdgeMassTetrahedronDestruction(tRemove->getArray());
        });

        hasTriangles = true; // Tetrahedron imply triangles
    }

    if (topologyType == TopologyElementType::QUAD || hasQuads)
    {
        d_vertexMass.linkToQuadDataArray();
        d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::QUADSADDED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::QuadsAdded* qAdd = static_cast<const core::topology::QuadsAdded*>(eventTopo);
            applyVertexMassQuadCreation(qAdd->getIndexArray(), qAdd->getElementArray(), qAdd->ancestorsList, qAdd->coefs);
        });
        d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::QUADSREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::QuadsRemoved* qRemove = static_cast<const core::topology::QuadsRemoved*>(eventTopo);
            applyVertexMassQuadDestruction(qRemove->getArray());
        });

        d_edgeMass.linkToQuadDataArray();
        d_edgeMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::QUADSADDED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::QuadsAdded* qAdd = static_cast<const core::topology::QuadsAdded*>(eventTopo);
            applyEdgeMassQuadCreation(qAdd->getIndexArray(), qAdd->getElementArray(), qAdd->ancestorsList, qAdd->coefs);
        });
        d_edgeMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::QUADSREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::QuadsRemoved* qRemove = static_cast<const core::topology::QuadsRemoved*>(eventTopo);
            applyEdgeMassQuadDestruction(qRemove->getArray());
        });
    }

    if (topologyType == TopologyElementType::TRIANGLE || hasTriangles)
    {
        d_vertexMass.linkToTriangleDataArray();
        d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TRIANGLESADDED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::TrianglesAdded* tAdd = static_cast<const core::topology::TrianglesAdded*>(eventTopo);
            applyVertexMassTriangleCreation(tAdd->getIndexArray(), tAdd->getElementArray(), tAdd->ancestorsList, tAdd->coefs);
        });
        d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TRIANGLESREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::TrianglesRemoved* tRemove = static_cast<const core::topology::TrianglesRemoved*>(eventTopo);
            applyVertexMassTriangleDestruction(tRemove->getArray());
        });

        d_edgeMass.linkToTriangleDataArray();
        d_edgeMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TRIANGLESADDED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::TrianglesAdded* tAdd = static_cast<const core::topology::TrianglesAdded*>(eventTopo);
            applyEdgeMassTriangleCreation(tAdd->getIndexArray(), tAdd->getElementArray(), tAdd->ancestorsList, tAdd->coefs);
        });
        d_edgeMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TRIANGLESREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::TrianglesRemoved* tRemove = static_cast<const core::topology::TrianglesRemoved*>(eventTopo);
            applyEdgeMassTriangleDestruction(tRemove->getArray());
        });
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::massInitialization()
{
    //Mass initialization process
    if(d_vertexMass.isSet() || d_massDensity.isSet() || d_totalMass.isSet() )
    {
        //totalMass data is prioritary on vertexMass and massDensity
        if (d_totalMass.isSet())
        {
            if(d_vertexMass.isSet() || d_massDensity.isSet())
            {
                msg_warning(this) << "totalMass value overriding other mass information (vertexMass or massDensity).\n"
                                  << "To remove this warning you need to define only one single mass information data field.";
            }
            checkTotalMassInit();
            initFromTotalMass();
        }
        //massDensity is secondly considered
        else if(d_massDensity.isSet())
        {
            if(d_vertexMass.isSet())
            {
                msg_warning(this) << "massDensity value overriding the value of the attribute vertexMass.\n"
                                  << "To remove this warning you need to set either vertexMass or massDensity data field, but not both.";
            }
            if(!checkMassDensity())
            {
                checkTotalMassInit();
                initFromTotalMass();
            }
            else
            {
                initFromMassDensity();
            }
        }
        //finally, the vertexMass is used
        else if(d_vertexMass.isSet())
        {
            if(d_edgeMass.isSet())
            {
                if(!checkVertexMass() || !checkEdgeMass() )
                {
                    checkTotalMassInit();
                    initFromTotalMass();
                }
                else
                {
                    initFromVertexAndEdgeMass();
                }
            }
            else if(isLumped() && !d_edgeMass.isSet())
            {
                if(!checkVertexMass())
                {
                    checkTotalMassInit();
                    initFromTotalMass();
                }
                else
                {
                    initFromVertexMass();
                }
            }
            else
            {
                msg_error() << "Initialization using vertexMass requires the lumping option or the edgeMass information";
                checkTotalMassInit();
                initFromTotalMass();
            }
        }
    }
    // if no mass information provided, default initialization uses totalMass
    else
    {
        msg_info() << "No information about the mass is given." << msgendl
                      "Default : totalMass = 1.0";
        checkTotalMassInit();
        initFromTotalMass();
    }


    //Info post-init
    printMass();
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::printMass()
{
    if (this->f_printLog.getValue() == false)
        return;

    //Info post-init
    const MassVector &vertexM = d_vertexMass.getValue();
    const MassVector &mDensity = d_massDensity.getValue();

    Real average_vertex = 0.0;
    Real min_vertex = std::numeric_limits<Real>::max();
    Real max_vertex = 0.0;
    Real average_density = 0.0;
    Real min_density = std::numeric_limits<Real>::max();
    Real max_density = 0.0;

    for(unsigned int i=0; i<vertexM.size(); i++)
    {
        average_vertex += vertexM[i];
        if(vertexM[i]<min_vertex)
            min_vertex = vertexM[i];
        if(vertexM[i]>max_vertex)
            max_vertex = vertexM[i];
    }
    if(vertexM.size() > 0)
    {
        average_vertex /= Real(vertexM.size());
    }

    for(unsigned int i=0; i<mDensity.size(); i++)
    {
        average_density += mDensity[i];
        if(mDensity[i]<min_density)
            min_density = mDensity[i];
        if(mDensity[i]>max_density)
            max_density = mDensity[i];
    }
    if(mDensity.size() > 0)
    {
        average_density /= Real(mDensity.size());
    }

    msg_info() << "mass information computed :" << msgendl
               << "totalMass   = " << d_totalMass.getValue() << msgendl
               << "mean massDensity [min,max] = " << average_density << " [" << min_density << "," <<  max_density <<"]" << msgendl
               << "mean vertexMass [min,max] = " << average_vertex << " [" << min_vertex << "," <<  max_vertex <<"]";
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::computeMass()
{
    // resize array
    clear();

    // prepare to store info in the vertex array
    helper::WriteAccessor < Data < MassVector > > my_vertexMassInfo = d_vertexMass;
    helper::WriteAccessor < Data < MassVector > > my_edgeMassInfo = d_edgeMass;

    unsigned int ndof = this->mstate->getSize();
    unsigned int nbEdges=m_topology->getNbEdges();
    const type::vector<core::topology::BaseMeshTopology::Edge>& edges = m_topology->getEdges();

    my_vertexMassInfo.resize(ndof);
    my_edgeMassInfo.resize(nbEdges);

    const type::vector< Index > emptyAncestor;
    const type::vector< double > emptyCoefficient;
    const type::vector< type::vector< Index > > emptyAncestors;
    const type::vector< type::vector< double > > emptyCoefficients;

    // set vertex tensor to 0
    for (Index i = 0; i<ndof; ++i)
        applyVertexMassCreation(i, my_vertexMassInfo[i], i, emptyAncestor, emptyCoefficient);

    // set edge tensor to 0
    for (Index i = 0; i<nbEdges; ++i)
        applyEdgeMassCreation(i, my_edgeMassInfo[i], edges[i], emptyAncestor, emptyCoefficient);

    // Create mass matrix depending on current Topology:
    if (m_topology->getNbHexahedra()>0 && hexaGeo)  // Hexahedron topology
    {
        // create vector tensor by calling the hexahedron creation function on the entire mesh
        sofa::type::vector<Index> hexahedraAdded;
        setMassTopologyType(TopologyElementType::HEXAHEDRON);
        size_t n = m_topology->getNbHexahedra();
        for (Index i = 0; i<n; ++i)
            hexahedraAdded.push_back(i);

        m_massLumpingCoeff = 2.5;
        if (!isLumped())
        {
            applyEdgeMassHexahedronCreation(hexahedraAdded, m_topology->getHexahedra(), emptyAncestors, emptyCoefficients);
        }
        
        applyVertexMassHexahedronCreation(hexahedraAdded, m_topology->getHexahedra(), emptyAncestors, emptyCoefficients);
    }
    else if (m_topology->getNbTetrahedra()>0 && tetraGeo)  // Tetrahedron topology
    {
        // create vector tensor by calling the tetrahedron creation function on the entire mesh
        sofa::type::vector<Index> tetrahedraAdded;
        setMassTopologyType(TopologyElementType::TETRAHEDRON);

        size_t n = m_topology->getNbTetrahedra();
        for (Index i = 0; i<n; ++i)
            tetrahedraAdded.push_back(i);

        m_massLumpingCoeff = 2.5;
        if (!isLumped())
        {
            applyEdgeMassTetrahedronCreation(tetrahedraAdded, m_topology->getTetrahedra(), emptyAncestors, emptyCoefficients);
        }

        applyVertexMassTetrahedronCreation(tetrahedraAdded, m_topology->getTetrahedra(), emptyAncestors, emptyCoefficients);
    }
    else if (m_topology->getNbQuads()>0 && quadGeo)  // Quad topology
    {
        // create vector tensor by calling the quad creation function on the entire mesh
        sofa::type::vector<Index> quadsAdded;
        setMassTopologyType(TopologyElementType::QUAD);

        size_t n = m_topology->getNbQuads();
        for (Index i = 0; i<n; ++i)
            quadsAdded.push_back(i);

        m_massLumpingCoeff = 2.0;
        if (!isLumped())
        {
            applyEdgeMassQuadCreation(quadsAdded, m_topology->getQuads(), emptyAncestors, emptyCoefficients);
        }

        applyVertexMassQuadCreation(quadsAdded, m_topology->getQuads(), emptyAncestors, emptyCoefficients);
    }
    else if (m_topology->getNbTriangles()>0 && triangleGeo) // Triangle topology
    {
        // create vector tensor by calling the triangle creation function on the entire mesh
        sofa::type::vector<Index> trianglesAdded;
        setMassTopologyType(TopologyElementType::TRIANGLE);

        size_t n = m_topology->getNbTriangles();
        for (Index i = 0; i<n; ++i)
            trianglesAdded.push_back(i);

        m_massLumpingCoeff = 2.0;
        if (!isLumped())
        {
            applyEdgeMassTriangleCreation(trianglesAdded, m_topology->getTriangles(), emptyAncestors, emptyCoefficients);
        }

        applyVertexMassTriangleCreation(trianglesAdded, m_topology->getTriangles(), emptyAncestors, emptyCoefficients);
    }

}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::reinit()
{
    // Now update is handled through the doUpdateInternal mechanism
    // called at each begin of step through the UpdateInternalDataVisitor
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::doUpdateInternal()
{
    if (this->hasDataChanged(d_totalMass))
    {
        if(checkTotalMass())
        {
            initFromTotalMass();
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
        }
        else
        {
            msg_error() << "doUpdateInternal: incorrect update from totalMass";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        }
    }
    else if(this->hasDataChanged(d_massDensity))
    {
        if(checkMassDensity())
        {
            initFromMassDensity();
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
        }
        else
        {
            msg_error() << "doUpdateInternal: incorrect update from massDensity";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        }
    }

    //Info post-init
    msg_info() << "mass information updated";
    printMass();
}


template <class DataTypes, class MassType>
bool MeshMatrixMass<DataTypes, MassType>::checkTotalMass()
{
    //Check for negative or null value, if wrongly set use the default value totalMass = 1.0
    if(d_totalMass.getValue() < 0.0)
    {
        msg_warning(this) << "totalMass data can not have a negative value.\n"
                          << "To remove this warning, you need to set a strictly positive value to the totalMass data";
        return false;
    }
    else
    {
        return true;
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::checkTotalMassInit()
{
    //Check for negative or null value, if wrongly set use the default value totalMass = 1.0
    if(!checkTotalMass())
    {
        msg_warning(this) << "Switching back to default values: totalMass = 1.0\n";
        d_totalMass.setValue(1.0) ;
    }
}


template <class DataTypes, class MassType>
bool MeshMatrixMass<DataTypes, MassType>::checkVertexMass()
{
    const sofa::type::vector<Real> &vertexMass = d_vertexMass.getValue();
    //Check size of the vector
    if (vertexMass.size() != size_t(m_topology->getNbPoints()))
    {
        msg_warning() << "Inconsistent size of vertexMass vector ("<< vertexMass.size() <<") compared to the DOFs size ("<< m_topology->getNbPoints() <<").";
        return false;
    }
    else
    {
        //Check that the vertexMass vector has only strictly positive values
        for(size_t i=0; i<vertexMass.size(); i++)
        {
            if(vertexMass[i]<0)
            {
                msg_warning() << "Negative value of vertexMass vector: vertexMass[" << i << "] = " << vertexMass[i];
                return false;
            }
        }
        return true;
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::initFromVertexMass()
{
    msg_info() << "vertexMass information is used";

    const sofa::type::vector<MassType> vertexMass = d_vertexMass.getValue();
    Real totalMassSave = 0.0;
    for(size_t i=0; i<vertexMass.size(); i++)
    {
        totalMassSave += vertexMass[i];
    }
    //Compute the volume
    setMassDensity(1.0);

    computeMass();

    helper::WriteAccessor<Data<MassVector> > vertexMassInfo = d_vertexMass;
    //Compute mass (which is equal to the volume since massDensity = 1.0)
    Real volume = 0.0;
    for(size_t i=0; i<vertexMassInfo.size(); i++)
    {
        volume += vertexMassInfo[i]*m_massLumpingCoeff;
        vertexMassInfo[i] = vertexMass[i];
    }

    //Update all computed values
    setMassDensity(Real(totalMassSave/volume));
    d_totalMass.setValue(totalMassSave);
}


template <class DataTypes, class MassType>
bool MeshMatrixMass<DataTypes, MassType>::checkEdgeMass()
{
    const sofa::type::vector<Real> edgeMass = d_edgeMass.getValue();
    //Check size of the vector
    if (edgeMass.size() != m_topology->getNbEdges())
    {
        msg_warning() << "Inconsistent size of vertexMass vector compared to the DOFs size.";
        return false;
    }
    else
    {
        //Check that the vertexMass vector has only strictly positive values
        for(size_t i=0; i<edgeMass.size(); i++)
        {
            if(edgeMass[i]<0)
            {
                msg_warning() << "Negative value of edgeMass vector: edgeMass[" << i << "] = " << edgeMass[i];
                return false;
            }
        }
        return true;
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::initFromVertexAndEdgeMass()
{
    msg_info() << "verteMass and edgeMass informations are used";

    const sofa::type::vector<MassType> vertexMass = d_vertexMass.getValue();
    const sofa::type::vector<MassType> edgeMass = d_edgeMass.getValue();
    Real totalMassSave = 0.0;
    for(size_t i=0; i<vertexMass.size(); i++)
    {
        totalMassSave += vertexMass[i];
    }
    for(size_t i=0; i<edgeMass.size(); i++)
    {
        totalMassSave += edgeMass[i];
    }
    //Compute the volume
    setMassDensity(1.0);

    computeMass();

    helper::WriteAccessor<Data<MassVector> > vertexMassInfo = d_vertexMass;
    helper::WriteAccessor<Data<MassVector> > edgeMassInfo = d_edgeMass;
    //Compute volume = mass since massDensity = 1.0
    Real volume = 0.0;
    for(size_t i=0; i<vertexMassInfo.size(); i++)
    {
        volume += vertexMassInfo[i]*m_massLumpingCoeff;
        vertexMassInfo[i] = vertexMass[i];
    }
    for(size_t i=0; i<edgeMass.size(); i++)
    {
        edgeMassInfo[i] = edgeMass[i];
    }
    //Update all computed values
    setMassDensity(Real(totalMassSave/volume));
    d_totalMass.setValue(totalMassSave);
}


template <class DataTypes, class MassType>
bool MeshMatrixMass<DataTypes, MassType>::checkMassDensity()
{
    const sofa::type::vector<Real> &massDensity = d_massDensity.getValue();
    Real density = massDensity[0];
    size_t sizeElements = 0;

    //Check size of the vector
    //Size = 1, homogeneous density
    //Otherwise, heterogeneous density
    if (m_topology->getNbHexahedra()>0 && hexaGeo)
    {
        sizeElements = m_topology->getNbHexahedra();

        if ( massDensity.size() != m_topology->getNbHexahedra() && massDensity.size() != 1)
        {
            msg_warning() << "Inconsistent size of massDensity = " << massDensity.size() << ", should be either 1 or " << m_topology->getNbHexahedra();
            return false;
        }
    }
    else if (m_topology->getNbTetrahedra()>0 && tetraGeo)
    {
        sizeElements = m_topology->getNbTetrahedra();

        if ( massDensity.size() != m_topology->getNbTetrahedra() && massDensity.size() != 1)
        {
            msg_warning() << "Inconsistent size of massDensity = " << massDensity.size() << ", should be either 1 or " << m_topology->getNbTetrahedra();
            return false;
        }
    }
    else if (m_topology->getNbQuads()>0 && quadGeo)
    {
        sizeElements = m_topology->getNbQuads();

        if ( massDensity.size() != m_topology->getNbQuads() && massDensity.size() != 1)
        {
            msg_warning() << "Inconsistent size of massDensity = " << massDensity.size() << ", should be either 1 or " << m_topology->getNbQuads();
            return false;
        }
    }
    else if (m_topology->getNbTriangles()>0 && triangleGeo)
    {
        sizeElements = m_topology->getNbTriangles();

        if ( massDensity.size() != m_topology->getNbTriangles() && massDensity.size() != 1)
        {
            msg_warning() << "Inconsistent size of massDensity = " << massDensity.size() << ", should be either 1 or " << m_topology->getNbTriangles();
            return false;
        }
    }


    //If single value of massDensity is given, propagate it to vector for all elements
    if(massDensity.size() == 1)
    {
        //Check that the massDensity is strictly positive
        if(density < 0.0)
        {
            msg_warning() << "Negative value of massDensity: massDensity = " << density;
            return false;
        }
        else
        {
            helper::WriteAccessor<Data<sofa::type::vector< Real > > > massDensityAccess = d_massDensity;
            massDensityAccess.clear();
            massDensityAccess.resize(sizeElements);

            for(size_t i=0; i<sizeElements; i++)
            {
                massDensityAccess[i] = density;
            }
            return true;
        }
    }
    //Vector input massDensity
    else
    {
        //Check that the massDensity has only strictly positive values
        for(size_t i=0; i<massDensity.size(); i++)
        {
            if(massDensity[i]<0)
            {
                msg_warning() << "Negative value of massDensity vector: massDensity[" << i << "] = " << massDensity[i];
                return false;
            }
        }
        return true;
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::initFromMassDensity()
{
    msg_info() << "massDensity information is used";

    // compute vertexMass and edgeMass vector with input density
    computeMass();

    const MassVector &vertexMassInfo = d_vertexMass.getValue();
    
    // Sum the mass per vertices and apply massLumping coef
    Real sumMass = std::accumulate(vertexMassInfo.begin(), vertexMassInfo.end(), Real(0));

    if (!isLumped())
    {
        // Add mass per edges if not lumped, *2 as it is added to both edge vertices
        helper::WriteAccessor<Data<MassVector> > edgeMass = d_edgeMass;
        sumMass += std::accumulate(edgeMass.begin(), edgeMass.end(), Real(0)) * 2.0;
    }
    else
    {
        sumMass *= m_massLumpingCoeff;
    }

    d_totalMass.setValue(sumMass);
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::initFromTotalMass()
{
    msg_info() << "totalMass information is used";

    // copy input totalMass
    Real totalMassCpy = d_totalMass.getValue();

    // set temporary density mass to 1 for every element
    setMassDensity(1.0);
    d_totalMass.setValue(0.0); // force totalMass to 0 to compute it with density = 1

    // compute vertexMass and edgeMass vector with density == 1. Will update totalMass as well
    computeMass();

    // total mass from geometry with density = 1
    const Real& sumMass = d_totalMass.getValue();

    // Set real density from sumMass found
    Real md = 1.0;
    if (sumMass > std::numeric_limits<typename DataTypes::Real>::epsilon())
        md = Real(totalMassCpy / sumMass);
    
    setMassDensity(md);

    // restore input total mass (was changed by computeMass())
    setTotalMass(totalMassCpy);

    // Apply the real density no nead to recompute vertexMass from the geometry as all vertices have the same density in this case
    helper::WriteAccessor<Data<MassVector> > vertexMass = d_vertexMass;
    for (auto& vm : vertexMass)
    {
        vm *= md;
    }

    // Same for edgeMass
    helper::WriteAccessor<Data<MassVector> > edgeMass = d_edgeMass;
    for (auto& em : edgeMass)
    {
        em *= md;
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::setVertexMass(sofa::type::vector< Real > vertexMass)
{
    const sofa::type::vector< Real > currentVertexMass = d_vertexMass.getValue();
    d_vertexMass.setValue(vertexMass);

    if(!checkVertexMass())
    {
        msg_warning() << "Given values to setVertexMass() are not correct.\n"
                      << "Previous values are used.";
        d_vertexMass.setValue(currentVertexMass);
    }
    else
    {
        d_vertexMass.setValue(vertexMass);
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::setMassDensity(sofa::type::vector< Real > massDensity)
{
    const sofa::type::vector< Real > currentMassDensity = d_massDensity.getValue();
    d_massDensity.setValue(massDensity);

    if(!checkMassDensity())
    {
        msg_warning() << "Given values to setMassDensity() are not correct.\n"
                      << "Previous values are used.";
        d_massDensity.setValue(currentMassDensity);
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::setMassDensity(Real massDensityValue)
{
    const sofa::type::vector< Real > currentMassDensity = d_massDensity.getValue();
    helper::WriteAccessor<Data<sofa::type::vector< Real > > > massDensity = d_massDensity;
    massDensity.clear();
    massDensity.resize(1);
    massDensity[0] = massDensityValue;

    if(!checkMassDensity())
    {
        msg_warning() << "Given values to setMassDensity() are not correct.\n"
                      << "Previous values are used.";
        d_massDensity.setValue(currentMassDensity);
    }
}

template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addMassDensity(const sofa::type::vector< Index >& indices, 
    const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
    const sofa::type::vector< sofa::type::vector< double > >& coefs)
{
    helper::WriteOnlyAccessor<Data<sofa::type::vector< Real > > > massDensity = d_massDensity;
    for (unsigned int id = 0; id < indices.size(); ++id)
    {
        // if no ancestor apply the first density of the vector
        if (ancestors[id].empty())
        {
            massDensity.push_back(massDensity[0]);
            continue;
        }

        // Accumulate mass of their neighbours
        Real massD = 0.0;
        for (unsigned int j = 0; j < ancestors[id].size(); ++j)
        {
            Index ancId = ancestors[id][j];
            double coef = coefs[id][j];
            massD += massDensity[ancId] * coef;
        }
        massDensity.push_back(massD);
    }
}

template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::setTotalMass(Real totalMass)
{
    const Real currentTotalMass = d_totalMass.getValue();
    d_totalMass.setValue(totalMass);
    if(!checkTotalMass())
    {
        msg_warning() << "Given value to setTotalMass() is not a strictly positive value\n"
                      << "Previous value is used: totalMass = " << currentTotalMass;
        d_totalMass.setValue(currentTotalMass);
    }
}


template <class DataTypes, class MassType>
const sofa::type::vector< typename MeshMatrixMass<DataTypes, MassType>::Real > &  MeshMatrixMass<DataTypes, MassType>::getVertexMass()
{
    return d_vertexMass.getValue();
}


template <class DataTypes, class MassType>
const sofa::type::vector< typename MeshMatrixMass<DataTypes, MassType>::Real > &  MeshMatrixMass<DataTypes, MassType>::getMassDensity()
{
    return d_massDensity.getValue();
}


template <class DataTypes, class MassType>
const typename MeshMatrixMass<DataTypes, MassType>::Real &MeshMatrixMass<DataTypes, MassType>::getTotalMass()
{
    return d_totalMass.getValue();
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::copyVertexMass(){}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::clear()
{
    helper::WriteOnlyAccessor < Data < MassVector > > vertexMass = d_vertexMass;
    helper::WriteOnlyAccessor < Data < MassVector > > edgeMass = d_edgeMass;
    vertexMass.clear();
    edgeMass.clear();
}


// -- Mass interface
template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addMDx(const core::MechanicalParams*, DataVecDeriv& vres, const DataVecDeriv& vdx, SReal factor)
{
    const MassVector &vertexMass= d_vertexMass.getValue();
    const MassVector &edgeMass= d_edgeMass.getValue();

    helper::WriteAccessor< DataVecDeriv > res = vres;
    helper::ReadAccessor< DataVecDeriv > dx = vdx;

    SReal massTotal = 0.0;

    //using a lumped matrix (default)-----
    if(isLumped())
    {
        for (size_t i=0; i<dx.size(); i++)
        {
            res[i] += dx[i] * vertexMass[i] * m_massLumpingCoeff * Real(factor);
            massTotal += vertexMass[i]*m_massLumpingCoeff * Real(factor);
        }

    }
    //using a sparse matrix---------------
    else
    {
        size_t nbEdges=m_topology->getNbEdges();
        size_t v0,v1;

        for (unsigned int i=0; i<dx.size(); i++)
        {
            res[i] += dx[i] * vertexMass[i] * Real(factor);
            massTotal += vertexMass[i] * Real(factor);
        }

        Real tempMass=0.0;

        for (unsigned int j=0; j<nbEdges; ++j)
        {
            tempMass = edgeMass[j] * Real(factor);

            v0=m_topology->getEdge(j)[0];
            v1=m_topology->getEdge(j)[1];

            res[v0] += dx[v1] * tempMass;
            res[v1] += dx[v0] * tempMass;

            massTotal += 2*edgeMass[j] * Real(factor);
        }
    }

    if(d_printMass.getValue() && (this->getContext()->getTime()==0.0))
    {
        msg_info() <<"Total Mass = "<<massTotal;
    }

    if(d_printMass.getValue())
    {
        std::map < std::string, sofa::type::vector<double> >& graph = *f_graph.beginEdit();
        sofa::type::vector<double>& graph_error = graph["Mass variations"];
        graph_error.push_back(massTotal+0.000001);

        f_graph.endEdit();
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f)
{
    SOFA_UNUSED(mparams);
    if( !isLumped() )
    {
        msg_error() << "the method 'accFromF' can't be used with MeshMatrixMass as this SPARSE mass matrix can't be inversed easily. "
                    << "Please proceed to mass lumping or use a DiagonalMass (both are equivalent).";
        return;
    }

    helper::WriteAccessor< DataVecDeriv > _a = a;
    const VecDeriv& _f = f.getValue();
    const MassVector &vertexMass= d_vertexMass.getValue();

    for (unsigned int i=0; i<vertexMass.size(); i++)
    {
        _a[i] = _f[i] / ( vertexMass[i] * m_massLumpingCoeff);
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addForce(const core::MechanicalParams*, DataVecDeriv& vf, const DataVecCoord& , const DataVecDeriv& )
{

    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if(this->m_separateGravity.getValue())
        return ;

    helper::WriteAccessor< DataVecDeriv > f = vf;

    const MassVector &vertexMass= d_vertexMass.getValue();

    // gravity
    type::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);

    // add weight and inertia force
    for (unsigned int i=0; i<f.size(); ++i)
        f[i] += theGravity * vertexMass[i] * m_massLumpingCoeff;
}


template <class DataTypes, class MassType>
SReal MeshMatrixMass<DataTypes, MassType>::getKineticEnergy( const core::MechanicalParams*, const DataVecDeriv& vv ) const
{
    const MassVector &vertexMass= d_vertexMass.getValue();
    const MassVector &edgeMass= d_edgeMass.getValue();

    helper::ReadAccessor< DataVecDeriv > v = vv;

    unsigned int nbEdges=m_topology->getNbEdges();
    unsigned int v0,v1;

    SReal e = 0;

    for (unsigned int i=0; i<v.size(); i++)
    {
        e += dot(v[i],v[i]) * vertexMass[i]; // v[i]*v[i]*masses[i] would be more efficient but less generic
    }

    for (unsigned int i = 0; i < nbEdges; ++i)
    {
        v0 = m_topology->getEdge(i)[0];
        v1 = m_topology->getEdge(i)[1];

        e += 2 * dot(v[v0], v[v1])*edgeMass[i];

    }

    return e/2;
}


template <class DataTypes, class MassType>
SReal MeshMatrixMass<DataTypes, MassType>::getPotentialEnergy( const core::MechanicalParams*, const DataVecCoord& vx) const
{
    const MassVector &vertexMass= d_vertexMass.getValue();

    helper::ReadAccessor< DataVecCoord > x = vx;

    SReal e = 0;
    // gravity
    type::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);

    for (unsigned int i=0; i<x.size(); i++)
        e -= dot(theGravity,x[i])*vertexMass[i] * m_massLumpingCoeff;

    return e;
}


// does nothing by default, need to be specialized in .cpp
template <class DataTypes, class MassType>
type::Vector6 MeshMatrixMass<DataTypes, MassType>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& /*vx*/, const DataVecDeriv& /*vv*/  ) const
{
    return type::Vector6();
}



template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if(this->mstate && mparams)
    {
        VecDeriv& v = *d_v.beginEdit();

        // gravity
        type::Vec3d g ( this->getContext()->getGravity() );
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2]);
        Deriv hg = theGravity * (typename DataTypes::Real(sofa::core::mechanicalparams::dt(mparams)));

        for (unsigned int i=0; i<v.size(); i++)
            v[i] += hg;
        d_v.endEdit();
    }

}




template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    const MassVector &vertexMass= d_vertexMass.getValue();
    const MassVector &edgeMass= d_edgeMass.getValue();

    size_t nbEdges=m_topology->getNbEdges();
    sofa::Index v0,v1;

    const int N = defaulttype::DataTypeInfo<Deriv>::size();
    AddMToMatrixFunctor<Deriv,MassType> calc;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = r.matrix;
    Real mFactor = Real(sofa::core::mechanicalparams::mFactorIncludingRayleighDamping(mparams, this->rayleighMass.getValue()));

    if((mat->colSize()) != (defaulttype::BaseMatrix::Index)(m_topology->getNbPoints()*N) || (mat->rowSize()) != (defaulttype::BaseMatrix::Index)(m_topology->getNbPoints()*N))
    {
        msg_error() <<"Wrong size of the input Matrix: need resize in addMToMatrix function.";
        mat->resize(m_topology->getNbPoints()*N,m_topology->getNbPoints()*N);
    }

    SReal massTotal=0.0;

    if(isLumped())
    {
        for (size_t i=0; i<vertexMass.size(); i++)
        {
            calc(r.matrix, vertexMass[i] * m_massLumpingCoeff, r.offset + N*i, mFactor);
            massTotal += vertexMass[i] * m_massLumpingCoeff;
        }

        if(d_printMass.getValue() && (this->getContext()->getTime()==0.0))
            msg_info() <<"Total Mass = "<<massTotal ;

        if(d_printMass.getValue())
        {
            std::map < std::string, sofa::type::vector<double> >& graph = *f_graph.beginEdit();
            sofa::type::vector<double>& graph_error = graph["Mass variations"];
            graph_error.push_back(massTotal);

            f_graph.endEdit();
        }
    }
    else
    {

        for (size_t i = 0; i < vertexMass.size(); i++)
        {
            calc(r.matrix, vertexMass[i], r.offset + N*i, mFactor);
            massTotal += vertexMass[i];
        }


        for (size_t j = 0; j < nbEdges; ++j)
        {
            v0 = m_topology->getEdge(j)[0];
            v1 = m_topology->getEdge(j)[1];

            calc(r.matrix, edgeMass[j], r.offset + N*v0, r.offset + N*v1, mFactor);
            calc(r.matrix, edgeMass[j], r.offset + N*v1, r.offset + N*v0, mFactor);

            massTotal += 2 * edgeMass[j];
        }

        if(d_printMass.getValue() && (this->getContext()->getTime()==0.0))
            msg_info() <<"Total Mass  = "<<massTotal ;

        if(d_printMass.getValue())
        {
            std::map < std::string, sofa::type::vector<double> >& graph = *f_graph.beginEdit();
            sofa::type::vector<double>& graph_error = graph["Mass variations"];
            graph_error.push_back(massTotal+0.000001);

            f_graph.endEdit();
        }

    }


}


template <class DataTypes, class MassType>
SReal MeshMatrixMass<DataTypes, MassType>::getElementMass(Index index) const
{
    const MassVector &vertexMass= d_vertexMass.getValue();
    SReal mass = vertexMass[index] * m_massLumpingCoeff;

    return mass;
}


//TODO: special case for Rigid Mass
template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::getElementMass(Index index, defaulttype::BaseMatrix *m) const
{
    static const defaulttype::BaseMatrix::Index dimension = defaulttype::BaseMatrix::Index(defaulttype::DataTypeInfo<Deriv>::size());
    if (m->rowSize() != dimension || m->colSize() != dimension) m->resize(dimension,dimension);

    m->clear();
    AddMToMatrixFunctor<Deriv,MassType>()(m, d_vertexMass.getValue()[index] * m_massLumpingCoeff, 0, 1);
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::handleEvent(sofa::core::objectmodel::Event *event)
{
    SOFA_UNUSED(event);
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    const MassVector &vertexMass= d_vertexMass.getValue();

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    Coord gravityCenter;
    Real totalMass=0.0;

    std::vector<  type::Vector3 > points;
    for (unsigned int i=0; i<x.size(); i++)
    {
        type::Vector3 p;
        p = DataTypes::getCPos(x[i]);

        points.push_back(p);
        gravityCenter += x[i]*vertexMass[i]*m_massLumpingCoeff;
        totalMass += vertexMass[i]*m_massLumpingCoeff;
    }

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->disableLighting();
    sofa::type::RGBAColor color(1.0,1.0,1.0,1.0);

    vparams->drawTool()->drawPoints(points, 2, color);

    std::vector<sofa::type::Vector3> vertices;

    if(d_showCenterOfGravity.getValue())
    {
        color = sofa::type::RGBAColor(1.0,1.0,0,1.0);
        gravityCenter /= totalMass;
        for(unsigned int i=0 ; i<Coord::spatial_dimensions ; i++)
        {
            Coord v, diff;
            v[i] = d_showAxisSize.getValue();
            diff = gravityCenter-v;
            vertices.push_back(sofa::type::Vector3(diff));
            diff = gravityCenter+v;
            vertices.push_back(sofa::type::Vector3(diff));
        }
    }
    vparams->drawTool()->drawLines(vertices,5,color);
    vparams->drawTool()->restoreLastState();
}


} // namespace sofa::component::mass
