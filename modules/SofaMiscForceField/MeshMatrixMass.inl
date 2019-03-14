/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MASS_MESHMATRIXMASS_INL
#define SOFA_COMPONENT_MASS_MESHMATRIXMASS_INL

#include <SofaMiscForceField/MeshMatrixMass.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaBaseMechanics/AddMToMatrixFunctor.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>
#include <SofaBaseTopology/CommonAlgorithms.h>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.h>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.h>
#include <sofa/simulation/AnimateEndEvent.h>


namespace sofa
{

namespace component
{

namespace mass
{

template <class DataTypes, class MassType>
MeshMatrixMass<DataTypes, MassType>::MeshMatrixMass()
    :
      d_vertexMass( initData(&d_vertexMass, "vertexMass", "Specify a vector giving the mass of each vertex. \n"
                                                           "If unspecified or wrongly set, another mass information is used.") )
    , d_massDensity( initData(&d_massDensity, "massDensity", "Specify real and strictly positive value(s) for the mass density. \n"
                                                                "If unspecified or wrongly set, the totalMass information is used.") )
    , d_totalMass( initData(&d_totalMass, (Real) 1.0, "totalMass","Specify the total mass resulting from all particles. \n"
                                                           "If unspecified or wrongly set, the default value is used: totalMass = 1.0") )
    , d_vertexMassInfo( initData(&d_vertexMassInfo, "vertexMassInfo", "internal values of the particles masses on vertices, supporting topological changes") )
    , d_edgeMassInfo( initData(&d_edgeMassInfo, "edgeMassInfo", "internal values of the particles masses on edges, supporting topological changes") )
    , d_edgeMass( initData(&d_edgeMass, "edgeMass", "values of the particles masses on edges") )
    , d_computeMassOnRest(initData(&d_computeMassOnRest, false, "computeMassOnRest", "If true, the mass of every element is computed based on the rest position rather than the position"))
    , d_showCenterOfGravity( initData(&d_showCenterOfGravity, false, "showGravityCenter", "display the center of gravity of the system" ) )
    , d_showAxisSize( initData(&d_showAxisSize, (Real)1.0, "showAxisSizeFactor", "factor length of the axis displayed (only used for rigids)" ) )
    , d_lumping( initData(&d_lumping, false, "lumping","boolean if you need to use a lumped mass matrix") )
    , d_printMass( initData(&d_printMass, false, "printMass","boolean if you want to check the mass conservation") )
    , f_graph( initData(&f_graph,"graph","Graph of the controlled potential") )
    , m_topologyType(TOPOLOGY_UNKNOWN)
    , m_vertexMassHandler(NULL)
    , m_edgeMassHandler(NULL)
{
    f_graph.setWidget("graph");

    /// Internal data, not supposed to be accessed by the user
    d_vertexMassInfo.setDisplayed(false);
    d_edgeMassInfo.setDisplayed(false);
}

template <class DataTypes, class MassType>
MeshMatrixMass<DataTypes, MassType>::~MeshMatrixMass()
{
    if (m_vertexMassHandler) delete m_vertexMassHandler;
    if (m_edgeMassHandler) delete m_edgeMassHandler;
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyCreateFunction(unsigned int, MassType & VertexMass,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    VertexMass = 0;
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyCreateFunction(unsigned int, MassType & EdgeMass,
        const core::topology::BaseMeshTopology::Edge&,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    EdgeMass = 0;
}


// -------------------------------------------------------
// ------- Triangle Creation/Destruction functions -------
// -------------------------------------------------------
//{

/// Creation fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyTriangleCreation(const sofa::helper::vector< unsigned int >& triangleAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Triangle >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->d_vertexMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<triangleAdded.size(); ++i)
        {
            // Get the triangle to be added
            const core::topology::BaseMeshTopology::Triangle &t = MMM->_topology->getTriangle(triangleAdded[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(MMM->triangleGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[triangleAdded[i]] * MMM->triangleGeo->computeRestTriangleArea(triangleAdded[i]))/(typename DataTypes::Real)6.0;
                }
                else
                {
                    mass=(densityM[triangleAdded[i]] * MMM->triangleGeo->computeTriangleArea(triangleAdded[i]))/(typename DataTypes::Real)6.0;
                }
            }

            // Adding mass
            for (unsigned int j=0; j<3; ++j)
                VertexMasses[ t[j] ] += mass;
        }
    }
}

/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyTriangleCreation(const sofa::helper::vector< unsigned int >& triangleAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Triangle >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->d_edgeMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<triangleAdded.size(); ++i)
        {
            // Get the edgesInTriangle to be added
            const core::topology::BaseMeshTopology::EdgesInTriangle &te = MMM->_topology->getEdgesInTriangle(triangleAdded[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(MMM->triangleGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[triangleAdded[i]] * MMM->triangleGeo->computeRestTriangleArea(triangleAdded[i]))/(typename DataTypes::Real)12.0;
                }
                else
                {
                    mass=(densityM[triangleAdded[i]] * MMM->triangleGeo->computeTriangleArea(triangleAdded[i]))/(typename DataTypes::Real)12.0;
                }
            }

            // Adding mass edges of concerne triangle
            for (unsigned int j=0; j<3; ++j)
                EdgeMasses[ te[j] ] += mass;
        }
    }
}

/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyTriangleDestruction(const sofa::helper::vector< unsigned int >& triangleRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->d_vertexMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<triangleRemoved.size(); ++i)
        {
            // Get the triangle to be removed
            const core::topology::BaseMeshTopology::Triangle &t = MMM->_topology->getTriangle(triangleRemoved[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(MMM->triangleGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[triangleRemoved[i]] * MMM->triangleGeo->computeRestTriangleArea(triangleRemoved[i]))/(typename DataTypes::Real)6.0;
                }
                else
                {
                    mass=(densityM[triangleRemoved[i]] * MMM->triangleGeo->computeTriangleArea(triangleRemoved[i]))/(typename DataTypes::Real)6.0;
                }
            }

            // Removing mass
            for (unsigned int j=0; j<3; ++j)
                VertexMasses[ t[j] ] -= mass;
        }
    }
}

/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyTriangleDestruction(const sofa::helper::vector< unsigned int >& triangleRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->d_edgeMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<triangleRemoved.size(); ++i)
        {
            // Get the triangle to be removed
            const core::topology::BaseMeshTopology::EdgesInTriangle &te = MMM->_topology->getEdgesInTriangle(triangleRemoved[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(MMM->triangleGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[triangleRemoved[i]] * MMM->triangleGeo->computeRestTriangleArea(triangleRemoved[i]))/(typename DataTypes::Real)12.0;
                }
                else
                {
                    mass=(densityM[triangleRemoved[i]] * MMM->triangleGeo->computeTriangleArea(triangleRemoved[i]))/(typename DataTypes::Real)12.0;
                }
            }

            // Removing mass edges of concerne triangle
            for (unsigned int j=0; j<3; ++j)
                EdgeMasses[ te[j] ] -= mass;
        }
    }
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::TrianglesAdded* e)
{
    const sofa::helper::vector<unsigned int> &triangleAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyTriangleCreation(triangleAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::TrianglesRemoved* e)
{
    const sofa::helper::vector<unsigned int> &triangleRemoved = e->getArray();

    applyTriangleDestruction(triangleRemoved);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::TrianglesAdded* e)
{
    const sofa::helper::vector<unsigned int> &triangleAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyTriangleCreation(triangleAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::TrianglesRemoved* e)
{
    const sofa::helper::vector<unsigned int> &triangleRemoved = e->getArray();

    applyTriangleDestruction(triangleRemoved);
}

// }

// ---------------------------------------------------
// ------- Quad Creation/Destruction functions -------
// ---------------------------------------------------
//{

/// Creation fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyQuadCreation(const sofa::helper::vector< unsigned int >& quadAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Quad >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_QUADSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->d_vertexMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<quadAdded.size(); ++i)
        {
            // Get the quad to be added
            const core::topology::BaseMeshTopology::Quad &q = MMM->_topology->getQuad(quadAdded[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(MMM->quadGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[quadAdded[i]] * MMM->quadGeo->computeRestQuadArea(quadAdded[i]))/(typename DataTypes::Real)8.0;
                }
                else
                {
                    mass=(densityM[quadAdded[i]] * MMM->quadGeo->computeQuadArea(quadAdded[i]))/(typename DataTypes::Real)8.0;
                }
            }

            // Adding mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ q[j] ] += mass;
        }
    }
}

/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyQuadCreation(const sofa::helper::vector< unsigned int >& quadAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Quad >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_QUADSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->d_edgeMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<quadAdded.size(); ++i)
        {
            // Get the EdgesInQuad to be added
            const core::topology::BaseMeshTopology::EdgesInQuad &qe = MMM->_topology->getEdgesInQuad(quadAdded[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(MMM->quadGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[quadAdded[i]] * MMM->quadGeo->computeRestQuadArea(quadAdded[i]))/(typename DataTypes::Real)16.0;
                }
                else
                {
                    mass=(densityM[quadAdded[i]] * MMM->quadGeo->computeQuadArea(quadAdded[i]))/(typename DataTypes::Real)16.0;
                }
            }

            // Adding mass edges of concerne quad
            for (unsigned int j=0; j<4; ++j)
                EdgeMasses[ qe[j] ] += mass;
        }
    }
}

/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyQuadDestruction(const sofa::helper::vector< unsigned int >& quadRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_QUADSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->d_vertexMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<quadRemoved.size(); ++i)
        {
            // Get the quad to be removed
            const core::topology::BaseMeshTopology::Quad &q = MMM->_topology->getQuad(quadRemoved[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(MMM->quadGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[quadRemoved[i]] * MMM->quadGeo->computeRestQuadArea(quadRemoved[i]))/(typename DataTypes::Real)8.0;
                }
                else
                {
                    mass=(densityM[quadRemoved[i]] * MMM->quadGeo->computeQuadArea(quadRemoved[i]))/(typename DataTypes::Real)8.0;
                }
            }

            // Removing mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ q[j] ] -= mass;
        }
    }
}

/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyQuadDestruction(const sofa::helper::vector< unsigned int >& quadRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_QUADSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->d_edgeMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<quadRemoved.size(); ++i)
        {
            // Get the EdgesInQuad to be removed
            const core::topology::BaseMeshTopology::EdgesInQuad &qe = MMM->_topology->getEdgesInQuad(quadRemoved[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(MMM->quadGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[quadRemoved[i]] * MMM->quadGeo->computeRestQuadArea(quadRemoved[i]))/(typename DataTypes::Real)16.0;
                }
                else
                {
                    mass=(densityM[quadRemoved[i]] * MMM->quadGeo->computeQuadArea(quadRemoved[i]))/(typename DataTypes::Real)16.0;
                }
            }

            // Removing mass edges of concerne quad
            for (unsigned int j=0; j<4; ++j)
                EdgeMasses[ qe[j] ] -= mass/2;
        }
    }
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::QuadsAdded* e)
{
    const sofa::helper::vector<unsigned int> &quadAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Quad> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyQuadCreation(quadAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::QuadsRemoved* e)
{
    const sofa::helper::vector<unsigned int> &quadRemoved = e->getArray();

    applyQuadDestruction(quadRemoved);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::QuadsAdded* e)
{
    const sofa::helper::vector<unsigned int> &quadAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Quad> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyQuadCreation(quadAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::QuadsRemoved* e)
{
    const sofa::helper::vector<unsigned int> &quadRemoved = e->getArray();

    applyQuadDestruction(quadRemoved);
}

// }



// ----------------------------------------------------------
// ------- Tetrahedron Creation/Destruction functions -------
// ----------------------------------------------------------
//{

/// Creation fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyTetrahedronCreation(const sofa::helper::vector< unsigned int >& tetrahedronAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Tetrahedron >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->d_vertexMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<tetrahedronAdded.size(); ++i)
        {
            // Get the tetrahedron to be added
            const core::topology::BaseMeshTopology::Tetrahedron &t = MMM->_topology->getTetrahedron(tetrahedronAdded[i]);

            // Compute rest mass of conserne tetrahedron = density * tetrahedron volume.
            if(MMM->tetraGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[tetrahedronAdded[i]] * MMM->tetraGeo->computeRestTetrahedronVolume(tetrahedronAdded[i]))/(typename DataTypes::Real)10.0;
                }
                else
                {
                    mass=(densityM[tetrahedronAdded[i]] * MMM->tetraGeo->computeTetrahedronVolume(tetrahedronAdded[i]))/(typename DataTypes::Real)10.0;
                }
            }

            // Adding mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ t[j] ] += mass;
        }
    }
}

/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyTetrahedronCreation(const sofa::helper::vector< unsigned int >& tetrahedronAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Tetrahedron >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->d_edgeMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<tetrahedronAdded.size(); ++i)
        {
            // Get the edgesInTetrahedron to be added
            const core::topology::BaseMeshTopology::EdgesInTetrahedron &te = MMM->_topology->getEdgesInTetrahedron(tetrahedronAdded[i]);

            // Compute rest mass of conserne triangle = density * tetrahedron volume.
            if(MMM->tetraGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[tetrahedronAdded[i]] * MMM->tetraGeo->computeRestTetrahedronVolume(tetrahedronAdded[i]))/(typename DataTypes::Real)20.0;
                }
                else
                {
                    mass=(densityM[tetrahedronAdded[i]] * MMM->tetraGeo->computeTetrahedronVolume(tetrahedronAdded[i]))/(typename DataTypes::Real)20.0;
                }
            }

            // Adding mass edges of concerne triangle
            for (unsigned int j=0; j<6; ++j)
                EdgeMasses[ te[j] ] += mass;
        }
    }
}

/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyTetrahedronDestruction(const sofa::helper::vector< unsigned int >& tetrahedronRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->d_vertexMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<tetrahedronRemoved.size(); ++i)
        {
            // Get the tetrahedron to be removed
            const core::topology::BaseMeshTopology::Tetrahedron &t = MMM->_topology->getTetrahedron(tetrahedronRemoved[i]);

            // Compute rest mass of conserne tetrahedron = density * tetrahedron volume.
            if(MMM->tetraGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[tetrahedronRemoved[i]] * MMM->tetraGeo->computeRestTetrahedronVolume(tetrahedronRemoved[i]))/(typename DataTypes::Real)10.0;
                }
                else
                {
                    mass=(densityM[tetrahedronRemoved[i]] * MMM->tetraGeo->computeTetrahedronVolume(tetrahedronRemoved[i]))/(typename DataTypes::Real)10.0;
                }
            }

            // Removing mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ t[j] ] -= mass;
        }
    }
}

/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyTetrahedronDestruction(const sofa::helper::vector< unsigned int >& tetrahedronRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->d_edgeMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<tetrahedronRemoved.size(); ++i)
        {
            // Get the edgesInTetrahedron to be removed
            const core::topology::BaseMeshTopology::EdgesInTetrahedron &te = MMM->_topology->getEdgesInTetrahedron(tetrahedronRemoved[i]);

            // Compute rest mass of conserne triangle = density * tetrahedron volume.
            if(MMM->tetraGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[tetrahedronRemoved[i]] * MMM->tetraGeo->computeRestTetrahedronVolume(tetrahedronRemoved[i]))/(typename DataTypes::Real)20.0;
                }
                else
                {
                    mass=(densityM[tetrahedronRemoved[i]] * MMM->tetraGeo->computeTetrahedronVolume(tetrahedronRemoved[i]))/(typename DataTypes::Real)20.0;
                }
            }

            // Removing mass edges of concerne triangle
            for (unsigned int j=0; j<6; ++j)
                EdgeMasses[ te[j] ] -= mass;
        }
    }
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::TetrahedraAdded* e)
{
    const sofa::helper::vector<unsigned int> &tetraAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyTetrahedronCreation(tetraAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::TetrahedraRemoved* e)
{
    const sofa::helper::vector<unsigned int> &tetraRemoved = e->getArray();

    applyTetrahedronDestruction(tetraRemoved);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::TetrahedraAdded* e)
{
    const sofa::helper::vector<unsigned int> &tetraAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyTetrahedronCreation(tetraAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::TetrahedraRemoved* e)
{
    const sofa::helper::vector<unsigned int> &tetraRemoved = e->getArray();

    applyTetrahedronDestruction(tetraRemoved);
}

// }


// ---------------------------------------------------------
// ------- Hexahedron Creation/Destruction functions -------
// ---------------------------------------------------------
//{

/// Creation fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyHexahedronCreation(const sofa::helper::vector< unsigned int >& hexahedronAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Hexahedron >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_HEXAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->d_vertexMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<hexahedronAdded.size(); ++i)
        {
            // Get the hexahedron to be added
            const core::topology::BaseMeshTopology::Hexahedron &h = MMM->_topology->getHexahedron(hexahedronAdded[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(MMM->hexaGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[hexahedronAdded[i]] * MMM->hexaGeo->computeRestHexahedronVolume(hexahedronAdded[i]))/(typename DataTypes::Real)20.0;
                }
                else
                {
                    mass=(densityM[hexahedronAdded[i]] * MMM->hexaGeo->computeHexahedronVolume(hexahedronAdded[i]))/(typename DataTypes::Real)20.0;
                }
            }

            // Adding mass
            for (unsigned int j=0; j<8; ++j)
                VertexMasses[ h[j] ] += mass;
        }
    }
}

/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyHexahedronCreation(const sofa::helper::vector< unsigned int >& hexahedronAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Hexahedron >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_HEXAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->d_edgeMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<hexahedronAdded.size(); ++i)
        {
            // Get the EdgesInHexahedron to be added
            const core::topology::BaseMeshTopology::EdgesInHexahedron &he = MMM->_topology->getEdgesInHexahedron(hexahedronAdded[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(MMM->hexaGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[hexahedronAdded[i]] * MMM->hexaGeo->computeRestHexahedronVolume(hexahedronAdded[i]))/(typename DataTypes::Real)40.0;
                }
                else
                {
                    mass=(densityM[hexahedronAdded[i]] * MMM->hexaGeo->computeHexahedronVolume(hexahedronAdded[i]))/(typename DataTypes::Real)40.0;
                }
            }

            // Adding mass edges of concerne triangle
            for (unsigned int j=0; j<12; ++j)
                EdgeMasses[ he[j] ] += mass;
        }
    }
}

/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyHexahedronDestruction(const sofa::helper::vector< unsigned int >& hexahedronRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_HEXAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->d_vertexMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<hexahedronRemoved.size(); ++i)
        {
            // Get the hexahedron to be removed
            const core::topology::BaseMeshTopology::Hexahedron &h = MMM->_topology->getHexahedron(hexahedronRemoved[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(MMM->hexaGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[hexahedronRemoved[i]] * MMM->hexaGeo->computeRestHexahedronVolume(hexahedronRemoved[i]))/(typename DataTypes::Real)20.0;
                }
                else
                {
                    mass=(densityM[hexahedronRemoved[i]] * MMM->hexaGeo->computeHexahedronVolume(hexahedronRemoved[i]))/(typename DataTypes::Real)20.0;
                }
            }

            // Removing mass
            for (unsigned int j=0; j<8; ++j)
                VertexMasses[ h[j] ] -= mass;
        }
    }
}

/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyHexahedronDestruction(const sofa::helper::vector< unsigned int >& hexahedronRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_HEXAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->d_edgeMassInfo );
        // Initialisation
        const helper::vector<Real> densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<hexahedronRemoved.size(); ++i)
        {
            // Get the EdgesInHexahedron to be removed
            const core::topology::BaseMeshTopology::EdgesInHexahedron &he = MMM->_topology->getEdgesInHexahedron(hexahedronRemoved[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(MMM->hexaGeo)
            {
                if(MMM->d_computeMassOnRest.getValue())
                {
                    mass=(densityM[hexahedronRemoved[i]] * MMM->hexaGeo->computeRestHexahedronVolume(hexahedronRemoved[i]))/(typename DataTypes::Real)40.0;
                }
                else
                {
                    mass=(densityM[hexahedronRemoved[i]] * MMM->hexaGeo->computeHexahedronVolume(hexahedronRemoved[i]))/(typename DataTypes::Real)40.0;
                }
            }

            // Removing mass edges of concerne triangle
            for (unsigned int j=0; j<12; ++j)
                EdgeMasses[ he[j] ] -= mass;
        }
    }
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::HexahedraAdded* e)
{
    const sofa::helper::vector<unsigned int> &hexaAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Hexahedron> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyHexahedronCreation(hexaAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::HexahedraRemoved* e)
{
    const sofa::helper::vector<unsigned int> &hexaRemoved = e->getArray();

    applyHexahedronDestruction(hexaRemoved);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::HexahedraAdded* e)
{
    const sofa::helper::vector<unsigned int> &hexaAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Hexahedron> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyHexahedronCreation(hexaAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::HexahedraRemoved* e)
{
    const sofa::helper::vector<unsigned int> &hexaRemoved = e->getArray();

    applyHexahedronDestruction(hexaRemoved);
}

// }


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::init()
{
    m_dataTrackerVertex.trackData(d_vertexMass);
    m_dataTrackerEdge.trackData(d_edgeMass);
    m_dataTrackerDensity.trackData(d_massDensity);
    m_dataTrackerTotal.trackData(d_totalMass);

    m_massLumpingCoeff = 0.0;

    if(!checkTopology())
    {
        return;
    }
    Inherited::init();
    initTopologyHandlers();

    massInitialization();

    //Reset the graph
    f_graph.beginEdit()->clear();
    f_graph.endEdit();

    //Function for GPU-CUDA version only
    this->copyVertexMass();
}


template <class DataTypes, class MassType>
bool MeshMatrixMass<DataTypes, MassType>::checkTopology()
{
    _topology = this->getContext()->getMeshTopology();

    this->getContext()->get(edgeGeo);
    this->getContext()->get(triangleGeo);
    this->getContext()->get(quadGeo);
    this->getContext()->get(tetraGeo);
    this->getContext()->get(hexaGeo);

    if (_topology)
    {
        if (_topology->getNbHexahedra() > 0)
        {
            if(!hexaGeo)
            {
                msg_error() << "Hexahedron topology but no geometry algorithms found. Add the component HexahedronSetGeometryAlgorithms.";
                return false;
            }
            else
            {
                msg_info() << "Hexahedral topology found.";
                return true;
            }
        }
        else if (_topology->getNbTetrahedra() > 0)
        {
            if(!tetraGeo)
            {
                msg_error() << "Tetrahedron topology but no geometry algorithms found. Add the component TetrahedronSetGeometryAlgorithms.";
                return false;
            }
            else
            {
                msg_info() << "Tetrahedral topology found.";
                return true;
            }
        }
        else if (_topology->getNbQuads() > 0)
        {
            if(!quadGeo)
            {
                msg_error() << "Quad topology but no geometry algorithms found. Add the component QuadSetGeometryAlgorithms.";
                return false;
            }
            else
            {
                msg_info() << "Quad topology found.";
                return true;
            }
        }
        else if (_topology->getNbTriangles() > 0)
        {
            if(!triangleGeo)
            {
                msg_error() << "Triangle topology but no geometry algorithms found. Add the component TriangleSetGeometryAlgorithms.";
                return false;
            }
            else
            {
                msg_info() << "Triangular topology found.";
                return true;
            }
        }
        else if (_topology->getNbEdges() > 0)
        {
            if(!edgeGeo)
            {
                msg_error() << "Edge topology but no geometry algorithms found. Add the component EdgeSetGeometryAlgorithms.";
                return false;
            }
            else
            {
                msg_info() << "Edge topology found.";
                return true;
            }
        }
        else
        {
            msg_error() << "Topology empty.";
            return false;
        }
    }
    else
    {
        msg_error() << "Topology not found.";
        return false;
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::initTopologyHandlers()
{
    // add the functions to handle topology changes for Vertex informations
    m_vertexMassHandler = new VertexMassHandler(this, &d_vertexMassInfo);
    d_vertexMassInfo.createTopologicalEngine(_topology, m_vertexMassHandler);
    d_vertexMassInfo.linkToEdgeDataArray();
    d_vertexMassInfo.linkToTriangleDataArray();
    d_vertexMassInfo.linkToQuadDataArray();
    d_vertexMassInfo.linkToTetrahedronDataArray();
    d_vertexMassInfo.linkToHexahedronDataArray();
    d_vertexMassInfo.registerTopologicalData();

    // add the functions to handle topology changes for Edge informations
    m_edgeMassHandler = new EdgeMassHandler(this, &d_edgeMassInfo);
    d_edgeMassInfo.createTopologicalEngine(_topology, m_edgeMassHandler);
    d_edgeMassInfo.linkToTriangleDataArray();
    d_edgeMassInfo.linkToQuadDataArray();
    d_edgeMassInfo.linkToTetrahedronDataArray();
    d_edgeMassInfo.linkToHexahedronDataArray();
    d_edgeMassInfo.registerTopologicalData();
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
            else if(d_lumping.getValue() && !d_edgeMass.isSet())
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

    d_vertexMass.setValue(d_vertexMassInfo.getValue());
    d_edgeMass.setValue(d_edgeMassInfo.getValue());

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
        average_vertex /= (Real)(vertexM.size());
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
        average_density /= (Real)(mDensity.size());
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
    helper::vector<MassType>& my_vertexMassInfo = *d_vertexMassInfo.beginEdit();
    helper::vector<MassType>& my_edgeMassInfo = *d_edgeMassInfo.beginEdit();

    unsigned int ndof = this->mstate->getSize();
    unsigned int nbEdges=_topology->getNbEdges();
    const helper::vector<core::topology::BaseMeshTopology::Edge>& edges = _topology->getEdges();

    my_vertexMassInfo.resize(ndof);
    my_edgeMassInfo.resize(nbEdges);

    const helper::vector< unsigned int > emptyAncestor;
    const helper::vector< double > emptyCoefficient;
    const helper::vector< helper::vector< unsigned int > > emptyAncestors;
    const helper::vector< helper::vector< double > > emptyCoefficients;

    // set vertex tensor to 0
    for (unsigned int i = 0; i<ndof; ++i)
        m_vertexMassHandler->applyCreateFunction(i, my_vertexMassInfo[i], emptyAncestor, emptyCoefficient);

    // set edge tensor to 0
    for (unsigned int i = 0; i<nbEdges; ++i)
        m_edgeMassHandler->applyCreateFunction(i, my_edgeMassInfo[i], edges[i], emptyAncestor, emptyCoefficient);

    // Create mass matrix depending on current Topology:
    if (_topology->getNbHexahedra()>0 && hexaGeo)  // Hexahedron topology
    {
        // create vector tensor by calling the hexahedron creation function on the entire mesh
        sofa::helper::vector<unsigned int> hexahedraAdded;
        setMassTopologyType(TOPOLOGY_HEXAHEDRONSET);
        int n = _topology->getNbHexahedra();
        for (int i = 0; i<n; ++i)
            hexahedraAdded.push_back(i);

        m_vertexMassHandler->applyHexahedronCreation(hexahedraAdded, _topology->getHexahedra(), emptyAncestors, emptyCoefficients);
        m_edgeMassHandler->applyHexahedronCreation(hexahedraAdded, _topology->getHexahedra(), emptyAncestors, emptyCoefficients);
        m_massLumpingCoeff = 2.5;
    }
    else if (_topology->getNbTetrahedra()>0 && tetraGeo)  // Tetrahedron topology
    {
        // create vector tensor by calling the tetrahedron creation function on the entire mesh
        sofa::helper::vector<unsigned int> tetrahedraAdded;
        setMassTopologyType(TOPOLOGY_TETRAHEDRONSET);

        int n = _topology->getNbTetrahedra();
        for (int i = 0; i<n; ++i)
            tetrahedraAdded.push_back(i);

        m_vertexMassHandler->applyTetrahedronCreation(tetrahedraAdded, _topology->getTetrahedra(), emptyAncestors, emptyCoefficients);
        m_edgeMassHandler->applyTetrahedronCreation(tetrahedraAdded, _topology->getTetrahedra(), emptyAncestors, emptyCoefficients);
        m_massLumpingCoeff = 2.5;
    }
    else if (_topology->getNbQuads()>0 && quadGeo)  // Quad topology
    {
        // create vector tensor by calling the quad creation function on the entire mesh
        sofa::helper::vector<unsigned int> quadsAdded;
        setMassTopologyType(TOPOLOGY_QUADSET);

        int n = _topology->getNbQuads();
        for (int i = 0; i<n; ++i)
            quadsAdded.push_back(i);

        m_vertexMassHandler->applyQuadCreation(quadsAdded, _topology->getQuads(), emptyAncestors, emptyCoefficients);
        m_edgeMassHandler->applyQuadCreation(quadsAdded, _topology->getQuads(), emptyAncestors, emptyCoefficients);
        m_massLumpingCoeff = 2.0;
    }
    else if (_topology->getNbTriangles()>0 && triangleGeo) // Triangle topology
    {
        // create vector tensor by calling the triangle creation function on the entire mesh
        sofa::helper::vector<unsigned int> trianglesAdded;
        setMassTopologyType(TOPOLOGY_TRIANGLESET);

        int n = _topology->getNbTriangles();
        for (int i = 0; i<n; ++i)
            trianglesAdded.push_back(i);

        m_vertexMassHandler->applyTriangleCreation(trianglesAdded, _topology->getTriangles(), emptyAncestors, emptyCoefficients);
        m_edgeMassHandler->applyTriangleCreation(trianglesAdded, _topology->getTriangles(), emptyAncestors, emptyCoefficients);
        m_massLumpingCoeff = 2.0;
    }

    d_vertexMassInfo.registerTopologicalData();
    d_edgeMassInfo.registerTopologicalData();

    d_vertexMassInfo.endEdit();
    d_edgeMassInfo.endEdit();
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::reinit()
{
    if (m_dataTrackerTotal.hasChanged() || m_dataTrackerDensity.hasChanged() || m_dataTrackerVertex.hasChanged())
    {
        update();
    }
}


template <class DataTypes, class MassType>
bool MeshMatrixMass<DataTypes, MassType>::update()
{
    bool update = false;

    if (m_dataTrackerTotal.hasChanged())
    {
        if(checkTotalMass())
        {
            initFromTotalMass();
            update = true;
        }
        m_dataTrackerTotal.clean();
    }
    else if(m_dataTrackerDensity.hasChanged())
    {
        if(checkMassDensity())
        {
            initFromMassDensity();
            update = true;
        }
        m_dataTrackerDensity.clean();
    }
    else if(m_dataTrackerVertex.hasChanged())
    {
        if(m_dataTrackerEdge.hasChanged())
        {
            if(checkVertexMass() && checkEdgeMass() )
            {
                initFromVertexAndEdgeMass();
                update = true;
            }
            m_dataTrackerVertex.clean();
            m_dataTrackerEdge.clean();
        }
        else if(d_lumping.getValue() && (!m_dataTrackerEdge.hasChanged()))
        {
            if(checkVertexMass())
            {
                initFromVertexMass();
                update = true;
            }
            m_dataTrackerVertex.clean();
        }
        else
        {
            msg_error() << "Initialization using vertexMass requires the lumping option or the edgeMass information";
        }
    }

    if(update)
    {
        d_vertexMass.setValue(d_vertexMassInfo.getValue());
        d_edgeMass.setValue(d_edgeMassInfo.getValue());

        //Info post-init
        msg_info() << "mass information updated";
        printMass();
    }

    return update;
}


template <class DataTypes, class MassType>
bool MeshMatrixMass<DataTypes, MassType>::checkTotalMass()
{
    //Check for negative or null value, if wrongly set use the default value totalMass = 1.0
    if(d_totalMass.getValue() <= 0.0)
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
    const sofa::helper::vector<Real> &vertexMass = d_vertexMass.getValue();
    //Check size of the vector
    if (vertexMass.size() != (size_t)_topology->getNbPoints())
    {
        msg_warning() << "Inconsistent size of vertexMass vector ("<< vertexMass.size() <<") compared to the DOFs size ("<< _topology->getNbPoints() <<").";
        return false;
    }
    else
    {
        //Check that the vertexMass vector has only strictly positive values
        for(size_t i=0; i<vertexMass.size(); i++)
        {
            if(vertexMass[i]<=0)
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

    const sofa::helper::vector<MassType> vertexMass = d_vertexMass.getValue();
    Real totalMassSave = 0.0;
    for(size_t i=0; i<vertexMass.size(); i++)
    {
        totalMassSave += vertexMass[i];
    }
    //Compute the volume
    setMassDensity(1.0);

    computeMass();

    helper::WriteAccessor<Data<MassVector> > vertexMassInfo = d_vertexMassInfo;
    //Compute volume = mass since massDensity = 1.0
    Real volume = 0.0;
    for(size_t i=0; i<vertexMassInfo.size(); i++)
    {
        volume += vertexMassInfo[i]*m_massLumpingCoeff;
        vertexMassInfo[i] = vertexMass[i];
    }
    m_massLumpingCoeff = 1.0;
    //Update all computed values
    setMassDensity((Real)totalMassSave/volume);
    d_totalMass.setValue(totalMassSave);
}


template <class DataTypes, class MassType>
bool MeshMatrixMass<DataTypes, MassType>::checkEdgeMass()
{
    const sofa::helper::vector<Real> edgeMass = d_edgeMass.getValue();
    //Check size of the vector
    if (edgeMass.size() != (size_t)_topology->getNbEdges())
    {
        msg_warning() << "Inconsistent size of vertexMass vector compared to the DOFs size.";
        return false;
    }
    else
    {
        //Check that the vertexMass vector has only strictly positive values
        for(size_t i=0; i<edgeMass.size(); i++)
        {
            if(edgeMass[i]<=0)
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

    const sofa::helper::vector<MassType> vertexMass = d_vertexMass.getValue();
    const sofa::helper::vector<MassType> edgeMass = d_edgeMass.getValue();
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

    helper::WriteAccessor<Data<MassVector> > vertexMassInfo = d_vertexMassInfo;
    helper::WriteAccessor<Data<MassVector> > edgeMassInfo = d_edgeMassInfo;
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
    setMassDensity((Real)totalMassSave/volume);
    d_totalMass.setValue(totalMassSave);
}


template <class DataTypes, class MassType>
bool MeshMatrixMass<DataTypes, MassType>::checkMassDensity()
{
    const sofa::helper::vector<Real> &massDensity = d_massDensity.getValue();
    Real density = massDensity[0];
    size_t sizeElements = 0;

    //Check size of the vector
    //Size = 1, homogeneous density
    //Otherwise, heterogeneous density
    if (_topology->getNbHexahedra()>0 && hexaGeo)
    {
        sizeElements = (size_t)_topology->getNbHexahedra();

        if ( massDensity.size() != (size_t)_topology->getNbHexahedra() && massDensity.size() != 1)
        {
            msg_warning() << "Inconsistent size of massDensity = " << massDensity.size() << ", should be either 1 or " << _topology->getNbHexahedra();
            return false;
        }
    }
    else if (_topology->getNbTetrahedra()>0 && tetraGeo)
    {
        sizeElements = (size_t)_topology->getNbTetrahedra();

        if ( massDensity.size() != (size_t)_topology->getNbTetrahedra() && massDensity.size() != 1)
        {
            msg_warning() << "Inconsistent size of massDensity = " << massDensity.size() << ", should be either 1 or " << _topology->getNbTetrahedra();
            return false;
        }
    }
    else if (_topology->getNbQuads()>0 && quadGeo)
    {
        sizeElements = (size_t)_topology->getNbQuads();

        if ( massDensity.size() != (size_t)_topology->getNbQuads() && massDensity.size() != 1)
        {
            msg_warning() << "Inconsistent size of massDensity = " << massDensity.size() << ", should be either 1 or " << _topology->getNbQuads();
            return false;
        }
    }
    else if (_topology->getNbTriangles()>0 && triangleGeo)
    {
        sizeElements = (size_t)_topology->getNbTriangles();

        if ( massDensity.size() != (size_t)_topology->getNbTriangles() && massDensity.size() != 1)
        {
            msg_warning() << "Inconsistent size of massDensity = " << massDensity.size() << ", should be either 1 or " << _topology->getNbTriangles();
            return false;
        }
    }


    //If single value of massDensity is given, propagate it to vector for all elements
    if(massDensity.size() == 1)
    {
        //Check that the massDensity is strictly positive
        if(density <= 0.0)
        {
            msg_warning() << "Negative value of massDensity: massDensity = " << density;
            return false;
        }
        else
        {
            helper::WriteAccessor<Data<sofa::helper::vector< Real > > > massDensityAccess = d_massDensity;
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
            if(massDensity[i]<=0)
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

    computeMass();

    const MassVector &vertexMassInfo = d_vertexMassInfo.getValue();
    Real sumMass = 0.0;
    for (size_t i=0; i<(size_t)_topology->getNbPoints(); i++)
    {
        sumMass += vertexMassInfo[i]*m_massLumpingCoeff;
    }
    d_totalMass.setValue(sumMass);
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::initFromTotalMass()
{
    msg_info() << "totalMass information is used";

    const Real totalMassTemp = d_totalMass.getValue();
    Real sumMass = 0.0;
    setMassDensity(1.0);

    computeMass();

    const MassVector &vertexMassInfo = d_vertexMassInfo.getValue();
    for (size_t i=0; i<(size_t)_topology->getNbPoints(); i++)
    {
        sumMass += vertexMassInfo[i]*m_massLumpingCoeff;
    }
    setMassDensity((Real)totalMassTemp/sumMass);

    computeMass();
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::setVertexMass(sofa::helper::vector< Real > vertexMass)
{
    const sofa::helper::vector< Real > currentVertexMass = d_vertexMass.getValue();
    d_vertexMass.setValue(vertexMass);

    if(!checkVertexMass())
    {
        msg_warning() << "Given values to setVertexMass() are not correct.\n"
                      << "Previous values are used.";
        d_vertexMass.setValue(currentVertexMass);
    }
    else
    {
        d_vertexMassInfo.setValue(vertexMass);
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::setMassDensity(sofa::helper::vector< Real > massDensity)
{
    const sofa::helper::vector< Real > currentMassDensity = d_massDensity.getValue();
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
    const sofa::helper::vector< Real > currentMassDensity = d_massDensity.getValue();
    helper::WriteAccessor<Data<sofa::helper::vector< Real > > > massDensity = d_massDensity;
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
const sofa::helper::vector< typename MeshMatrixMass<DataTypes, MassType>::Real > &  MeshMatrixMass<DataTypes, MassType>::getVertexMass()
{
    return d_vertexMass.getValue();
}


template <class DataTypes, class MassType>
const sofa::helper::vector< typename MeshMatrixMass<DataTypes, MassType>::Real > &  MeshMatrixMass<DataTypes, MassType>::getMassDensity()
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
    MassVector& vertexMass = *d_vertexMassInfo.beginEdit();
    MassVector& edgeMass = *d_edgeMassInfo.beginEdit();
    vertexMass.clear();
    edgeMass.clear();
    d_vertexMassInfo.endEdit();
    d_edgeMassInfo.endEdit();
}


// -- Mass interface
template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addMDx(const core::MechanicalParams*, DataVecDeriv& vres, const DataVecDeriv& vdx, SReal factor)
{
    const MassVector &vertexMass= d_vertexMassInfo.getValue();
    const MassVector &edgeMass= d_edgeMassInfo.getValue();

    helper::WriteAccessor< DataVecDeriv > res = vres;
    helper::ReadAccessor< DataVecDeriv > dx = vdx;

    SReal massTotal = 0.0;

    //using a lumped matrix (default)-----
    if(d_lumping.getValue())
    {
        for (size_t i=0; i<dx.size(); i++)
        {
            res[i] += dx[i] * vertexMass[i] * m_massLumpingCoeff * (Real)factor;
            massTotal += vertexMass[i]*m_massLumpingCoeff * (Real)factor;
        }

    }
    //using a sparse matrix---------------
    else
    {
        size_t nbEdges=_topology->getNbEdges();
        size_t v0,v1;

        for (unsigned int i=0; i<dx.size(); i++)
        {
            res[i] += dx[i] * vertexMass[i] * (Real)factor;
            massTotal += vertexMass[i] * (Real)factor;
        }

        Real tempMass=0.0;

        for (unsigned int j=0; j<nbEdges; ++j)
        {
            tempMass = edgeMass[j] * (Real)factor;

            v0=_topology->getEdge(j)[0];
            v1=_topology->getEdge(j)[1];

            res[v0] += dx[v1] * tempMass;
            res[v1] += dx[v0] * tempMass;

            massTotal += 2*edgeMass[j] * (Real)factor;
        }
    }

    if(d_printMass.getValue() && (this->getContext()->getTime()==0.0))
    {
        msg_info() <<"Total Mass = "<<massTotal;
    }

    if(d_printMass.getValue())
    {
        std::map < std::string, sofa::helper::vector<double> >& graph = *f_graph.beginEdit();
        sofa::helper::vector<double>& graph_error = graph["Mass variations"];
        graph_error.push_back(massTotal+0.000001);

        f_graph.endEdit();
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::accFromF(const core::MechanicalParams*, DataVecDeriv& a, const DataVecDeriv& f)
{
    helper::WriteAccessor< DataVecDeriv > _a = a;
    const VecDeriv& _f = f.getValue();
    const MassVector &vertexMass= d_vertexMassInfo.getValue();

    if(d_lumping.getValue())
    {
        for (unsigned int i=0; i<vertexMass.size(); i++)
        {
            _a[i] = _f[i] / ( vertexMass[i] * m_massLumpingCoeff);
        }
    }
    else
    {
        (void)a;
        (void)f;
        msg_error() << "WARNING: the methode 'accFromF' can't be used with MeshMatrixMass as this SPARSE mass matrix can't be inversed easily. \nPlease proceed to mass lumping.";
        return;
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addForce(const core::MechanicalParams*, DataVecDeriv& vf, const DataVecCoord& , const DataVecDeriv& )
{

    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if(this->m_separateGravity.getValue())
        return ;

    helper::WriteAccessor< DataVecDeriv > f = vf;

    const MassVector &vertexMass= d_vertexMassInfo.getValue();

    // gravity
    defaulttype::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);

    // add weight and inertia force
    for (unsigned int i=0; i<f.size(); ++i)
        f[i] += theGravity * vertexMass[i] * m_massLumpingCoeff;
}


template <class DataTypes, class MassType>
SReal MeshMatrixMass<DataTypes, MassType>::getKineticEnergy( const core::MechanicalParams*, const DataVecDeriv& vv ) const
{
    const MassVector &vertexMass= d_vertexMassInfo.getValue();
    const MassVector &edgeMass= d_edgeMassInfo.getValue();

    helper::ReadAccessor< DataVecDeriv > v = vv;

    unsigned int nbEdges=_topology->getNbEdges();
    unsigned int v0,v1;

    SReal e = 0;

    for (unsigned int i=0; i<v.size(); i++)
    {
        e += dot(v[i],v[i]) * vertexMass[i]; // v[i]*v[i]*masses[i] would be more efficient but less generic
    }

    for (unsigned int i = 0; i < nbEdges; ++i)
    {
        v0 = _topology->getEdge(i)[0];
        v1 = _topology->getEdge(i)[1];

        e += 2 * dot(v[v0], v[v1])*edgeMass[i];

    }

    return e/2;
}


template <class DataTypes, class MassType>
SReal MeshMatrixMass<DataTypes, MassType>::getPotentialEnergy( const core::MechanicalParams*, const DataVecCoord& vx) const
{
    const MassVector &vertexMass= d_vertexMassInfo.getValue();

    helper::ReadAccessor< DataVecCoord > x = vx;

    SReal e = 0;
    // gravity
    defaulttype::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);

    for (unsigned int i=0; i<x.size(); i++)
        e -= dot(theGravity,x[i])*vertexMass[i] * m_massLumpingCoeff;

    return e;
}


// does nothing by default, need to be specialized in .cpp
template <class DataTypes, class MassType>
defaulttype::Vector6 MeshMatrixMass<DataTypes, MassType>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& /*vx*/, const DataVecDeriv& /*vv*/  ) const
{
    return defaulttype::Vector6();
}



template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if(this->mstate && mparams)
    {
        VecDeriv& v = *d_v.beginEdit();

        // gravity
        defaulttype::Vec3d g ( this->getContext()->getGravity() );
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2]);
        Deriv hg = theGravity * (typename DataTypes::Real)(mparams->dt());

        for (unsigned int i=0; i<v.size(); i++)
            v[i] += hg;
        d_v.endEdit();
    }

}




template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    const MassVector &vertexMass= d_vertexMassInfo.getValue();
    const MassVector &edgeMass= d_edgeMassInfo.getValue();

    size_t nbEdges=_topology->getNbEdges();
    size_t v0,v1;

    const int N = defaulttype::DataTypeInfo<Deriv>::size();
    AddMToMatrixFunctor<Deriv,MassType> calc;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = r.matrix;
    Real mFactor = (Real)mparams->mFactorIncludingRayleighDamping(this->rayleighMass.getValue());

    if((int)mat->colSize() != (_topology->getNbPoints()*N) || (int)mat->rowSize() != (_topology->getNbPoints()*N))
    {
        msg_error() <<"Wrong size of the input Matrix: need resize in addMToMatrix function.";
        mat->resize(_topology->getNbPoints()*N,_topology->getNbPoints()*N);
    }

    SReal massTotal=0.0;

    if(d_lumping.getValue())
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
            std::map < std::string, sofa::helper::vector<double> >& graph = *f_graph.beginEdit();
            sofa::helper::vector<double>& graph_error = graph["Mass variations"];
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
            v0 = _topology->getEdge(j)[0];
            v1 = _topology->getEdge(j)[1];

            calc(r.matrix, edgeMass[j], r.offset + N*v0, r.offset + N*v1, mFactor);
            calc(r.matrix, edgeMass[j], r.offset + N*v1, r.offset + N*v0, mFactor);

            massTotal += 2 * edgeMass[j];
        }

        if(d_printMass.getValue() && (this->getContext()->getTime()==0.0))
            msg_info() <<"Total Mass  = "<<massTotal ;

        if(d_printMass.getValue())
        {
            std::map < std::string, sofa::helper::vector<double> >& graph = *f_graph.beginEdit();
            sofa::helper::vector<double>& graph_error = graph["Mass variations"];
            graph_error.push_back(massTotal+0.000001);

            f_graph.endEdit();
        }

    }


}


template <class DataTypes, class MassType>
SReal MeshMatrixMass<DataTypes, MassType>::getElementMass(unsigned int index) const
{
    const MassVector &vertexMass= d_vertexMassInfo.getValue();
    SReal mass = vertexMass[index] * m_massLumpingCoeff;

    return mass;
}


//TODO: special case for Rigid Mass
template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const
{
    static const defaulttype::BaseMatrix::Index dimension = (defaulttype::BaseMatrix::Index) defaulttype::DataTypeInfo<Deriv>::size();
    if (m->rowSize() != dimension || m->colSize() != dimension) m->resize(dimension,dimension);

    m->clear();
    AddMToMatrixFunctor<Deriv,MassType>()(m, d_vertexMassInfo.getValue()[index] * m_massLumpingCoeff, 0, 1);
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::simulation::AnimateEndEvent::checkEventType(event))
    {
        if (d_edgeMassInfo.isDirty() || d_vertexMassInfo.isDirty())
        {
            msg_error() << "edgeMassInfo or vertexMassInfo should not be directly accessed";
        }
        update();
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    const MassVector &vertexMass= d_vertexMassInfo.getValue();

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    Coord gravityCenter;
    Real totalMass=0.0;

    std::vector<  defaulttype::Vector3 > points;
    for (unsigned int i=0; i<x.size(); i++)
    {
        defaulttype::Vector3 p;
        p = DataTypes::getCPos(x[i]);

        points.push_back(p);
        gravityCenter += x[i]*vertexMass[i]*m_massLumpingCoeff;
        totalMass += vertexMass[i]*m_massLumpingCoeff;
    }

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->disableLighting();
    sofa::defaulttype::RGBAColor color(1.0,1.0,1.0,1.0);

    vparams->drawTool()->drawPoints(points, 2, color);

    std::vector<sofa::defaulttype::Vector3> vertices;

    if(d_showCenterOfGravity.getValue())
    {
        color = sofa::defaulttype::RGBAColor(1.0,1.0,0,1.0);
        gravityCenter /= totalMass;
        for(unsigned int i=0 ; i<Coord::spatial_dimensions ; i++)
        {
            Coord v, diff;
            v[i] = d_showAxisSize.getValue();
            diff = gravityCenter-v;
            vertices.push_back(sofa::defaulttype::Vector3(diff));
            diff = gravityCenter+v;
            vertices.push_back(sofa::defaulttype::Vector3(diff));
        }
    }
    vparams->drawTool()->drawLines(vertices,5,color);
    vparams->drawTool()->restoreLastState();
}


} // namespace mass

} // namespace component

} // namespace sofa

#endif
