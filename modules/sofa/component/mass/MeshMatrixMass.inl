/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_MASS_MESHMATRIXMASS_INL
#define SOFA_COMPONENT_MASS_MESHMATRIXMASS_INL

#include <sofa/component/mass/MeshMatrixMass.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/component/topology/TopologyChangedEvent.h>
#include <sofa/component/topology/PointData.inl>
#include <sofa/component/topology/EdgeData.inl>
#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/component/mass/AddMToMatrixFunctor.h>
#include <sofa/simulation/common/Simulation.h>

#ifdef SOFA_SUPPORT_MOVING_FRAMES
#include <sofa/core/behavior/InertiaForce.h>
#endif

namespace sofa
{

namespace component
{

namespace mass
{


using namespace	sofa::component::topology;
using namespace core::topology;

using namespace sofa::defaulttype;
using namespace sofa::core::behavior;


template <class DataTypes, class MassType>
MeshMatrixMass<DataTypes, MassType>::MeshMatrixMass()
    : vertexMassInfo( initData(&vertexMassInfo, "vertexMass", "values of the particles masses on vertices") )
    , edgeMassInfo( initData(&edgeMassInfo, "edgeMass", "values of the particles masses on edges") )
    , m_massDensity( initData(&m_massDensity, (Real)1.0,"massDensity", "mass density that allows to compute the  particles masses from a mesh topology and geometry.\nOnly used if > 0") )
    , showCenterOfGravity( initData(&showCenterOfGravity, false, "showGravityCenter", "display the center of gravity of the system" ) )
    , showAxisSize( initData(&showAxisSize, (Real)1.0, "showAxisSizeFactor", "factor length of the axis displayed (only used for rigids)" ) )
    , lumping( initData(&lumping, true, "lumping","boolean if you need to use a lumped mass matrix") )
    , printMass( initData(&printMass, false, "printMass","boolean if you want to get the totalMass") )
    , f_graph( initData(&f_graph,"graph","Graph of the controlled potential") )
    , topologyType(TOPOLOGY_UNKNOWN)

{
    f_graph.setWidget("graph");
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassCreationFunction(int, void* , MassType & VertexMass,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    VertexMass = 0;
}


template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassCreationFunction(int, void* , MassType & EdgeMass,
        const Edge&,
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
void MeshMatrixMass<DataTypes, MassType>::VertexMassTriangleCreationFunction(const sofa::helper::vector<unsigned int> &triangleAdded,
        void* param, vector<MassType> &VertexMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<triangleAdded.size(); ++i)
        {
            // Get the triangle to be added
            const Triangle &t = MMM->_topology->getTriangle(triangleAdded[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(MMM->triangleGeo)
            {
                mass=(densityM * MMM->triangleGeo->computeRestTriangleArea(triangleAdded[i]))/(typename DataTypes::Real)6.0;
            }

            // Adding mass
            for (unsigned int j=0; j<3; ++j)
                VertexMasses[ t[j] ] += mass;
        }
    }
}


/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassTriangleCreationFunction(const sofa::helper::vector<unsigned int> &triangleAdded,
        void* param, vector<MassType> &EdgeMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<triangleAdded.size(); ++i)
        {
            // Get the edgesInTriangle to be added
            const EdgesInTriangle &te = MMM->_topology->getEdgesInTriangle(triangleAdded[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(MMM->triangleGeo)
            {
                mass=(densityM * MMM->triangleGeo->computeRestTriangleArea(triangleAdded[i]))/(typename DataTypes::Real)12.0;
            }

            // Adding mass edges of concerne triangle
            for (unsigned int j=0; j<3; ++j)
                EdgeMasses[ te[j] ] += mass;
        }
    }
}


/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassTriangleDestroyFunction(const sofa::helper::vector<unsigned int> &triangleRemoved,
        void* param, vector<MassType> &VertexMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<triangleRemoved.size(); ++i)
        {
            // Get the triangle to be removed
            const Triangle &t = MMM->_topology->getTriangle(triangleRemoved[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(MMM->triangleGeo)
            {
                mass=(densityM * MMM->triangleGeo->computeRestTriangleArea(triangleRemoved[i]))/(typename DataTypes::Real)6.0;
            }

            // Removing mass
            for (unsigned int j=0; j<3; ++j)
                VertexMasses[ t[j] ] -= mass;
        }
    }
}


/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassTriangleDestroyFunction(const sofa::helper::vector<unsigned int> &triangleRemoved,
        void* param, vector<MassType> &EdgeMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<triangleRemoved.size(); ++i)
        {
            // Get the triangle to be removed
            const EdgesInTriangle &te = MMM->_topology->getEdgesInTriangle(triangleRemoved[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(MMM->triangleGeo)
            {
                mass=(densityM * MMM->triangleGeo->computeRestTriangleArea(triangleRemoved[i]))/(typename DataTypes::Real)12.0;
            }

            // Removing mass edges of concerne triangle
            for (unsigned int j=0; j<3; ++j)
                EdgeMasses[ te[j] ] -= mass;
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
void MeshMatrixMass<DataTypes, MassType>::VertexMassQuadCreationFunction(const sofa::helper::vector<unsigned int> &quadAdded,
        void* param, vector<MassType> &VertexMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_QUADSET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<quadAdded.size(); ++i)
        {
            // Get the quad to be added
            const Quad &q = MMM->_topology->getQuad(quadAdded[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(MMM->quadGeo)
            {
                mass=(densityM * MMM->quadGeo->computeRestQuadArea(quadAdded[i]))/(typename DataTypes::Real)8.0;
            }

            // Adding mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ q[j] ] += mass;
        }
    }
}


/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassQuadCreationFunction(const sofa::helper::vector<unsigned int> &quadAdded,
        void* param, vector<MassType> &EdgeMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_QUADSET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<quadAdded.size(); ++i)
        {
            // Get the EdgesInQuad to be added
            const EdgesInQuad &qe = MMM->_topology->getEdgesInQuad(quadAdded[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(MMM->quadGeo)
            {
                mass=(densityM * MMM->quadGeo->computeRestQuadArea(quadAdded[i]))/(typename DataTypes::Real)16.0;
            }

            // Adding mass edges of concerne quad
            for (unsigned int j=0; j<4; ++j)
                EdgeMasses[ qe[j] ] += mass;
        }
    }
}


/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassQuadDestroyFunction(const sofa::helper::vector<unsigned int> &quadRemoved,
        void* param, vector<MassType> &VertexMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_QUADSET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<quadRemoved.size(); ++i)
        {
            // Get the quad to be removed
            const Quad &q = MMM->_topology->getQuad(quadRemoved[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(MMM->quadGeo)
            {
                mass=(densityM * MMM->quadGeo->computeRestQuadArea(quadRemoved[i]))/(typename DataTypes::Real)8.0;
            }

            // Removing mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ q[j] ] -= mass;
        }
    }
}


/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassQuadDestroyFunction(const sofa::helper::vector<unsigned int> &quadRemoved,
        void* param, vector<MassType> &EdgeMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_QUADSET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<quadRemoved.size(); ++i)
        {
            // Get the EdgesInQuad to be removed
            const EdgesInQuad &qe = MMM->_topology->getEdgesInQuad(quadRemoved[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(MMM->quadGeo)
            {
                mass=(densityM * MMM->quadGeo->computeRestQuadArea(quadRemoved[i]))/(typename DataTypes::Real)16.0;
            }

            // Removing mass edges of concerne quad
            for (unsigned int j=0; j<4; ++j)
                EdgeMasses[ qe[j] ] -= mass/2;
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
void MeshMatrixMass<DataTypes, MassType>::VertexMassTetrahedronCreationFunction(const sofa::helper::vector<unsigned int> &tetrahedronAdded,
        void* param, vector<MassType> &VertexMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<tetrahedronAdded.size(); ++i)
        {
            // Get the tetrahedron to be added
            const Tetrahedron &t = MMM->_topology->getTetrahedron(tetrahedronAdded[i]);

            // Compute rest mass of conserne tetrahedron = density * tetrahedron volume.
            if(MMM->tetraGeo)
            {
                mass=(densityM * MMM->tetraGeo->computeRestTetrahedronVolume(tetrahedronAdded[i]))/(typename DataTypes::Real)10.0;
            }

            // Adding mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ t[j] ] += mass;
        }
    }

}


/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassTetrahedronCreationFunction(const sofa::helper::vector<unsigned int> &tetrahedronAdded,
        void* param, vector<MassType> &EdgeMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<tetrahedronAdded.size(); ++i)
        {
            // Get the edgesInTetrahedron to be added
            const EdgesInTetrahedron &te = MMM->_topology->getEdgesInTetrahedron(tetrahedronAdded[i]);

            // Compute rest mass of conserne triangle = density * tetrahedron volume.
            if(MMM->tetraGeo)
            {
                mass=(densityM * MMM->tetraGeo->computeRestTetrahedronVolume(tetrahedronAdded[i]))/(typename DataTypes::Real)20.0;
            }

            // Adding mass edges of concerne triangle
            for (unsigned int j=0; j<6; ++j)
                EdgeMasses[ te[j] ] += mass;
        }
    }
}


/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassTetrahedronDestroyFunction(const sofa::helper::vector<unsigned int> &tetrahedronRemoved,
        void* param, vector<MassType> &VertexMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<tetrahedronRemoved.size(); ++i)
        {
            // Get the tetrahedron to be removed
            const Tetrahedron &t = MMM->_topology->getTetrahedron(tetrahedronRemoved[i]);

            // Compute rest mass of conserne tetrahedron = density * tetrahedron volume.
            if(MMM->tetraGeo)
            {
                mass=(densityM * MMM->tetraGeo->computeRestTetrahedronVolume(tetrahedronRemoved[i]))/(typename DataTypes::Real)10.0;
            }

            // Removing mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ t[j] ] -= mass;
        }
    }
}


/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassTetrahedronDestroyFunction(const sofa::helper::vector<unsigned int> &tetrahedronRemoved,
        void* param, vector<MassType> &EdgeMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<tetrahedronRemoved.size(); ++i)
        {
            // Get the edgesInTetrahedron to be removed
            const EdgesInTetrahedron &te = MMM->_topology->getEdgesInTetrahedron(tetrahedronRemoved[i]);

            // Compute rest mass of conserne triangle = density * tetrahedron volume.
            if(MMM->tetraGeo)
            {
                mass=(densityM * MMM->tetraGeo->computeRestTetrahedronVolume(tetrahedronRemoved[i]))/(typename DataTypes::Real)20.0;
            }

            // Removing mass edges of concerne triangle
            for (unsigned int j=0; j<6; ++j)
                EdgeMasses[ te[j] ] -= mass; //?
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
void MeshMatrixMass<DataTypes, MassType>::VertexMassHexahedronCreationFunction(const sofa::helper::vector<unsigned int> &hexahedronAdded,
        void* param, vector<MassType> &VertexMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_HEXAHEDRONSET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<hexahedronAdded.size(); ++i)
        {
            // Get the hexahedron to be added
            const Hexahedron &h = MMM->_topology->getHexahedron(hexahedronAdded[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(MMM->hexaGeo)
            {
                mass=(densityM * MMM->hexaGeo->computeRestHexahedronVolume(hexahedronAdded[i]))/(typename DataTypes::Real)20.0;
            }

            // Adding mass
            for (unsigned int j=0; j<8; ++j)
                VertexMasses[ h[j] ] += mass;
        }
    }
}


/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHexahedronCreationFunction(const sofa::helper::vector<unsigned int> &hexahedronAdded,
        void* param, vector<MassType> &EdgeMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_HEXAHEDRONSET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<hexahedronAdded.size(); ++i)
        {
            // Get the EdgesInHexahedron to be added
            const EdgesInHexahedron &he = MMM->_topology->getEdgesInHexahedron(hexahedronAdded[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(MMM->hexaGeo)
            {
                mass=(densityM * MMM->hexaGeo->computeRestHexahedronVolume(hexahedronAdded[i]))/(typename DataTypes::Real)40.0;
            }

            // Adding mass edges of concerne triangle
            for (unsigned int j=0; j<12; ++j)
                EdgeMasses[ he[j] ] += mass;
        }
    }
}


/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHexahedronDestroyFunction(const sofa::helper::vector<unsigned int> &hexahedronRemoved,
        void* param, vector<MassType> &VertexMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_HEXAHEDRONSET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<hexahedronRemoved.size(); ++i)
        {
            // Get the hexahedron to be removed
            const Hexahedron &h = MMM->_topology->getHexahedron(hexahedronRemoved[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(MMM->hexaGeo)
            {
                mass=(densityM * MMM->hexaGeo->computeRestHexahedronVolume(hexahedronRemoved[i]))/(typename DataTypes::Real)20.0;
            }

            // Removing mass
            for (unsigned int j=0; j<8; ++j)
                VertexMasses[ h[j] ] -= mass;
        }
    }
}


/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHexahedronDestroyFunction(const sofa::helper::vector<unsigned int> &hexahedronRemoved,
        void* param, vector<MassType> &EdgeMasses)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = (MeshMatrixMass<DataTypes, MassType> *)param;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_HEXAHEDRONSET)
    {
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<hexahedronRemoved.size(); ++i)
        {
            // Get the EdgesInHexahedron to be removed
            const EdgesInHexahedron &he = MMM->_topology->getEdgesInHexahedron(hexahedronRemoved[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(MMM->hexaGeo)
            {
                mass=(densityM * MMM->hexaGeo->computeRestHexahedronVolume(hexahedronRemoved[i]))/(typename DataTypes::Real)40.0;
            }

            // Removing mass edges of concerne triangle
            for (unsigned int j=0; j<12; ++j)
                EdgeMasses[ he[j] ] -= mass;
        }
    }
}

// }



template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::init()
{
    /*  using sofa::component::topology::RegularGridTopology;
    RegularGridTopology* reg = dynamic_cast<RegularGridTopology*>( this->getContext()->getMeshTopology() );
    if( reg != NULL )
    {
    Real weight = reg->getDx().norm() * reg->getDy().norm() * reg->getDz().norm() * m_massDensity.getValue()/8;
    VecMass& m = *f_mass.beginEdit();
    for( int i=0; i<reg->getNx()-1; i++ )
    {
    for( int j=0; j<reg->getNy()-1; j++ )
    {
        for( int k=0; k<reg->getNz()-1; k++ )
        {
    m[reg->point(i,j,k)] += weight;
    m[reg->point(i,j,k+1)] += weight;
    m[reg->point(i,j+1,k)] += weight;
    m[reg->point(i,j+1,k+1)] += weight;
    m[reg->point(i+1,j,k)] += weight;
    m[reg->point(i+1,j,k+1)] += weight;
    m[reg->point(i+1,j+1,k)] += weight;
    m[reg->point(i+1,j+1,k+1)] += weight;
        }
    }
    }
    f_mass.endEdit();
    }*/

    this->Inherited::init();
    massLumpingCoeff = 0.0;

    _topology = this->getContext()->getMeshTopology();
    savedMass = m_massDensity.getValue();

    //    sofa::core::objectmodel::Tag mechanicalTag(m_tagMeshMechanics.getValue());
    //    this->getContext()->get(triangleGeo, mechanicalTag,sofa::core::objectmodel::BaseContext::SearchUp);

    this->getContext()->get(edgeGeo);
    this->getContext()->get(triangleGeo);
    this->getContext()->get(quadGeo);
    this->getContext()->get(tetraGeo);
    this->getContext()->get(hexaGeo);

    // add the functions to handle topology changes for Vertex informations
    vertexMassInfo.setCreateFunction(VertexMassCreationFunction);
    vertexMassInfo.setCreateParameter ((void *) this );
    // Triangle
    vertexMassInfo.setCreateTriangleFunction(VertexMassTriangleCreationFunction);
    vertexMassInfo.setDestroyTriangleFunction(VertexMassTriangleDestroyFunction);
    // Quad
    vertexMassInfo.setCreateQuadFunction(VertexMassQuadCreationFunction);
    vertexMassInfo.setDestroyQuadFunction(VertexMassQuadDestroyFunction);
    // Tetrahedron
    vertexMassInfo.setCreateTetrahedronFunction(VertexMassTetrahedronCreationFunction);
    vertexMassInfo.setDestroyTetrahedronFunction(VertexMassTetrahedronDestroyFunction);
    // Hexahedron (NOT HANDLE YET)
    //vertexMassInfo.setCreateHexahedronFunction(VertexMassHexahedronCreationFunction);
    //vertexMassInfo.setDestroyHexahedronFunction(VertexMassHexahedronDestroyFunction);


    // add the functions to handle topology changes for Edge informations
    edgeMassInfo.setCreateFunction(EdgeMassCreationFunction);
    edgeMassInfo.setCreateParameter ((void *) this );
    // Triangle
    edgeMassInfo.setCreateTriangleFunction(EdgeMassTriangleCreationFunction);
    edgeMassInfo.setDestroyTriangleFunction(EdgeMassTriangleDestroyFunction);
    // Quad
    edgeMassInfo.setCreateQuadFunction(EdgeMassQuadCreationFunction);
    edgeMassInfo.setDestroyQuadFunction(EdgeMassQuadDestroyFunction);
    // Tetrahedron
    edgeMassInfo.setCreateTetrahedronFunction(EdgeMassTetrahedronCreationFunction);
    edgeMassInfo.setDestroyTetrahedronFunction(EdgeMassTetrahedronDestroyFunction);
    // Hexahedron (NOT HANDLE YET)
    //edgeMassInfo.setCreateHexahedronFunction(EdgeMassHexahedronCreationFunction);
    //edgeMassInfo.setDestroyHexahedronFunction(EdgeMassHexahedronDestroyFunction);

    if ((vertexMassInfo.getValue().size()==0 || edgeMassInfo.getValue().size()==0) && (_topology!=0))
        reinit();

    //Reset the graph
    f_graph.beginEdit()->clear();
    f_graph.endEdit();
}

template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::reinit()
{

    if (_topology && ((m_massDensity.getValue() > 0 && (vertexMassInfo.getValue().size() == 0 || edgeMassInfo.getValue().size() == 0)) || (m_massDensity.getValue()!= savedMass) ))
    {
        // resize array
        clear();

        /// prepare to store info in the vertex array
        vector<MassType>& my_vertexMassInfo = *vertexMassInfo.beginEdit();
        vector<MassType>& my_edgeMassInfo = *edgeMassInfo.beginEdit();

        unsigned int ndof = this->mstate->getSize();
        unsigned int nbEdges=_topology->getNbEdges();

        my_vertexMassInfo.resize(ndof);
        my_edgeMassInfo.resize(nbEdges);

        // set vertex tensor to 0
        for (unsigned int i = 0; i<ndof; ++i)
            VertexMassCreationFunction (i, (void*) this, my_vertexMassInfo[i],
                    (const sofa::helper::vector< unsigned int > )0,
                    (const sofa::helper::vector< double >)0);

        // set edge tensor to 0
        for (unsigned int i = 0; i<nbEdges; ++i)
            EdgeMassCreationFunction (i, (void*) this, my_edgeMassInfo[i], _topology->getEdge(0),
                    (const sofa::helper::vector< unsigned int > )0,
                    (const sofa::helper::vector< double >)0);


        // Create mass matrix depending on current Topology:
        if (_topology->getNbHexahedra()>0 && hexaGeo)  // Hexahedron topology
        {
            // create vector tensor by calling the hexahedron creation function on the entire mesh
            sofa::helper::vector<unsigned int> hexahedraAdded;
            setMassTopologyType(TOPOLOGY_HEXAHEDRONSET);

            for (int i = 0; i<_topology->getNbHexahedra(); ++i)
                hexahedraAdded.push_back(i);

            VertexMassHexahedronCreationFunction(hexahedraAdded, (void*) this, my_vertexMassInfo);
            EdgeMassHexahedronCreationFunction(hexahedraAdded, (void*) this, my_edgeMassInfo);
            massLumpingCoeff = 2.5;
        }
        else if (_topology->getNbTetrahedra()>0 && tetraGeo)  // Tetrahedron topology
        {
            // create vector tensor by calling the tetrahedron creation function on the entire mesh
            sofa::helper::vector<unsigned int> tetrahedraAdded;
            setMassTopologyType(TOPOLOGY_TETRAHEDRONSET);

            for (int i = 0; i<_topology->getNbTetrahedra(); ++i)
                tetrahedraAdded.push_back(i);

            VertexMassTetrahedronCreationFunction(tetrahedraAdded, (void*) this, my_vertexMassInfo);
            EdgeMassTetrahedronCreationFunction(tetrahedraAdded, (void*) this, my_edgeMassInfo);
            massLumpingCoeff = 2.5;
        }

        else if (_topology->getNbQuads()>0 && quadGeo)  // Quad topology
        {
            // create vector tensor by calling the quad creation function on the entire mesh
            sofa::helper::vector<unsigned int> quadsAdded;
            setMassTopologyType(TOPOLOGY_QUADSET);

            for (int i = 0; i<_topology->getNbQuads(); ++i)
                quadsAdded.push_back(i);

            VertexMassQuadCreationFunction(quadsAdded, (void*) this, my_vertexMassInfo);
            EdgeMassQuadCreationFunction(quadsAdded, (void*) this, my_edgeMassInfo);
            massLumpingCoeff = 2.0;
        }
        else if (_topology->getNbTriangles()>0 && triangleGeo) // Triangle topology
        {
            // create vector tensor by calling the triangle creation function on the entire mesh
            sofa::helper::vector<unsigned int> trianglesAdded;
            setMassTopologyType(TOPOLOGY_TRIANGLESET);

            for (int i = 0; i<_topology->getNbTriangles(); ++i)
                trianglesAdded.push_back(i);

            VertexMassTriangleCreationFunction(trianglesAdded, (void*) this, my_vertexMassInfo);
            EdgeMassTriangleCreationFunction(trianglesAdded, (void*) this, my_edgeMassInfo);
            massLumpingCoeff = 2.0;
        }
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::handleTopologyChange()
{
    std::list<const TopologyChange *>::const_iterator itBegin=_topology->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=_topology->endChange();

    vertexMassInfo.handleTopologyEvents(itBegin,itEnd);
    edgeMassInfo.handleTopologyEvents(itBegin,itEnd);
}



template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::clear()
{
    MassVector& vertexMass = *vertexMassInfo.beginEdit();
    MassVector& edgeMass = *edgeMassInfo.beginEdit();
    vertexMass.clear();
    edgeMass.clear();
    vertexMassInfo.endEdit();
    edgeMassInfo.endEdit();
}


// -- Mass interface
template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addMDx(const core::MechanicalParams* /* PARAMS FIRST */, DataVecDeriv& vres, const DataVecDeriv& vdx, double factor)
{
    const MassVector &vertexMass= vertexMassInfo.getValue();
    const MassVector &edgeMass= edgeMassInfo.getValue();

    helper::WriteAccessor< DataVecDeriv > res = vres;
    helper::ReadAccessor< DataVecDeriv > dx = vdx;

    double massTotal = 0.0;

    //using a lumped matrix (default)-----
    if(this->lumping.getValue())
    {
        for (unsigned int i=0; i<dx.size(); i++)
        {
            res[i] += dx[i] * vertexMass[i] * massLumpingCoeff * (Real)factor;
            massTotal += vertexMass[i]*massLumpingCoeff * (Real)factor;
        }
        if(printMass.getValue() && (this->getContext()->getTime()==0.0))
            std::cout<<"Mass totale = "<<massTotal<<std::endl;

        if(printMass.getValue())
        {
            std::map < std::string, sofa::helper::vector<double> >& graph = *f_graph.beginEdit();
            sofa::helper::vector<double>& graph_error = graph["Mass variations"];
            graph_error.push_back(massTotal);

            f_graph.endEdit();
        }
    }


    //using a sparse matrix---------------
    else
    {
        unsigned int nbEdges=_topology->getNbEdges();
        unsigned int v0,v1;

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

        if(printMass.getValue() && (this->getContext()->getTime()==0.0))
            std::cout<<"Mass totale = "<<massTotal<<std::endl;

        if(printMass.getValue())
        {
            std::map < std::string, sofa::helper::vector<double> >& graph = *f_graph.beginEdit();
            sofa::helper::vector<double>& graph_error = graph["Mass variations"];
            graph_error.push_back(massTotal+0.000001);

            f_graph.endEdit();
        }
    }

}



template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::accFromF(const core::MechanicalParams* /* PARAMS FIRST */, DataVecDeriv& a, const DataVecDeriv& f)
{
    (void)a;
    (void)f;

    serr << "WARNING: the methode 'accFromF' can't be used with MeshMatrixMass as this SPARSE mass matrix can't be inversed easily. \nPlease proceed to mass lumping." << sendl;
    return;
}




#ifdef SOFA_SUPPORT_MOVING_FRAMES
template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addForce(const core::MechanicalParams* /* PARAMS FIRST */, DataVecDeriv& vf, const DataVecCoord& vx, const DataVecDeriv& vv)
{
    helper::WriteAccessor< DataVecDeriv > f = vf;
    helper::ReadAccessor< DataVecCoord > x = vx;
    helper::ReadAccessor< DataVecDeriv > v = vv;

    const MassVector &vertexMass= vertexMassInfo.getValue();

    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);

    // velocity-based stuff
    core::objectmodel::BaseContext::SpatialVector vframe = this->getContext()->getVelocityInWorld();
    core::objectmodel::BaseContext::Vec3 aframe = this->getContext()->getVelocityBasedLinearAccelerationInWorld() ;

    // project back to local frame
    vframe = this->getContext()->getPositionInWorld() / vframe;
    aframe = this->getContext()->getPositionInWorld().backProjectVector( aframe );

    // add weight and inertia force
    if(this->m_separateGravity.getValue())
        for (unsigned int i=0; i<x.size(); ++i)
            f[i] += massLumpingCoeff + core::behavior::inertiaForce(vframe,aframe,vertexMass[i] * massLumpingCoeff ,x[i],v[i]);
    else for (unsigned int i=0; i<x.size(); ++i)
            f[i] += theGravity * vertexMass[i] * massLumpingCoeff + core::behavior::inertiaForce(vframe,aframe,vertexMass[i] * massLumpingCoeff ,x[i],v[i]);
}
#else
template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addForce(const core::MechanicalParams* /* PARAMS FIRST */, DataVecDeriv& vf, const DataVecCoord& , const DataVecDeriv& )
{

    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if(this->m_separateGravity.getValue())
        return ;

    helper::WriteAccessor< DataVecDeriv > f = vf;

    const MassVector &vertexMass= vertexMassInfo.getValue();

    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);

    // add weight and inertia force
    for (unsigned int i=0; i<f.size(); ++i)
        f[i] += theGravity * vertexMass[i] * massLumpingCoeff;
}
#endif


template <class DataTypes, class MassType>
double MeshMatrixMass<DataTypes, MassType>::getKineticEnergy( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecDeriv& vv ) const
{
    const MassVector &vertexMass= vertexMassInfo.getValue();
    const MassVector &edgeMass= edgeMassInfo.getValue();

    helper::ReadAccessor< DataVecDeriv > v = vv;

    unsigned int nbEdges=_topology->getNbEdges();
    unsigned int v0,v1;

    double e = 0;

    for (unsigned int i=0; i<v.size(); i++)
    {
        e += dot(v[i],v[i]) * vertexMass[i]; // v[i]*v[i]*masses[i] would be more efficient but less generic
    }

    for (unsigned int i=0; i<nbEdges; ++i)
    {
        v0=_topology->getEdge(i)[0];
        v1=_topology->getEdge(i)[1];

        e += 2*dot(v[v0],v[v1])*edgeMass[i];
    }

    return e/2;
}


template <class DataTypes, class MassType>
double MeshMatrixMass<DataTypes, MassType>::getPotentialEnergy( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx) const
{
    const MassVector &vertexMass= vertexMassInfo.getValue();

    helper::ReadAccessor< DataVecCoord > x = vx;

    SReal e = 0;
    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);

    for (unsigned int i=0; i<x.size(); i++)
        e -= dot(theGravity,x[i])*vertexMass[i] * massLumpingCoeff;

    return e;
}



template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addGravityToV(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_v)
{
    if(this->mstate && mparams)
    {
        VecDeriv& v = *d_v.beginEdit();

        // gravity
        Vec3d g ( this->getContext()->getGravity() );
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2]);
        Deriv hg = theGravity * (typename DataTypes::Real)(mparams->dt());

        for (unsigned int i=0; i<v.size(); i++)
            v[i] += hg;
        d_v.endEdit();
    }

}




template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addMToMatrix(const core::MechanicalParams *mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    const MassVector &vertexMass= vertexMassInfo.getValue();
    const MassVector &edgeMass= edgeMassInfo.getValue();

    unsigned int nbEdges=_topology->getNbEdges();
    unsigned int v0,v1;

    const int N = defaulttype::DataTypeInfo<Deriv>::size();
    AddMToMatrixFunctor<Deriv,MassType> calc;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    Real mFactor = (Real)mparams->mFactor();

    double massTotal=0.0;

    if(this->lumping.getValue())
    {
        for (unsigned int i=0; i<vertexMass.size(); i++)
        {
            calc(r.matrix, vertexMass[i] * massLumpingCoeff, r.offset + N*i, mFactor);
            massTotal += vertexMass[i] * massLumpingCoeff;
        }

        if(printMass.getValue() && (this->getContext()->getTime()==0.0))
            std::cout<<"Mass totale = "<<massTotal<<std::endl;

        if(printMass.getValue())
        {
            std::map < std::string, sofa::helper::vector<double> >& graph = *f_graph.beginEdit();
            sofa::helper::vector<double>& graph_error = graph["Mass variations"];
            graph_error.push_back(massTotal);

            f_graph.endEdit();
        }
    }


    else
    {
        for (unsigned int i=0; i<vertexMass.size(); i++)
        {
            calc(r.matrix, vertexMass[i], r.offset + N*i, mFactor);
            massTotal += vertexMass[i];
        }


        for (unsigned int j=0; j<nbEdges; ++j)
        {
            v0=_topology->getEdge(j)[0];
            v1=_topology->getEdge(j)[1];

            calc(r.matrix, edgeMass[j], r.offset + N*v0, r.offset + N*v1, mFactor);
            calc(r.matrix, edgeMass[j], r.offset + N*v1, r.offset + N*v0, mFactor);

            massTotal += 2*edgeMass[j];
        }

        if(printMass.getValue() && (this->getContext()->getTime()==0.0))
            std::cout<<"Mass totale = "<<massTotal<<std::endl;

        if(printMass.getValue())
        {
            std::map < std::string, sofa::helper::vector<double> >& graph = *f_graph.beginEdit();
            sofa::helper::vector<double>& graph_error = graph["Mass variations"];
            graph_error.push_back(massTotal+0.000001);

            f_graph.endEdit();
        }

    }


}





template <class DataTypes, class MassType>
double MeshMatrixMass<DataTypes, MassType>::getElementMass(unsigned int index) const
{
    const MassVector &vertexMass= vertexMassInfo.getValue();
    double mass = vertexMass[index] * massLumpingCoeff;

    return mass;
}



//TODO: special case for Rigid Mass
template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const
{
    const unsigned int dimension = defaulttype::DataTypeInfo<Deriv>::size();
    if (m->rowSize() != dimension || m->colSize() != dimension) m->resize(dimension,dimension);

    m->clear();
    AddMToMatrixFunctor<Deriv,MassType>()(m, vertexMassInfo.getValue()[index] * massLumpingCoeff, 0, 1);
}




template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    const MassVector &vertexMass= vertexMassInfo.getValue();
    const MassVector &edgeMass= edgeMassInfo.getValue();

    unsigned int nbEdges=_topology->getNbEdges();
    unsigned int v0,v1;

    const VecCoord& x = *this->mstate->getX();
    Coord gravityCenter;
    Real totalMass=0.0;

    std::vector<  Vector3 > points;

    if(this->lumping.getValue())
    {
        for (unsigned int i=0; i<x.size(); i++)
        {
            Vector3 p;
            p = DataTypes::getCPos(x[i]);

            points.push_back(p);
            gravityCenter += x[i]*vertexMass[i]*massLumpingCoeff;
            totalMass += vertexMass[i]*massLumpingCoeff;
        }
    }
    else
    {
        for (unsigned int i=0; i<x.size(); i++)
        {
            Vector3 p;
            p = DataTypes::getCPos(x[i]);

            points.push_back(p);
            gravityCenter += x[i]*vertexMass[i];
            totalMass += vertexMass[i];
        }

        for (unsigned int i=0; i<nbEdges; ++i)
        {
            v0=_topology->getEdge(i)[0];
            v1=_topology->getEdge(i)[1];

            gravityCenter += x[v0]*edgeMass[v0];
            gravityCenter += x[v1]*edgeMass[v1];
            totalMass += edgeMass[v0];
            totalMass += edgeMass[v1];
        }
    }


    vparams->drawTool()->drawPoints(points, 2, Vec<4,float>(1,1,1,1));

    if(showCenterOfGravity.getValue())
    {
        glBegin (GL_LINES);
        glColor4f (1,1,0,1);
        glPointSize(5);
        gravityCenter /= totalMass;
        for(unsigned int i=0 ; i<Coord::spatial_dimensions ; i++)
        {
            Coord v;
            v[i] = showAxisSize.getValue();
            helper::gl::glVertexT(gravityCenter-v);
            helper::gl::glVertexT(gravityCenter+v);
        }
        glEnd();
    }
}

template <class DataTypes, class MassType>
bool MeshMatrixMass<DataTypes, MassType>::addBBox(double* minBBox, double* maxBBox)
{
    const VecCoord& x = *this->mstate->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        //const Coord& p = x[i];
        Real p[3] = {0.0, 0.0, 0.0};
        DataTypes::get(p[0],p[1],p[2],x[i]);
        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    return true;
}


} // namespace mass

} // namespace component

} // namespace sofa

#endif
