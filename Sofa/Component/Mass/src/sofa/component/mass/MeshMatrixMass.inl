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

#include <sofa/component/mass/MeshMatrixMass.h>
#include <sofa/core/behavior/Mass.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/core/topology/TopologyData.inl>
#include <sofa/component/mass/AddMToMatrixFunctor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/vector.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <numeric>

#include <sofa/core/behavior/BaseLocalMassMatrix.h>

namespace sofa::component::mass
{
using namespace sofa::core::topology;

template <class DataTypes, class GeometricalTypes>
MeshMatrixMass<DataTypes, GeometricalTypes>::MeshMatrixMass()
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
    , l_geometryState(initLink("geometryState", "link to the MechanicalObject associated with the geometry"))
    , m_massTopologyType(geometry::ElementType::UNKNOWN)
{
    f_graph.setWidget("graph");
}

template <class DataTypes, class GeometricalTypes>
MeshMatrixMass<DataTypes, GeometricalTypes>::~MeshMatrixMass()
{

}

template< class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyVertexMassCreation(Index, MassType & VertexMass,
    const core::topology::BaseMeshTopology::Point&,
    const sofa::type::vector< Index > &,
    const sofa::type::vector< SReal >&)
{
    VertexMass = 0;
}

template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyEdgeMassCreation(Index, MassType & EdgeMass,
    const core::topology::BaseMeshTopology::Edge&,
    const sofa::type::vector< Index > &,
    const sofa::type::vector< SReal >&)
{
    EdgeMass = 0;
}


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyVertexMassDestruction(Index id, MassType& VertexMass)
{
    SOFA_UNUSED(id);
    auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);
    totalMass -= VertexMass;
}


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyEdgeMassDestruction(Index id, MassType& EdgeMass)
{
    SOFA_UNUSED(id);
    if(!isLumped())
    {
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);
        totalMass -= EdgeMass;
    }
}



// -------------------------------------------------------
// ------- Triangle Creation/Destruction functions -------
// -------------------------------------------------------
//{

/// Creation fonction for mass stored on vertices
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 2, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyVertexMassTriangleCreation(const sofa::type::vector< Index >& triangleAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Triangle >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs)
{
    SOFA_UNUSED(elems);

    if (this->getMassTopologyType() == geometry::ElementType::TRIANGLE)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses (d_vertexMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);        

        // update mass density vector
        const std::size_t nbMass = getMassDensity().size();
        const sofa::Size nbTri = this->l_topology->getNbTriangles();
        if (nbMass < nbTri)
            addMassDensity(triangleAdded, ancestors, coefs);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < triangleAdded.size(); ++i)
        {
            // Get the triangle to be added
            const core::topology::BaseMeshTopology::Triangle& t = this->l_topology->getTriangle(triangleAdded[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            const auto& pos0 = GeometricalTypes::getCPos(positions[t[0]]);
            const auto& pos1 = GeometricalTypes::getCPos(positions[t[1]]);
            const auto& pos2 = GeometricalTypes::getCPos(positions[t[2]]);

            const auto triangleArea = sofa::geometry::Triangle::area(pos0, pos1, pos2);
            mass = (densityM[triangleAdded[i]] * triangleArea) / (typename DataTypes::Real(6.0));

            // Adding mass
            for (unsigned int j = 0; j < 3; ++j)
                VertexMasses[t[j]] += mass;

            // update total mass: 
            if (!this->isLumped())
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
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 2, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyEdgeMassTriangleCreation(const sofa::type::vector< Index >& triangleAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Triangle >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs)
{
    SOFA_UNUSED(elems);

    if (this->getMassTopologyType() == geometry::ElementType::TRIANGLE)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses ( d_edgeMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);

        // update mass density vector
        const std::size_t nbMass = getMassDensity().size();
        const sofa::Size nbTri = this->l_topology->getNbTriangles();
        if (nbMass < nbTri)
            addMassDensity(triangleAdded, ancestors, coefs);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < triangleAdded.size(); ++i)
        {
            // Get the edgesInTriangle to be added
            const core::topology::BaseMeshTopology::EdgesInTriangle& te = this->l_topology->getEdgesInTriangle(triangleAdded[i]);
            const core::topology::BaseMeshTopology::Triangle& t = this->l_topology->getTriangle(triangleAdded[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            const auto& pos0 = GeometricalTypes::getCPos(positions[t[0]]);
            const auto& pos1 = GeometricalTypes::getCPos(positions[t[1]]);
            const auto& pos2 = GeometricalTypes::getCPos(positions[t[2]]);

            const auto triangleArea = sofa::geometry::Triangle::area(pos0, pos1, pos2);
            mass = (densityM[triangleAdded[i]] * triangleArea) / (typename DataTypes::Real(12.0));

            // Adding mass edges of concerne triangle
            for (unsigned int j = 0; j < 3; ++j)
                EdgeMasses[te[j]] += mass;

            // update total mass
            if (!this->isLumped())
            {
                totalMass += 3.0 * mass * 2.0; // x 2 because mass is actually splitted over half-edges
            }
        }
    }
}

/// Destruction fonction for mass stored on vertices
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 2, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyVertexMassTriangleDestruction(const sofa::type::vector< Index >& triangleRemoved)
{
    if (this->getMassTopologyType() == geometry::ElementType::TRIANGLE)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses (d_vertexMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < triangleRemoved.size(); ++i)
        {
            // Get the triangle to be removed
            const core::topology::BaseMeshTopology::Triangle& t = this->l_topology->getTriangle(triangleRemoved[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            const auto& pos0 = GeometricalTypes::getCPos(positions[t[0]]);
            const auto& pos1 = GeometricalTypes::getCPos(positions[t[1]]);
            const auto& pos2 = GeometricalTypes::getCPos(positions[t[2]]);

            const auto triangleArea = sofa::geometry::Triangle::area(pos0, pos1, pos2);
            mass = (densityM[triangleRemoved[i]] * triangleArea) / (typename DataTypes::Real(6.0));

            // Removing mass
            for (unsigned int j = 0; j < 3; ++j)
                VertexMasses[t[j]] -= mass;

            // update total mass
            if (!this->isLumped())
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
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 2, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyEdgeMassTriangleDestruction(const sofa::type::vector< Index >& triangleRemoved)
{
    if (this->getMassTopologyType() == geometry::ElementType::TRIANGLE)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses (d_edgeMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < triangleRemoved.size(); ++i)
        {
            // Get the triangle to be removed
            const core::topology::BaseMeshTopology::EdgesInTriangle& te = this->l_topology->getEdgesInTriangle(triangleRemoved[i]);
            const core::topology::BaseMeshTopology::Triangle& t = this->l_topology->getTriangle(triangleRemoved[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            const auto& pos0 = GeometricalTypes::getCPos(positions[t[0]]);
            const auto& pos1 = GeometricalTypes::getCPos(positions[t[1]]);
            const auto& pos2 = GeometricalTypes::getCPos(positions[t[2]]);

            const auto triangleArea = sofa::geometry::Triangle::area(pos0, pos1, pos2);
            mass = (densityM[triangleRemoved[i]] * triangleArea) / (typename DataTypes::Real(12.0));

            // Removing mass edges of concerne triangle
            for (unsigned int j = 0; j < 3; ++j)
                EdgeMasses[te[j]] -= mass;

            // update total mass
            if (!this->isLumped())
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
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 2, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyVertexMassQuadCreation(const sofa::type::vector< Index >& quadAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Quad >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs)
{
    SOFA_UNUSED(elems);

    if (this->getMassTopologyType() == geometry::ElementType::QUAD)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses ( d_vertexMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);

        // update mass density vector
        const std::size_t nbMass = getMassDensity().size();
        const sofa::Size nbQ = this->l_topology->getNbQuads();
        if (nbMass < nbQ)
            addMassDensity(quadAdded, ancestors, coefs);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < quadAdded.size(); ++i)
        {
            // Get the quad to be added
            const core::topology::BaseMeshTopology::Quad& q = this->l_topology->getQuad(quadAdded[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            const auto& pos0 = GeometricalTypes::getCPos(positions[q[0]]);
            const auto& pos1 = GeometricalTypes::getCPos(positions[q[1]]);
            const auto& pos2 = GeometricalTypes::getCPos(positions[q[2]]);
            const auto& pos3 = GeometricalTypes::getCPos(positions[q[3]]);

            const auto quadArea = sofa::geometry::Quad::area(pos0, pos1, pos2, pos3);
            mass = (densityM[quadAdded[i]] * quadArea) / (typename DataTypes::Real(8.0));

            // Adding mass
            for (unsigned int j = 0; j < 4; ++j)
                VertexMasses[q[j]] += mass;

            // update total mass
            if (!this->isLumped())
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
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 2, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyEdgeMassQuadCreation(const sofa::type::vector< Index >& quadAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Quad >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs)
{
    SOFA_UNUSED(elems);

    if (this->getMassTopologyType() == geometry::ElementType::QUAD)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses ( d_edgeMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);
        
        // update mass density vector
        const std::size_t nbMass = getMassDensity().size();
        const sofa::Size nbQ = this->l_topology->getNbQuads();
        if (nbMass < nbQ)
            addMassDensity(quadAdded, ancestors, coefs);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < quadAdded.size(); ++i)
        {
            // Get the EdgesInQuad to be added
            const core::topology::BaseMeshTopology::EdgesInQuad& qe = this->l_topology->getEdgesInQuad(quadAdded[i]);
            const core::topology::BaseMeshTopology::Quad& q = this->l_topology->getQuad(quadAdded[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            const auto& pos0 = GeometricalTypes::getCPos(positions[q[0]]);
            const auto& pos1 = GeometricalTypes::getCPos(positions[q[1]]);
            const auto& pos2 = GeometricalTypes::getCPos(positions[q[2]]);
            const auto& pos3 = GeometricalTypes::getCPos(positions[q[3]]);

            const auto quadArea = sofa::geometry::Quad::area(pos0, pos1, pos2, pos3);
            mass = (densityM[quadAdded[i]] * quadArea) / (typename DataTypes::Real(16.0));

            // Adding mass edges of concerne quad
            for (unsigned int j = 0; j < 4; ++j)
                EdgeMasses[qe[j]] += mass;

            // update total mass
            if (!this->isLumped())
            {
                totalMass += 4.0 * mass * 2.0; // x 2 because mass is actually splitted over half-edges
            }
        }
    }
}

/// Destruction fonction for mass stored on vertices
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 2, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyVertexMassQuadDestruction(const sofa::type::vector< Index >& quadRemoved)
{
    if (this->getMassTopologyType() == geometry::ElementType::QUAD)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses ( d_vertexMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < quadRemoved.size(); ++i)
        {
            // Get the quad to be removed
            const core::topology::BaseMeshTopology::Quad& q = this->l_topology->getQuad(quadRemoved[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            const auto& pos0 = GeometricalTypes::getCPos(positions[q[0]]);
            const auto& pos1 = GeometricalTypes::getCPos(positions[q[1]]);
            const auto& pos2 = GeometricalTypes::getCPos(positions[q[2]]);
            const auto& pos3 = GeometricalTypes::getCPos(positions[q[3]]);

            const auto quadArea = sofa::geometry::Quad::area(pos0, pos1, pos2, pos3);
            mass = (densityM[quadRemoved[i]] * quadArea) / (typename DataTypes::Real(8.0));

            // Removing mass
            for (unsigned int j = 0; j < 4; ++j)
                VertexMasses[q[j]] -= mass;

            // update total mass
            if (!this->isLumped())
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
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 2, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyEdgeMassQuadDestruction(const sofa::type::vector< Index >& quadRemoved)
{
    if (this->getMassTopologyType() == geometry::ElementType::QUAD)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses ( d_edgeMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < quadRemoved.size(); ++i)
        {
            // Get the EdgesInQuad to be removed
            const core::topology::BaseMeshTopology::EdgesInQuad& qe = this->l_topology->getEdgesInQuad(quadRemoved[i]);
            const core::topology::BaseMeshTopology::Quad& q = this->l_topology->getQuad(quadRemoved[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            const auto& pos0 = GeometricalTypes::getCPos(positions[q[0]]);
            const auto& pos1 = GeometricalTypes::getCPos(positions[q[1]]);
            const auto& pos2 = GeometricalTypes::getCPos(positions[q[2]]);
            const auto& pos3 = GeometricalTypes::getCPos(positions[q[3]]);

            const auto quadArea = sofa::geometry::Quad::area(pos0, pos1, pos2, pos3);
            mass = (densityM[quadRemoved[i]] * quadArea) / (typename DataTypes::Real(16.0));

            // Removing mass edges of concerne quad
            for (unsigned int j = 0; j < 4; ++j)
                EdgeMasses[qe[j]] -= mass;

            // update total mass
            if (!this->isLumped())
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
template <class DataTypes, class GeometricalTypes>

template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 3, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyVertexMassTetrahedronCreation(const sofa::type::vector< Index >& tetrahedronAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Tetrahedron >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs)
{
    SOFA_UNUSED(elems);

    if (this->getMassTopologyType() == geometry::ElementType::TETRAHEDRON)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses ( d_vertexMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);

        // update mass density vector
        const std::size_t nbMass = getMassDensity().size();
        const sofa::Size nbT = this->l_topology->getNbTetrahedra();
        if (nbMass < nbT)
            addMassDensity(tetrahedronAdded, ancestors, coefs);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i<tetrahedronAdded.size(); ++i)
        {
            // Get the tetrahedron to be added
            const core::topology::BaseMeshTopology::Tetrahedron &t = this->l_topology->getTetrahedron(tetrahedronAdded[i]);

            // Compute rest mass of conserne tetrahedron = density * tetrahedron volume.
            const auto& rpos0 = GeometricalTypes::getCPos(positions[t[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(positions[t[1]]);
            const auto& rpos2 = GeometricalTypes::getCPos(positions[t[2]]);
            const auto& rpos3 = GeometricalTypes::getCPos(positions[t[3]]);

            const auto tetraVolume = sofa::geometry::Tetrahedron::volume(rpos0, rpos1, rpos2, rpos3);
            mass = (densityM[tetrahedronAdded[i]] * tetraVolume) / (typename DataTypes::Real(10.0));

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
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 3, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyEdgeMassTetrahedronCreation(const sofa::type::vector< Index >& tetrahedronAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Tetrahedron >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs)
{
    SOFA_UNUSED(elems);

    if (this->getMassTopologyType() == geometry::ElementType::TETRAHEDRON)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses ( d_edgeMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);

        // update mass density vector
        const std::size_t nbMass = getMassDensity().size();
        const sofa::Size nbT = this->l_topology->getNbTetrahedra();
        if (nbMass < nbT)
            addMassDensity(tetrahedronAdded, ancestors, coefs);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < tetrahedronAdded.size(); ++i)
        {
            // Get the edgesInTetrahedron to be added
            const core::topology::BaseMeshTopology::EdgesInTetrahedron& te = this->l_topology->getEdgesInTetrahedron(tetrahedronAdded[i]);
            const core::topology::BaseMeshTopology::Tetrahedron& t = this->l_topology->getTetrahedron(tetrahedronAdded[i]);

            // Compute rest mass of conserne tetrahedron = density * tetrahedron volume.
            const auto& rpos0 = GeometricalTypes::getCPos(positions[t[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(positions[t[1]]);
            const auto& rpos2 = GeometricalTypes::getCPos(positions[t[2]]);
            const auto& rpos3 = GeometricalTypes::getCPos(positions[t[3]]);

            const auto tetraVolume = sofa::geometry::Tetrahedron::volume(rpos0, rpos1, rpos2, rpos3);
            mass = (densityM[tetrahedronAdded[i]] * tetraVolume) / (typename DataTypes::Real(20.0));

            // Adding mass edges of concerne triangle
            for (unsigned int j = 0; j < 6; ++j)
                EdgeMasses[te[j]] += mass;

            // update total mass
            if (!this->isLumped())
            {
                totalMass += 6.0 * mass * 2.0; // x 2 because mass is actually splitted over half-edges
            }
        }
    }
}

/// Destruction fonction for mass stored on vertices
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 3, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyVertexMassTetrahedronDestruction(const sofa::type::vector< Index >& tetrahedronRemoved)
{
    if (this->getMassTopologyType() == geometry::ElementType::TETRAHEDRON)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses ( d_vertexMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < tetrahedronRemoved.size(); ++i)
        {
            // Get the tetrahedron to be removed
            const core::topology::BaseMeshTopology::Tetrahedron& t = this->l_topology->getTetrahedron(tetrahedronRemoved[i]);

            // Compute rest mass of conserne tetrahedron = density * tetrahedron volume.
            const auto& rpos0 = GeometricalTypes::getCPos(positions[t[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(positions[t[1]]);
            const auto& rpos2 = GeometricalTypes::getCPos(positions[t[2]]);
            const auto& rpos3 = GeometricalTypes::getCPos(positions[t[3]]);

            const auto tetraVolume = sofa::geometry::Tetrahedron::volume(rpos0, rpos1, rpos2, rpos3);
            mass = (densityM[tetrahedronRemoved[i]] * tetraVolume) / (typename DataTypes::Real(10.0));

            // Removing mass
            for (unsigned int j = 0; j < 4; ++j)
                VertexMasses[t[j]] -= mass;

            // update total mass
            if (!this->isLumped())
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
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 3, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyEdgeMassTetrahedronDestruction(const sofa::type::vector< Index >& tetrahedronRemoved)
{
    if (this->getMassTopologyType() == geometry::ElementType::TETRAHEDRON)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses ( d_edgeMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < tetrahedronRemoved.size(); ++i)
        {
            // Get the edgesInTetrahedron to be removed
            const core::topology::BaseMeshTopology::EdgesInTetrahedron& te = this->l_topology->getEdgesInTetrahedron(tetrahedronRemoved[i]);
            const core::topology::BaseMeshTopology::Tetrahedron& t = this->l_topology->getTetrahedron(tetrahedronRemoved[i]);

            // Compute rest mass of conserne tetrahedron = density * tetrahedron volume.
            const auto& rpos0 = GeometricalTypes::getCPos(positions[t[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(positions[t[1]]);
            const auto& rpos2 = GeometricalTypes::getCPos(positions[t[2]]);
            const auto& rpos3 = GeometricalTypes::getCPos(positions[t[3]]);

            const auto tetraVolume = sofa::geometry::Tetrahedron::volume(rpos0, rpos1, rpos2, rpos3);
            mass = (densityM[tetrahedronRemoved[i]] * tetraVolume) / (typename DataTypes::Real(20.0));

            // Removing mass edges of concerne triangle
            for (unsigned int j = 0; j < 6; ++j)
                EdgeMasses[te[j]] -= mass;

            // update total mass
            if (!this->isLumped())
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
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 3, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyVertexMassHexahedronCreation(const sofa::type::vector< Index >& hexahedronAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Hexahedron >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs)
{
    SOFA_UNUSED(elems);

    if (this->getMassTopologyType() == geometry::ElementType::HEXAHEDRON)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses ( d_vertexMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);

        // update mass density vector
        const std::size_t nbMass = getMassDensity().size();
        const sofa::Size nbT = this->l_topology->getNbHexahedra();
        if (nbMass < nbT)
            addMassDensity(hexahedronAdded, ancestors, coefs);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < hexahedronAdded.size(); ++i)
        {
            // Get the hexahedron to be added
            const core::topology::BaseMeshTopology::Hexahedron& h = this->l_topology->getHexahedron(hexahedronAdded[i]);

            /// compute its mass based on the mass density and the hexahedron volume
            const auto& rpos0 = GeometricalTypes::getCPos(positions[h[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(positions[h[1]]);
            const auto& rpos2 = GeometricalTypes::getCPos(positions[h[2]]);
            const auto& rpos3 = GeometricalTypes::getCPos(positions[h[3]]);
            const auto& rpos4 = GeometricalTypes::getCPos(positions[h[4]]);
            const auto& rpos5 = GeometricalTypes::getCPos(positions[h[5]]);
            const auto& rpos6 = GeometricalTypes::getCPos(positions[h[6]]);
            const auto& rpos7 = GeometricalTypes::getCPos(positions[h[7]]);

            const auto hexaVolume = sofa::geometry::Hexahedron::volume(rpos0, rpos1, rpos2, rpos3, rpos4, rpos5, rpos6, rpos7);
            mass = (densityM[hexahedronAdded[i]] * hexaVolume) / (typename DataTypes::Real(20.0));

            // Adding mass
            for (unsigned int j = 0; j < 8; ++j)
                VertexMasses[h[j]] += mass;

            // update total mass
            if (!this->isLumped())
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
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 3, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyEdgeMassHexahedronCreation(const sofa::type::vector< Index >& hexahedronAdded,
        const sofa::type::vector< core::topology::BaseMeshTopology::Hexahedron >& elems,
        const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs)
{
    SOFA_UNUSED(elems);

    if (this->getMassTopologyType() == geometry::ElementType::HEXAHEDRON)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses ( d_edgeMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);

        // update mass density vector
        const std::size_t nbMass = getMassDensity().size();
        const sofa::Size nbT = this->l_topology->getNbHexahedra();
        if (nbMass < nbT)
            addMassDensity(hexahedronAdded, ancestors, coefs);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < hexahedronAdded.size(); ++i)
        {
            // Get the EdgesInHexahedron to be added
            const core::topology::BaseMeshTopology::EdgesInHexahedron& he = this->l_topology->getEdgesInHexahedron(hexahedronAdded[i]);
            const core::topology::BaseMeshTopology::Hexahedron& h = this->l_topology->getHexahedron(hexahedronAdded[i]);

            /// compute its mass based on the mass density and the hexahedron volume
            const auto& rpos0 = GeometricalTypes::getCPos(positions[h[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(positions[h[1]]);
            const auto& rpos2 = GeometricalTypes::getCPos(positions[h[2]]);
            const auto& rpos3 = GeometricalTypes::getCPos(positions[h[3]]);
            const auto& rpos4 = GeometricalTypes::getCPos(positions[h[4]]);
            const auto& rpos5 = GeometricalTypes::getCPos(positions[h[5]]);
            const auto& rpos6 = GeometricalTypes::getCPos(positions[h[6]]);
            const auto& rpos7 = GeometricalTypes::getCPos(positions[h[7]]);

            const auto hexaVolume = sofa::geometry::Hexahedron::volume(rpos0, rpos1, rpos2, rpos3, rpos4, rpos5, rpos6, rpos7);
            mass = (densityM[hexahedronAdded[i]] * hexaVolume) / (typename DataTypes::Real(40.0));

            // Adding mass edges of concerne triangle
            for (unsigned int j = 0; j < 12; ++j)
                EdgeMasses[he[j]] += mass;

            // update total mass
            if (!this->isLumped())
            {
                totalMass += 12.0 * mass * 2.0; // x 2 because mass is actually splitted over half-edges
            }
        }
    }
}

/// Destruction fonction for mass stored on vertices
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 3, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyVertexMassHexahedronDestruction(const sofa::type::vector< Index >& hexahedronRemoved)
{
    if (this->getMassTopologyType() == geometry::ElementType::HEXAHEDRON)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > VertexMasses ( d_vertexMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < hexahedronRemoved.size(); ++i)
        {
            // Get the hexahedron to be removed
            const core::topology::BaseMeshTopology::Hexahedron& h = this->l_topology->getHexahedron(hexahedronRemoved[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            const auto& rpos0 = GeometricalTypes::getCPos(positions[h[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(positions[h[1]]);
            const auto& rpos2 = GeometricalTypes::getCPos(positions[h[2]]);
            const auto& rpos3 = GeometricalTypes::getCPos(positions[h[3]]);
            const auto& rpos4 = GeometricalTypes::getCPos(positions[h[4]]);
            const auto& rpos5 = GeometricalTypes::getCPos(positions[h[5]]);
            const auto& rpos6 = GeometricalTypes::getCPos(positions[h[6]]);
            const auto& rpos7 = GeometricalTypes::getCPos(positions[h[7]]);

            const auto hexaVolume = sofa::geometry::Hexahedron::volume(rpos0, rpos1, rpos2, rpos3, rpos4, rpos5, rpos6, rpos7);
            mass = (densityM[hexahedronRemoved[i]] * hexaVolume) / (typename DataTypes::Real(20.0));

            // Removing mass
            for (unsigned int j = 0; j < 8; ++j)
                VertexMasses[h[j]] -= mass;

            // update total mass
            if (!this->isLumped())
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
template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 3, int > >
void MeshMatrixMass<DataTypes, GeometricalTypes>::applyEdgeMassHexahedronDestruction(const sofa::type::vector< Index >& hexahedronRemoved)
{
    if (this->getMassTopologyType() == geometry::ElementType::HEXAHEDRON)
    {
        core::ConstVecCoordId posid = this->d_computeMassOnRest.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
        const auto& positions = l_geometryState->read(posid)->getValue();

        helper::WriteAccessor< Data< type::vector<MassType> > > EdgeMasses ( d_edgeMass );
        auto totalMass = sofa::helper::getWriteOnlyAccessor(d_totalMass);

        // Initialisation
        const auto& densityM = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);

        for (unsigned int i = 0; i < hexahedronRemoved.size(); ++i)
        {
            // Get the EdgesInHexahedron to be removed
            const core::topology::BaseMeshTopology::EdgesInHexahedron& he = this->l_topology->getEdgesInHexahedron(hexahedronRemoved[i]);
            const core::topology::BaseMeshTopology::Hexahedron& h = this->l_topology->getHexahedron(hexahedronRemoved[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            const auto& rpos0 = GeometricalTypes::getCPos(positions[h[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(positions[h[1]]);
            const auto& rpos2 = GeometricalTypes::getCPos(positions[h[2]]);
            const auto& rpos3 = GeometricalTypes::getCPos(positions[h[3]]);
            const auto& rpos4 = GeometricalTypes::getCPos(positions[h[4]]);
            const auto& rpos5 = GeometricalTypes::getCPos(positions[h[5]]);
            const auto& rpos6 = GeometricalTypes::getCPos(positions[h[6]]);
            const auto& rpos7 = GeometricalTypes::getCPos(positions[h[7]]);

            const auto hexaVolume = sofa::geometry::Hexahedron::volume(rpos0, rpos1, rpos2, rpos3, rpos4, rpos5, rpos6, rpos7);
            mass = (densityM[hexahedronRemoved[i]] * hexaVolume) / (typename DataTypes::Real(40.0));

            // Removing mass edges of concerne triangle
            for (unsigned int j = 0; j < 12; ++j)
                EdgeMasses[he[j]] -= mass;

            // update total mass
            if (!this->isLumped())
            {
                totalMass -= 12.0 * mass * 2.0; // x 2 because mass is actually splitted over half-edges
            }
        }
    }
}


// }

using sofa::geometry::ElementType;

template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::init()
{
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);

    m_massLumpingCoeff = 0.0;

    Inherited::init();

    const geometry::ElementType topoType = checkTopology();
    if(topoType == geometry::ElementType::POINT)
    {
        return;
    }
    setMassTopologyType(topoType);
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

    //everything has been initialized so mark the component in a valid state
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);

    // Adding callback warning in case d_lumping data is modified after init()
    sofa::core::objectmodel::Base::addUpdateCallback("updateLumping", {&d_lumping}, [this](const core::DataTracker& )
    {
        msg_error() << "Data \'lumping\' should not be modified after the component initialization";
        return sofa::core::objectmodel::ComponentState::Invalid;
    }, {});
}


template <class DataTypes, class GeometricalTypes>
sofa::geometry::ElementType MeshMatrixMass<DataTypes, GeometricalTypes>::checkTopology()
{
    if (l_topology.empty())
    {
        msg_info() << "Link \"topology\" to the Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    if (l_topology.get() == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return sofa::geometry::ElementType::POINT;
    }
    else
    {
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";
    }

    if (l_geometryState.empty())
    {
        msg_info() << "Link \"geometryState\" to the MechanicalObject associated with the geometry should be set to ensure right behavior. First container found from the topology context will be used.";
        sofa::core::behavior::BaseMechanicalState::SPtr baseState;
        l_topology->getContext()->get(baseState);

        if (baseState == nullptr)
        {
            msg_error() << "No compatible state associated with the topology has been found.";
            return sofa::geometry::ElementType::POINT;
        }
        else
        {
            typename sofa::core::behavior::MechanicalState<GeometricalTypes>::SPtr geometryState = boost::dynamic_pointer_cast<sofa::core::behavior::MechanicalState<GeometricalTypes>>(baseState);
            if (geometryState == nullptr)
            {
                msg_error() << "A state associated with the topology has been found but is incompatible with the definition of the mass (templates mismatch).";
                return sofa::geometry::ElementType::POINT;
            }
            else
            {
                l_geometryState.set(geometryState);
                msg_info() << "Topology is associated with the state: '" << l_geometryState->getPathName() << "'";
            }
        }
    }
        
    if (l_topology->getNbHexahedra() > 0)
    {
        msg_info() << "Hexahedral topology found.";
        return geometry::ElementType::HEXAHEDRON;
    }
    else if (l_topology->getNbTetrahedra() > 0)
    {
        msg_info() << "Tetrahedral topology found.";
        return geometry::ElementType::TETRAHEDRON;
    }
    else if (l_topology->getNbQuads() > 0)
    {
        msg_info() << "Quad topology found.";
        return geometry::ElementType::QUAD;
    }
    else if (l_topology->getNbTriangles() > 0)
    {
        msg_info() << "Triangular topology found.";
        return geometry::ElementType::TRIANGLE;
    }
    else if (l_topology->getNbEdges() > 0)
    {
        msg_info() << "Edge topology found.";
        return geometry::ElementType::EDGE;
    }
    else
    {
        msg_error() << "Topology empty.";
        return geometry::ElementType::POINT;
    }
}


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::initTopologyHandlers(sofa::geometry::ElementType topologyType)
{
    // add the functions to handle topology changes for Vertex informations
    d_vertexMass.createTopologyHandler(l_topology);
    d_vertexMass.setCreationCallback([this](Index pointIndex, MassType& m,
        const core::topology::BaseMeshTopology::Point& point,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs)
    {
        applyVertexMassCreation(pointIndex, m, point, ancestors, coefs);
    });
    d_vertexMass.setDestructionCallback([this](Index pointIndex, MassType& m)
    {
        applyVertexMassDestruction(pointIndex, m);
    });

    // add the functions to handle topology changes for Edge informations
    d_edgeMass.createTopologyHandler(l_topology);
    d_edgeMass.setCreationCallback([this](Index edgeIndex, MassType& EdgeMass,
        const core::topology::BaseMeshTopology::Edge& edge,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs)
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


    if constexpr (GeometricalTypes::spatial_dimensions >= 3)
    {
        if (topologyType == geometry::ElementType::HEXAHEDRON)
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

            if (!this->isLumped())
            {
                d_edgeMass.linkToHexahedronDataArray();
                d_edgeMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::HEXAHEDRAADDED, [this](const core::topology::TopologyChange* eventTopo) {
                    const core::topology::HexahedraAdded* hAdd = static_cast<const core::topology::HexahedraAdded*>(eventTopo);
                    applyEdgeMassHexahedronCreation(hAdd->getIndexArray(), hAdd->getElementArray(), hAdd->ancestorsList, hAdd->coefs);
                    });
                d_edgeMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::HEXAHEDRAREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
                    const core::topology::HexahedraRemoved* hRemove = static_cast<const core::topology::HexahedraRemoved*>(eventTopo);
                    applyEdgeMassHexahedronDestruction(hRemove->getArray());
                    });
            }

            hasQuads = true; // hexahedron imply quads
        }
        else if (topologyType == geometry::ElementType::TETRAHEDRON)
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

            if (!this->isLumped())
            {
                d_edgeMass.linkToTetrahedronDataArray();
                d_edgeMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TETRAHEDRAADDED, [this](const core::topology::TopologyChange* eventTopo) {
                    const core::topology::TetrahedraAdded* tAdd = static_cast<const core::topology::TetrahedraAdded*>(eventTopo);
                    applyEdgeMassTetrahedronCreation(tAdd->getIndexArray(), tAdd->getElementArray(), tAdd->ancestorsList, tAdd->coefs);
                    });
                d_edgeMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TETRAHEDRAREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
                    const core::topology::TetrahedraRemoved* tRemove = static_cast<const core::topology::TetrahedraRemoved*>(eventTopo);
                    applyEdgeMassTetrahedronDestruction(tRemove->getArray());
                    });
            }

            hasTriangles = true; // Tetrahedron imply triangles
        }
    }

    if constexpr (GeometricalTypes::spatial_dimensions >= 2)
    {
        if (topologyType == geometry::ElementType::QUAD || hasQuads)
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

            if (!this->isLumped())
            {
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
        }

        if (topologyType == geometry::ElementType::TRIANGLE || hasTriangles)
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

            if (!this->isLumped())
            {
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
    }
}

template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::massInitialization()
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


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::printMass()
{
    if (this->f_printLog.getValue() == false)
        return;

    //Info post-init
    const auto &vertexM = d_vertexMass.getValue();
    const auto &mDensity = d_massDensity.getValue();

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


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::computeMass()
{
    // resize array
    clear();

    // prepare to store info in the vertex array

    auto my_vertexMassInfo = sofa::helper::getWriteOnlyAccessor(d_vertexMass);
    auto my_edgeMassInfo = sofa::helper::getWriteOnlyAccessor(d_edgeMass);

    unsigned int ndof = this->mstate->getSize();
    unsigned int nbEdges=l_topology->getNbEdges();
    const type::vector<core::topology::BaseMeshTopology::Edge>& edges = l_topology->getEdges();

    my_vertexMassInfo.resize(ndof);
    my_edgeMassInfo.resize(nbEdges);

    const type::vector< Index > emptyAncestor;
    const type::vector< SReal > emptyCoefficient;
    const type::vector< type::vector< Index > > emptyAncestors;
    const type::vector< type::vector< SReal > > emptyCoefficients;

    // set vertex tensor to 0
    for (Index i = 0; i<ndof; ++i)
        applyVertexMassCreation(i, my_vertexMassInfo[i], i, emptyAncestor, emptyCoefficient);

    // set edge tensor to 0
    for (Index i = 0; i<nbEdges; ++i)
        applyEdgeMassCreation(i, my_edgeMassInfo[i], edges[i], emptyAncestor, emptyCoefficient);

    if constexpr (GeometricalTypes::spatial_dimensions >= 2)
    {
        if (getMassTopologyType() == geometry::ElementType::QUAD)
        {
            // create vector tensor by calling the quad creation function on the entire mesh
            sofa::type::vector<Index> quadsAdded;

            const size_t n = l_topology->getNbQuads();
            for (Index i = 0; i < n; ++i)
                quadsAdded.push_back(i);

            m_massLumpingCoeff = 2.0;
            if (!isLumped())
            {
                applyEdgeMassQuadCreation(quadsAdded, l_topology->getQuads(), emptyAncestors, emptyCoefficients);
            }

            applyVertexMassQuadCreation(quadsAdded, l_topology->getQuads(), emptyAncestors, emptyCoefficients);
        }

        if (getMassTopologyType() == geometry::ElementType::TRIANGLE)
        {
            // create vector tensor by calling the triangle creation function on the entire mesh
            sofa::type::vector<Index> trianglesAdded;

            const size_t n = l_topology->getNbTriangles();
            for (Index i = 0; i < n; ++i)
                trianglesAdded.push_back(i);

            m_massLumpingCoeff = 2.0;
            if (!isLumped())
            {
                applyEdgeMassTriangleCreation(trianglesAdded, l_topology->getTriangles(), emptyAncestors, emptyCoefficients);
            }

            applyVertexMassTriangleCreation(trianglesAdded, l_topology->getTriangles(), emptyAncestors, emptyCoefficients);
        }
    }

    if constexpr (GeometricalTypes::spatial_dimensions >= 3)
    {
        if (getMassTopologyType() == geometry::ElementType::HEXAHEDRON)
        {
            // create vector tensor by calling the hexahedron creation function on the entire mesh
            sofa::type::vector<Index> hexahedraAdded;
            const size_t n = l_topology->getNbHexahedra();
            for (Index i = 0; i < n; ++i)
                hexahedraAdded.push_back(i);

            m_massLumpingCoeff = 2.5;
            if (!isLumped())
            {
                applyEdgeMassHexahedronCreation(hexahedraAdded, l_topology->getHexahedra(), emptyAncestors, emptyCoefficients);
            }

            applyVertexMassHexahedronCreation(hexahedraAdded, l_topology->getHexahedra(), emptyAncestors, emptyCoefficients);
        }

        if (getMassTopologyType() == geometry::ElementType::TETRAHEDRON)
        {
            // create vector tensor by calling the tetrahedron creation function on the entire mesh
            sofa::type::vector<Index> tetrahedraAdded;

            const size_t n = l_topology->getNbTetrahedra();
            for (Index i = 0; i < n; ++i)
                tetrahedraAdded.push_back(i);

            m_massLumpingCoeff = 2.5;
            if (!isLumped())
            {
                applyEdgeMassTetrahedronCreation(tetrahedraAdded, l_topology->getTetrahedra(), emptyAncestors, emptyCoefficients);
            }

            applyVertexMassTetrahedronCreation(tetrahedraAdded, l_topology->getTetrahedra(), emptyAncestors, emptyCoefficients);
        }
    }
}


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::reinit()
{
    // Now update is handled through the doUpdateInternal mechanism
    // called at each begin of step through the UpdateInternalDataVisitor
}


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::doUpdateInternal()
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


template <class DataTypes, class GeometricalTypes>
bool MeshMatrixMass<DataTypes, GeometricalTypes>::checkTotalMass()
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


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::checkTotalMassInit()
{
    //Check for negative or null value, if wrongly set use the default value totalMass = 1.0
    if(!checkTotalMass())
    {
        msg_warning(this) << "Switching back to default values: totalMass = 1.0\n";
        d_totalMass.setValue(1.0) ;
    }
}


template <class DataTypes, class GeometricalTypes>
bool MeshMatrixMass<DataTypes, GeometricalTypes>::checkVertexMass()
{
    const auto &vertexMass = d_vertexMass.getValue();
    //Check size of the vector
    if (vertexMass.size() != size_t(l_topology->getNbPoints()))
    {
        msg_warning() << "Inconsistent size of vertexMass vector ("<< vertexMass.size() <<") compared to the DOFs size ("<< l_topology->getNbPoints() <<").";
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


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::initFromVertexMass()
{
    msg_info() << "vertexMass information is used";

    const auto vertexMass = d_vertexMass.getValue();
    Real totalMassSave = 0.0;
    for(size_t i=0; i<vertexMass.size(); i++)
    {
        totalMassSave += vertexMass[i];
    }
    //Compute the volume
    setMassDensity(1.0);

    computeMass();

    auto vertexMassInfo = sofa::helper::getWriteOnlyAccessor(d_vertexMass);
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


template <class DataTypes, class GeometricalTypes>
bool MeshMatrixMass<DataTypes, GeometricalTypes>::checkEdgeMass()
{
    const auto& edgeMass = d_edgeMass.getValue();
    //Check size of the vector
    if (edgeMass.size() != l_topology->getNbEdges())
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


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::initFromVertexAndEdgeMass()
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


template <class DataTypes, class GeometricalTypes>
bool MeshMatrixMass<DataTypes, GeometricalTypes>::checkMassDensity()
{
    const auto &massDensity = d_massDensity.getValue();
    auto density = massDensity[0];
    size_t sizeElements = 0;

    //Check size of the vector
    //Size = 1, homogeneous density
    //Otherwise, heterogeneous density
    if (l_topology->getNbHexahedra()>0)
    {
        sizeElements = l_topology->getNbHexahedra();

        if ( massDensity.size() != l_topology->getNbHexahedra() && massDensity.size() != 1)
        {
            msg_warning() << "Inconsistent size of massDensity = " << massDensity.size() << ", should be either 1 or " << l_topology->getNbHexahedra();
            return false;
        }
    }
    else if (l_topology->getNbTetrahedra()>0)
    {
        sizeElements = l_topology->getNbTetrahedra();

        if ( massDensity.size() != l_topology->getNbTetrahedra() && massDensity.size() != 1)
        {
            msg_warning() << "Inconsistent size of massDensity = " << massDensity.size() << ", should be either 1 or " << l_topology->getNbTetrahedra();
            return false;
        }
    }
    else if (l_topology->getNbQuads()>0)
    {
        sizeElements = l_topology->getNbQuads();

        if ( massDensity.size() != l_topology->getNbQuads() && massDensity.size() != 1)
        {
            msg_warning() << "Inconsistent size of massDensity = " << massDensity.size() << ", should be either 1 or " << l_topology->getNbQuads();
            return false;
        }
    }
    else if (l_topology->getNbTriangles()>0)
    {
        sizeElements = l_topology->getNbTriangles();

        if ( massDensity.size() != l_topology->getNbTriangles() && massDensity.size() != 1)
        {
            msg_warning() << "Inconsistent size of massDensity = " << massDensity.size() << ", should be either 1 or " << l_topology->getNbTriangles();
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
            auto massDensityAccess = sofa::helper::getWriteOnlyAccessor(d_massDensity);
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


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::initFromMassDensity()
{
    msg_info() << "massDensity information is used";

    // compute vertexMass and edgeMass vector with input density
    computeMass();

    const auto &vertexMassInfo = d_vertexMass.getValue();
    
    // Sum the mass per vertices and apply massLumping coef
    Real sumMass = std::accumulate(vertexMassInfo.begin(), vertexMassInfo.end(), Real(0));

    if (!isLumped())
    {
        // Add mass per edges if not lumped, *2 as it is added to both edge vertices
        auto edgeMass = sofa::helper::getWriteOnlyAccessor(d_edgeMass);
        sumMass += std::accumulate(edgeMass.begin(), edgeMass.end(), Real(0)) * 2.0;
    }
    else
    {
        sumMass *= m_massLumpingCoeff;
    }

    d_totalMass.setValue(sumMass);
}


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::initFromTotalMass()
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
    auto vertexMass = sofa::helper::getWriteOnlyAccessor(d_vertexMass);
    for (auto& vm : vertexMass)
    {
        vm *= md;
    }

    // Same for edgeMass
    auto edgeMass = sofa::helper::getWriteOnlyAccessor(d_edgeMass);
    for (auto& em : edgeMass)
    {
        em *= md;
    }
}


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::setVertexMass(sofa::type::vector< MassType > vertexMass)
{
    const auto& currentVertexMass = d_vertexMass.getValue();
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


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::setMassDensity(sofa::type::vector< MassType > massDensity)
{
    const auto& currentMassDensity = d_massDensity.getValue();
    d_massDensity.setValue(massDensity);

    if(!checkMassDensity())
    {
        msg_warning() << "Given values to setMassDensity() are not correct.\n"
                      << "Previous values are used.";
        d_massDensity.setValue(currentMassDensity);
    }
}


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::setMassDensity(MassType massDensityValue)
{
    const auto currentMassDensity = d_massDensity.getValue();
    auto massDensity = sofa::helper::getWriteOnlyAccessor(d_massDensity);

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

template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::addMassDensity(const sofa::type::vector< Index >& indices, 
    const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
    const sofa::type::vector< sofa::type::vector< SReal > >& coefs)
{
    auto massDensity = sofa::helper::getWriteOnlyAccessor(d_massDensity);

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

template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::setTotalMass(MassType totalMass)
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


template <class DataTypes, class GeometricalTypes>
const sofa::type::vector< typename MeshMatrixMass<DataTypes, GeometricalTypes>::MassType > &  MeshMatrixMass<DataTypes, GeometricalTypes>::getVertexMass()
{
    return d_vertexMass.getValue();
}


template <class DataTypes, class GeometricalTypes>
const sofa::type::vector< typename MeshMatrixMass<DataTypes, GeometricalTypes>::MassType > &  MeshMatrixMass<DataTypes, GeometricalTypes>::getMassDensity()
{
    return d_massDensity.getValue();
}


template <class DataTypes, class GeometricalTypes>
const typename MeshMatrixMass<DataTypes, GeometricalTypes>::Real &MeshMatrixMass<DataTypes, GeometricalTypes>::getTotalMass()
{
    return d_totalMass.getValue();
}


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::copyVertexMass(){}


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::clear()
{
    auto vertexMass = sofa::helper::getWriteOnlyAccessor(d_vertexMass);
    auto edgeMass = sofa::helper::getWriteOnlyAccessor(d_edgeMass);

    vertexMass.clear();
    edgeMass.clear();
}


// -- Mass interface
template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::addMDx(const core::MechanicalParams*, DataVecDeriv& vres, const DataVecDeriv& vdx, SReal factor)
{
    const auto &vertexMass= d_vertexMass.getValue();
    const auto &edgeMass= d_edgeMass.getValue();

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
        const size_t nbEdges=l_topology->getNbEdges();
        const auto& edges = l_topology->getEdges();

        for (unsigned int i=0; i<dx.size(); i++)
        {
            res[i] += dx[i] * vertexMass[i] * Real(factor);
            massTotal += vertexMass[i] * Real(factor);
        }

        Real tempMass=0.0;

        for (unsigned int j=0; j<nbEdges; ++j)
        {
            const auto& e = edges[j];

            tempMass = edgeMass[j] * Real(factor);

            res[e[0]] += dx[e[1]] * tempMass;
            res[e[1]] += dx[e[0]] * tempMass;

            massTotal += 2 * tempMass;
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


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f)
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
    const auto &vertexMass= d_vertexMass.getValue();

    for (unsigned int i=0; i<vertexMass.size(); i++)
    {
        _a[i] = _f[i] / ( vertexMass[i] * m_massLumpingCoeff);
    }
}


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::addForce(const core::MechanicalParams*, DataVecDeriv& vf, const DataVecCoord& , const DataVecDeriv& )
{

    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if(this->m_separateGravity.getValue())
        return ;

    helper::WriteAccessor< DataVecDeriv > f = vf;

    const auto &vertexMass= d_vertexMass.getValue();

    // gravity
    type::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);

    // add weight and inertia force
    for (unsigned int i=0; i<f.size(); ++i)
        f[i] += theGravity * vertexMass[i] * m_massLumpingCoeff;
}


template <class DataTypes, class GeometricalTypes>
SReal MeshMatrixMass<DataTypes, GeometricalTypes>::getKineticEnergy( const core::MechanicalParams*, const DataVecDeriv& vv ) const
{
    const auto &vertexMass= d_vertexMass.getValue();
    const auto &edgeMass= d_edgeMass.getValue();

    helper::ReadAccessor< DataVecDeriv > v = vv;

    const unsigned int nbEdges=l_topology->getNbEdges();
    unsigned int v0,v1;

    SReal e = 0;

    for (unsigned int i=0; i<v.size(); i++)
    {
        e += dot(v[i],v[i]) * vertexMass[i]; // v[i]*v[i]*masses[i] would be more efficient but less generic
    }

    for (unsigned int i = 0; i < nbEdges; ++i)
    {
        v0 = l_topology->getEdge(i)[0];
        v1 = l_topology->getEdge(i)[1];

        e += 2 * dot(v[v0], v[v1])*edgeMass[i];

    }

    return e/2;
}


template <class DataTypes, class GeometricalTypes>
SReal MeshMatrixMass<DataTypes, GeometricalTypes>::getPotentialEnergy( const core::MechanicalParams*, const DataVecCoord& vx) const
{
    const auto &vertexMass= d_vertexMass.getValue();

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
template <class DataTypes, class GeometricalTypes>
type::Vec6 MeshMatrixMass<DataTypes, GeometricalTypes>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& /*vx*/, const DataVecDeriv& /*vv*/  ) const
{
    return type::Vec6();
}



template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v)
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




template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    const auto &vertexMass= d_vertexMass.getValue();
    const auto &edgeMass= d_edgeMass.getValue();

    const size_t nbEdges=l_topology->getNbEdges();
    sofa::Index v0,v1;

    static constexpr auto N = Deriv::total_size;
    AddMToMatrixFunctor<Deriv,MassType, sofa::linearalgebra::BaseMatrix> calc;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    sofa::linearalgebra::BaseMatrix* mat = r.matrix;
    const Real mFactor = Real(sofa::core::mechanicalparams::mFactorIncludingRayleighDamping(mparams, this->rayleighMass.getValue()));

    if((mat->colSize()) != (linearalgebra::BaseMatrix::Index)(l_topology->getNbPoints()*N) || (mat->rowSize()) != (linearalgebra::BaseMatrix::Index)(l_topology->getNbPoints()*N))
    {
        msg_error() <<"Wrong size of the input Matrix: need resize in addMToMatrix function.";
        mat->resize(l_topology->getNbPoints()*N,l_topology->getNbPoints()*N);
    }

    SReal massTotal=0.0;

    if(isLumped())
    {
        unsigned int i {};
        for (const auto& v : vertexMass)
        {
            calc(r.matrix, v * m_massLumpingCoeff, r.offset + N * i, mFactor);
            massTotal += v * m_massLumpingCoeff;
            ++i;
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
        unsigned int i {};
        for (const auto& v : vertexMass)
        {
            calc(r.matrix, v, r.offset + N * i, mFactor);
            massTotal += v;
            ++i;
        }

        const auto& edges = l_topology->getEdges();
        for (size_t j = 0; j < nbEdges; ++j)
        {
            v0 = edges[j][0];
            v1 = edges[j][1];

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

template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::buildMassMatrix(sofa::core::behavior::MassMatrixAccumulator* matrices)
{
    const MassVector &vertexMass= d_vertexMass.getValue();
    const MassVector &edgeMass= d_edgeMass.getValue();

    static constexpr auto N = Deriv::total_size;
    AddMToMatrixFunctor<Deriv,MassType, sofa::core::behavior::MassMatrixAccumulator> calc;

    if (isLumped())
    {
        for (size_t index=0; index < vertexMass.size(); index++)
        {
            const auto vm = vertexMass[index] * m_massLumpingCoeff;
            calc(matrices, vm, N * index, 1.);
        }
    }
    else
    {
        for (size_t index=0; index < vertexMass.size(); index++)
        {
            const auto& vm = vertexMass[index];
            calc(matrices, vm, N * index, 1.);
        }

        const size_t nbEdges = l_topology->getNbEdges();
        for (size_t j = 0; j < nbEdges; ++j)
        {
            const auto e = l_topology->getEdge(j);
            const sofa::Index v0 = e[0];
            const sofa::Index v1 = e[1];

            const auto em = edgeMass[j];

            calc(matrices, em, N * v0, N * v1, 1.);
            calc(matrices, em, N * v1, N * v0, 1.);
        }
    }
}


template <class DataTypes, class GeometricalTypes>
SReal MeshMatrixMass<DataTypes, GeometricalTypes>::getElementMass(Index index) const
{
    const auto &vertexMass= d_vertexMass.getValue();
    const SReal mass = vertexMass[index] * m_massLumpingCoeff;

    return mass;
}


//TODO: special case for Rigid Mass
template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::getElementMass(Index index, linearalgebra::BaseMatrix *m) const
{
    static const linearalgebra::BaseMatrix::Index dimension = linearalgebra::BaseMatrix::Index(defaulttype::DataTypeInfo<Deriv>::size());
    if (m->rowSize() != dimension || m->colSize() != dimension) m->resize(dimension,dimension);

    m->clear();
    AddMToMatrixFunctor<Deriv,MassType, sofa::linearalgebra::BaseMatrix>()(m, d_vertexMass.getValue()[index] * m_massLumpingCoeff, 0, 1);
}


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::handleEvent(sofa::core::objectmodel::Event *event)
{
    SOFA_UNUSED(event);
}


template <class DataTypes, class GeometricalTypes>
void MeshMatrixMass<DataTypes, GeometricalTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    const auto &vertexMass= d_vertexMass.getValue();

    const auto& x = l_geometryState->read(core::ConstVecCoordId::position())->getValue();
    type::Vec3 gravityCenter;
    Real totalMass=0.0;

    std::vector<  type::Vec3 > points;
    constexpr sofa::Size dimensions = std::min(static_cast<sofa::Size>(GeometricalTypes::spatial_dimensions), static_cast<sofa::Size>(3));
    for (unsigned int i = 0; i < x.size(); i++)
    {
        const auto& position = GeometricalTypes::getCPos(x[i]);
        type::Vec3 p;
        for (sofa::Index j = 0; j < dimensions; j++)
        {
            p[j] = position[j];
        }

        points.push_back(p);
        gravityCenter += p * vertexMass[i] * m_massLumpingCoeff;
        totalMass += vertexMass[i] * m_massLumpingCoeff;
    }

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();
    sofa::type::RGBAColor color = sofa::type::RGBAColor::white();

    vparams->drawTool()->drawPoints(points, 2, color);

    std::vector<sofa::type::Vec3> vertices;

    if(d_showCenterOfGravity.getValue())
    {
        color = sofa::type::RGBAColor::magenta();
        gravityCenter /= totalMass;
        for(unsigned int i=0 ; i<Coord::spatial_dimensions ; i++)
        {

            type::Vec3 v{};
            v[i] = d_showAxisSize.getValue();
            vertices.push_back(gravityCenter - v); 
            vertices.push_back(gravityCenter + v);
        }
    }
    vparams->drawTool()->drawLines(vertices,5,color);

}


} // namespace sofa::component::mass
