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
#include <sofa/component/mass/DiagonalMass.h>
#include <sofa/core/behavior/Mass.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/helper/io/XspLoader.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/core/topology/TopologyData.inl>
#include <sofa/component/mass/AddMToMatrixFunctor.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/core/behavior/BaseLocalMassMatrix.h>
#include <numeric>

namespace sofa::component::mass
{

using sofa::core::objectmodel::ComponentState;
using namespace sofa::core::topology;

template <class DataTypes, class GeometricalTypes>
DiagonalMass<DataTypes, GeometricalTypes>::DiagonalMass()
    : d_vertexMass( initData(&d_vertexMass, "vertexMass", "Specify a vector giving the mass of each vertex. \n"
                                                          "If unspecified or wrongly set, the massDensity or totalMass information is used.") )
    , d_massDensity( initData(&d_massDensity, Real(1.0),"massDensity","Specify one single real and positive value for the mass density. \n"
                                                                      "If unspecified or wrongly set, the totalMass information is used.") )
    , d_totalMass(initData(&d_totalMass, Real(1.0), "totalMass", "Specify the total mass resulting from all particles. \n"
                                                                 "If unspecified or wrongly set, the default value is used: totalMass = 1.0"))
    , d_computeMassOnRest(initData(&d_computeMassOnRest, true, "computeMassOnRest", "If true, the mass of every element is computed based on the rest position rather than the position"))
    , d_showCenterOfGravity( initData(&d_showCenterOfGravity, false, "showGravityCenter", "Display the center of gravity of the system" ) )
    , d_showAxisSize( initData(&d_showAxisSize, 1.0f, "showAxisSizeFactor", "Factor length of the axis displayed (only used for rigids)" ) )
    , d_fileMass( initData(&d_fileMass,  "filename", "Xsp3.0 file to specify the mass parameters" ) )
    , l_topology(initLink("topology", "link to the topology container"))
    , l_geometryState(initLink("geometryState", "link to the MechanicalObject associated with the geometry"))
    , m_massTopologyType(sofa::geometry::ElementType::UNKNOWN)
{
    this->addAlias(&d_fileMass,"fileMass");
}

template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::applyPointCreation(PointID, MassType &m, const Point &, const sofa::type::vector<PointID> &, const sofa::type::vector<SReal> &)
{
    m=0;
}


template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::applyPointDestruction(Index id, MassType& VertexMass)
{
    SOFA_UNUSED(id);
    helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);
    totalMass -= VertexMass;
    this->cleanTracker();
}


template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::applyEdgeCreation(const sofa::type::vector< EdgeID >& edgeAdded,
        const sofa::type::vector< Edge >& /*elems*/,
        const sofa::type::vector< sofa::type::vector< EdgeID > >& /*ancestors*/,
        const sofa::type::vector< sofa::type::vector< SReal > >& /*coefs*/)
{
    if (this->getMassTopologyType() == sofa::geometry::ElementType::EDGE)
    {
        const auto& restPositions = l_geometryState->read(core::vec_id::read_access::restPosition)->getValue();

        helper::WriteAccessor<Data<MassVector> > masses(d_vertexMass);
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        typename DataTypes::Real md=getMassDensity();
        typename DataTypes::Real mass=typename DataTypes::Real(0);
        unsigned int i;

        for (i=0; i<edgeAdded.size(); ++i)
        {
            /// get the edge to be added
            const Edge &e=this->l_topology->getEdge(edgeAdded[i]);
            // compute its mass based on the mass density and the edge length
            const auto& rpos0 = GeometricalTypes::getCPos(restPositions[e[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(restPositions[e[1]]);

            const auto restEdgeLength = sofa::geometry::Edge::length(rpos0, rpos1);
            mass= (md* restEdgeLength) / (typename DataTypes::Real(2.0));

            // added mass on its two vertices
            masses[e[0]]+=mass;
            masses[e[1]]+=mass;

            totalMass += 2.0*mass;
        }

        this->cleanTracker();
        printMass();
    }
}

template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::applyEdgeDestruction(const sofa::type::vector<EdgeID> & edgeRemoved)
{
    if (this->getMassTopologyType() == sofa::geometry::ElementType::EDGE)
    {
        const auto& restPositions = l_geometryState->read(core::vec_id::read_access::restPosition)->getValue();

        helper::WriteAccessor<Data<MassVector> > masses(d_vertexMass);
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        typename DataTypes::Real md=getMassDensity();
        typename DataTypes::Real mass=typename DataTypes::Real(0);
        unsigned int i;

        for (i=0; i<edgeRemoved.size(); ++i)
        {
            /// get the edge to be added
            const Edge &e= this->l_topology->getEdge(edgeRemoved[i]);
            // compute its mass based on the mass density and the edge length
            const auto& rpos0 = GeometricalTypes::getCPos(restPositions[e[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(restPositions[e[1]]);

            const auto restEdgeLength = sofa::geometry::Edge::length(rpos0, rpos1);
            mass = (md * restEdgeLength) / (typename DataTypes::Real(2.0));

            // removed mass on its two vertices
            masses[e[0]]-=mass;
            masses[e[1]]-=mass;

            totalMass -= 2.0*mass;
        }

        this->cleanTracker();
        printMass();
    }
}


template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 2, int > >
void DiagonalMass<DataTypes, GeometricalTypes>::applyTriangleCreation(const sofa::type::vector< TriangleID >& triangleAdded,
        const sofa::type::vector< Triangle >& /*elems*/,
        const sofa::type::vector< sofa::type::vector< TriangleID > >& /*ancestors*/,
        const sofa::type::vector< sofa::type::vector< SReal > >& /*coefs*/)
{
    if (this->getMassTopologyType() == sofa::geometry::ElementType::TRIANGLE)
    {
        const auto& restPositions = l_geometryState->read(core::vec_id::read_access::restPosition)->getValue();

        helper::WriteAccessor<Data<MassVector> > masses(d_vertexMass);
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        typename DataTypes::Real md=getMassDensity();
        typename DataTypes::Real mass=typename DataTypes::Real(0);
        unsigned int i;

        for (i=0; i<triangleAdded.size(); ++i)
        {
            /// get the triangle to be added
            const Triangle &t=this->l_topology->getTriangle(triangleAdded[i]);
            // compute its mass based on the mass density and the triangle area
            const auto& rpos0 = GeometricalTypes::getCPos(restPositions[t[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(restPositions[t[1]]);
            const auto& rpos2 = GeometricalTypes::getCPos(restPositions[t[2]]);

            const auto restTriangleArea = sofa::geometry::Triangle::area(rpos0, rpos1, rpos2);
            mass = (md * restTriangleArea) / (typename DataTypes::Real(3.0));

            // added mass on its three vertices
            masses[t[0]]+=mass;
            masses[t[1]]+=mass;
            masses[t[2]]+=mass;

            totalMass+= 3.0*mass;
        }

        this->cleanTracker();
        printMass();
    }
}

template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 2, int > >
void DiagonalMass<DataTypes, GeometricalTypes>::applyTriangleDestruction(const sofa::type::vector<TriangleID > & triangleRemoved)
{

    if (this->getMassTopologyType() == sofa::geometry::ElementType::TRIANGLE)
    {
        const auto& restPositions = l_geometryState->read(core::vec_id::read_access::restPosition)->getValue();

        helper::WriteAccessor<Data<MassVector> > masses(d_vertexMass);
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        typename DataTypes::Real md = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);
        unsigned int i;

        for (i = 0; i < triangleRemoved.size(); ++i)
        {
            /// get the triangle to be added
            const Triangle& t = this->l_topology->getTriangle(triangleRemoved[i]);

            /// compute its mass based on the mass density and the triangle area
            const auto& rpos0 = GeometricalTypes::getCPos(restPositions[t[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(restPositions[t[1]]);
            const auto& rpos2 = GeometricalTypes::getCPos(restPositions[t[2]]);

            const auto restTriangleArea = sofa::geometry::Triangle::area(rpos0, rpos1, rpos2);
            mass = (md * restTriangleArea) / (typename DataTypes::Real(3.0));

            /// removed  mass on its three vertices
            masses[t[0]] -= mass;
            masses[t[1]] -= mass;
            masses[t[2]] -= mass;

            totalMass -= 3.0 * mass;
        }

        this->cleanTracker();
        printMass();
    }
}


template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 2, int > >
void DiagonalMass<DataTypes, GeometricalTypes>::applyQuadCreation(const sofa::type::vector< QuadID >& quadAdded,
    const sofa::type::vector< Quad >& /*elems*/,
    const sofa::type::vector< sofa::type::vector< QuadID > >& /*ancestors*/,
    const sofa::type::vector< sofa::type::vector< SReal > >& /*coefs*/)
{
    if (this->getMassTopologyType() == sofa::geometry::ElementType::QUAD)
    {
        const auto& restPositions = l_geometryState->read(core::vec_id::read_access::restPosition)->getValue();

        helper::WriteAccessor<Data<MassVector> > masses(d_vertexMass);
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        typename DataTypes::Real md = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);
        unsigned int i;

        for (i = 0; i < quadAdded.size(); ++i)
        {
            /// get the quad to be added
            const Quad& q = this->l_topology->getQuad(quadAdded[i]);

            // compute its mass based on the mass density and the quad area
            const auto& pos0 = GeometricalTypes::getCPos(restPositions[q[0]]);
            const auto& pos1 = GeometricalTypes::getCPos(restPositions[q[1]]);
            const auto& pos2 = GeometricalTypes::getCPos(restPositions[q[2]]);
            const auto& pos3 = GeometricalTypes::getCPos(restPositions[q[3]]);

            const auto quadArea = sofa::geometry::Quad::area(pos0, pos1, pos2, pos3);
            mass = (md * quadArea) / (typename DataTypes::Real(4.0));

            // added mass on its four vertices
            masses[q[0]] += mass;
            masses[q[1]] += mass;
            masses[q[2]] += mass;
            masses[q[3]] += mass;

            totalMass += 4.0 * mass;
        }

        this->cleanTracker();
        printMass();
    }
}

template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 2, int > >
void DiagonalMass<DataTypes, GeometricalTypes>::applyQuadDestruction(const sofa::type::vector<QuadID >& quadRemoved)
{
    if (this->getMassTopologyType() == sofa::geometry::ElementType::QUAD)
    {
        const auto& restPositions = l_geometryState->read(core::vec_id::read_access::restPosition)->getValue();

        helper::WriteAccessor<Data<MassVector> > masses(d_vertexMass);
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        typename DataTypes::Real md = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);
        unsigned int i;

        for (i = 0; i < quadRemoved.size(); ++i)
        {
            /// get the quad to be added
            const Quad& q = this->l_topology->getQuad(quadRemoved[i]);

            /// compute its mass based on the mass density and the quad area
            const auto& pos0 = GeometricalTypes::getCPos(restPositions[q[0]]);
            const auto& pos1 = GeometricalTypes::getCPos(restPositions[q[1]]);
            const auto& pos2 = GeometricalTypes::getCPos(restPositions[q[2]]);
            const auto& pos3 = GeometricalTypes::getCPos(restPositions[q[3]]);

            const auto quadArea = sofa::geometry::Quad::area(pos0, pos1, pos2, pos3);
            mass = (md * quadArea) / (typename DataTypes::Real(4.0));

            /// removed  mass on its four vertices
            masses[q[0]] -= mass;
            masses[q[1]] -= mass;
            masses[q[2]] -= mass;
            masses[q[3]] -= mass;

            totalMass -= 4.0 * mass;
        }

        this->cleanTracker();
        printMass();
    }
}


template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 3, int > >
void DiagonalMass<DataTypes, GeometricalTypes>::applyTetrahedronCreation(const sofa::type::vector< TetrahedronID >& tetrahedronAdded,
        const sofa::type::vector< Tetrahedron >& /*elems*/,
        const sofa::type::vector< sofa::type::vector< TetrahedronID > >& /*ancestors*/,
        const sofa::type::vector< sofa::type::vector< SReal > >& /*coefs*/)
{
    if (this->getMassTopologyType() == sofa::geometry::ElementType::TETRAHEDRON)
    {
        const auto& restPositions = l_geometryState->read(core::vec_id::read_access::restPosition)->getValue();

        helper::WriteAccessor<Data<MassVector> > masses(d_vertexMass);
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        typename DataTypes::Real md = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);
        unsigned int i;

        for (i = 0; i < tetrahedronAdded.size(); ++i)
        {
            /// get the tetrahedron to be added
            const Tetrahedron& t = this->l_topology->getTetrahedron(tetrahedronAdded[i]);

            /// compute its mass based on the mass density and the tetrahedron volume
            const auto& rpos0 = GeometricalTypes::getCPos(restPositions[t[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(restPositions[t[1]]);
            const auto& rpos2 = GeometricalTypes::getCPos(restPositions[t[2]]);
            const auto& rpos3 = GeometricalTypes::getCPos(restPositions[t[3]]);

            const auto restTetraVolume = sofa::geometry::Tetrahedron::volume(rpos0, rpos1, rpos2, rpos3);
            mass = (md * restTetraVolume) / (typename DataTypes::Real(4.0));

            /// added  mass on its four vertices
            masses[t[0]] += mass;
            masses[t[1]] += mass;
            masses[t[2]] += mass;
            masses[t[3]] += mass;

            totalMass += 4.0 * mass;
        }

        this->cleanTracker();
        printMass();
    }
}

template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 3, int > >
void DiagonalMass<DataTypes, GeometricalTypes>::applyTetrahedronDestruction(const sofa::type::vector<TetrahedronID> & tetrahedronRemoved)
{
    if (this->getMassTopologyType() == sofa::geometry::ElementType::TETRAHEDRON)
    {
        const auto& restPositions = l_geometryState->read(core::vec_id::read_access::restPosition)->getValue();

        helper::WriteAccessor<Data<MassVector> > masses(d_vertexMass);
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        typename DataTypes::Real md = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);
        unsigned int i;

        for (i = 0; i < tetrahedronRemoved.size(); ++i)
        {
            /// get the tetrahedron to be added
            const Tetrahedron& t = this->l_topology->getTetrahedron(tetrahedronRemoved[i]);

            /// compute its mass based on the mass density and the tetrahedron volume
            const auto& rpos0 = GeometricalTypes::getCPos(restPositions[t[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(restPositions[t[1]]);
            const auto& rpos2 = GeometricalTypes::getCPos(restPositions[t[2]]);
            const auto& rpos3 = GeometricalTypes::getCPos(restPositions[t[3]]);

            const auto restTetraVolume = sofa::geometry::Tetrahedron::volume(rpos0, rpos1, rpos2, rpos3);
            mass = (md * restTetraVolume) / (typename DataTypes::Real(4.0));

            // removed  mass on its four vertices
            masses[t[0]] -= mass;
            masses[t[1]] -= mass;
            masses[t[2]] -= mass;
            masses[t[3]] -= mass;

            totalMass -= 4.0 * mass;
        }

        this->cleanTracker();
        printMass();
    }
}


template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 3, int > >
void DiagonalMass<DataTypes, GeometricalTypes>::applyHexahedronCreation(const sofa::type::vector< HexahedronID >& hexahedronAdded,
        const sofa::type::vector< Hexahedron >& /*elems*/,
        const sofa::type::vector< sofa::type::vector< HexahedronID > >& /*ancestors*/,
        const sofa::type::vector< sofa::type::vector< SReal > >& /*coefs*/)
{
    if (this->getMassTopologyType() == sofa::geometry::ElementType::HEXAHEDRON)
    {
        const auto& restPositions = l_geometryState->read(core::vec_id::read_access::restPosition)->getValue();

        helper::WriteAccessor<Data<MassVector> > masses(d_vertexMass);
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        typename DataTypes::Real md = getMassDensity();
        typename DataTypes::Real mass = typename DataTypes::Real(0);
        unsigned int i;

        for (i = 0; i < hexahedronAdded.size(); ++i)
        {
            /// get the tetrahedron to be added
            const Hexahedron& h = this->l_topology->getHexahedron(hexahedronAdded[i]);
            // compute its mass based on the mass density and the tetrahedron volume
            const auto& rpos0 = GeometricalTypes::getCPos(restPositions[h[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(restPositions[h[1]]);
            const auto& rpos2 = GeometricalTypes::getCPos(restPositions[h[2]]);
            const auto& rpos3 = GeometricalTypes::getCPos(restPositions[h[3]]);
            const auto& rpos4 = GeometricalTypes::getCPos(restPositions[h[4]]);
            const auto& rpos5 = GeometricalTypes::getCPos(restPositions[h[5]]);
            const auto& rpos6 = GeometricalTypes::getCPos(restPositions[h[6]]);
            const auto& rpos7 = GeometricalTypes::getCPos(restPositions[h[7]]);

            const auto hexaVolume = sofa::geometry::Hexahedron::volume(rpos0, rpos1, rpos2, rpos3, rpos4, rpos5, rpos6, rpos7);
            mass = (md * hexaVolume) / (typename DataTypes::Real(8.0));

            // added  mass on its eight vertices
            for (unsigned int j = 0; j < 8; ++j)
                masses[h[j]] += mass;

            totalMass += 8.0 * mass;
        }

        this->cleanTracker();
        printMass();
    }
}

template <class DataTypes, class GeometricalTypes>
template <typename T, typename std::enable_if_t<T::spatial_dimensions >= 3, int > >
void DiagonalMass<DataTypes, GeometricalTypes>::applyHexahedronDestruction(const sofa::type::vector<HexahedronID> & hexahedronRemoved)
{
    if (this->getMassTopologyType() == sofa::geometry::ElementType::HEXAHEDRON)
    {
        const auto& restPositions = l_geometryState->read(core::vec_id::read_access::restPosition)->getValue();

        helper::WriteAccessor<Data<MassVector> > masses(d_vertexMass);
        helper::WriteAccessor<Data<Real> > totalMass(d_totalMass);

        typename DataTypes::Real md = getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;
        unsigned int i;

        for (i = 0; i < hexahedronRemoved.size(); ++i)
        {
            /// get the tetrahedron to be added
            const Hexahedron& h = this->l_topology->getHexahedron(hexahedronRemoved[i]);

            // compute its mass based on the mass density and the tetrahedron volume
            const auto& rpos0 = GeometricalTypes::getCPos(restPositions[h[0]]);
            const auto& rpos1 = GeometricalTypes::getCPos(restPositions[h[1]]);
            const auto& rpos2 = GeometricalTypes::getCPos(restPositions[h[2]]);
            const auto& rpos3 = GeometricalTypes::getCPos(restPositions[h[3]]);
            const auto& rpos4 = GeometricalTypes::getCPos(restPositions[h[4]]);
            const auto& rpos5 = GeometricalTypes::getCPos(restPositions[h[5]]);
            const auto& rpos6 = GeometricalTypes::getCPos(restPositions[h[6]]);
            const auto& rpos7 = GeometricalTypes::getCPos(restPositions[h[7]]);

            const auto hexaVolume = sofa::geometry::Hexahedron::volume(rpos0, rpos1, rpos2, rpos3, rpos4, rpos5, rpos6, rpos7);
            mass = (md * hexaVolume) / (typename DataTypes::Real(8.0));

            // removed  mass on its eight vertices
            for (unsigned int j = 0; j < 8; ++j)
                masses[h[j]] -= mass;

            totalMass -= 8.0 * mass;
        }

        this->cleanTracker();
        printMass();
    }
}


template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::clear()
{
    helper::WriteAccessor<Data<MassVector> > masses = d_vertexMass;
    masses.clear();
}

template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::addMass(const MassType& m)
{
    helper::WriteAccessor<Data<MassVector> > masses = d_vertexMass;
    masses.push_back(m);
}

template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::resize(int vsize)
{
    helper::WriteAccessor<Data<MassVector> > masses = d_vertexMass;
    masses.resize(vsize);
}

// -- Mass interface
template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::addMDx(const core::MechanicalParams* /*mparams*/, DataVecDeriv& res, const DataVecDeriv& dx, SReal factor)
{
    const MassVector &masses= d_vertexMass.getValue();
    helper::WriteOnlyAccessor< DataVecDeriv > _res = res;
    helper::ReadAccessor< DataVecDeriv > _dx = dx;

    size_t n = masses.size();
    if (_dx.size() < n) n = _dx.size();
    if (_res.size() < n) n = _res.size();
    if (factor == 1.0)
    {
        for (size_t i=0; i<n; i++)
        {
            _res[i] += _dx[i] * masses[i];
        }
    }
    else
    {
        for (size_t i=0; i<n; i++)
        {
            _res[i] += (_dx[i] * masses[i]) * Real(factor);
        }
    }
}



template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::accFromF(const core::MechanicalParams* /*mparams*/, DataVecDeriv& a, const DataVecDeriv& f)
{

    const MassVector &masses= d_vertexMass.getValue();
    helper::WriteOnlyAccessor< DataVecDeriv > _a = a;
    const VecDeriv& _f = f.getValue();

    for (unsigned int i=0; i<masses.size(); i++)
    {
        _a[i] = _f[i] / masses[i];
    }
}

template <class DataTypes, class GeometricalTypes>
SReal DiagonalMass<DataTypes, GeometricalTypes>::getKineticEnergy( const core::MechanicalParams* /*mparams*/, const DataVecDeriv& v ) const
{

    const MassVector &masses= d_vertexMass.getValue();
    helper::ReadAccessor< DataVecDeriv > _v = v;
    SReal e = 0.0;
    for (unsigned int i=0; i<masses.size(); i++)
    {
        e += _v[i]*masses[i]*_v[i]; // v[i]*v[i]*masses[i] would be more efficient but less generic
    }
    return e/2;
}

template <class DataTypes, class GeometricalTypes>
SReal DiagonalMass<DataTypes, GeometricalTypes>::getPotentialEnergy( const core::MechanicalParams* /*mparams*/, const DataVecCoord& x ) const
{

    const MassVector &masses= d_vertexMass.getValue();
    helper::ReadAccessor< DataVecCoord > _x = x;
    SReal e = 0;
    // gravity
    type::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<masses.size(); i++)
    {
        e -= type::dot(theGravity, _x[i]) * masses[i];
    }
    return e;
}

// does nothing by default, need to be specialized in .cpp
template <class DataTypes, class GeometricalTypes>
type::Vec6
DiagonalMass<DataTypes, GeometricalTypes>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& /*vx*/, const DataVecDeriv& /*vv*/  ) const
{
    return type::Vec6();
}

template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::addMToMatrix(sofa::linearalgebra::BaseMatrix * mat, SReal mFact, unsigned int &offset)
{
    const MassVector &masses= d_vertexMass.getValue();
    static constexpr auto N = Deriv::total_size;
    AddMToMatrixFunctor<Deriv,MassType, linearalgebra::BaseMatrix> calc;
    for (unsigned int i = 0; i < masses.size(); i++)
    {
        calc(mat, masses[i], offset + N * i, mFact);
    }
}

template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::buildMassMatrix(sofa::core::behavior::MassMatrixAccumulator* matrices)
{
    const MassVector &masses= d_vertexMass.getValue();
    static constexpr auto N = Deriv::total_size;
    AddMToMatrixFunctor<Deriv,MassType, sofa::core::behavior::MassMatrixAccumulator> calc;

    for (unsigned int i = 0; i < masses.size(); i++)
    {
        calc(matrices, masses[i], N * i, 1.);
    }
}


template <class DataTypes, class GeometricalTypes>
SReal DiagonalMass<DataTypes, GeometricalTypes>::getElementMass(sofa::Index index) const
{
    return SReal(d_vertexMass.getValue()[index]);
}


//TODO: special case for Rigid Mass
template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::getElementMass(sofa::Index index, linearalgebra::BaseMatrix *m) const
{
    static const linearalgebra::BaseMatrix::Index dimension = linearalgebra::BaseMatrix::Index(defaulttype::DataTypeInfo<Deriv>::size());
    if (m->rowSize() != dimension || m->colSize() != dimension) m->resize(dimension,dimension);

    m->clear();
    AddMToMatrixFunctor<Deriv,MassType, linearalgebra::BaseMatrix>()(m, d_vertexMass.getValue()[index], 0, 1);
}

template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::reinit()
{
    // Now update is handled through the doUpdateInternal mechanism
    // called at each begin of step through the UpdateInternalDataVisitor
}

template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::initTopologyHandlers()
{
    // add the functions to handle topology changes.
    d_vertexMass.createTopologyHandler(l_topology);
    d_vertexMass.setCreationCallback([this](Index pointIndex, MassType& m,
        const core::topology::BaseMeshTopology::Point& point,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs)
    {
        applyPointCreation(pointIndex, m, point, ancestors, coefs);
    });
    d_vertexMass.setDestructionCallback([this](Index pointIndex, MassType& m)
    {
        applyPointDestruction(pointIndex, m);
    });

    if (m_massTopologyType == sofa::geometry::ElementType::EDGE)
    {
        d_vertexMass.linkToEdgeDataArray();
        d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::EDGESADDED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::EdgesAdded* edgeAdd = static_cast<const core::topology::EdgesAdded*>(eventTopo);
            applyEdgeCreation(edgeAdd->getIndexArray(), edgeAdd->getElementArray(), edgeAdd->ancestorsList, edgeAdd->coefs);
        });
        d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::EDGESREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
            const core::topology::EdgesRemoved* edgeRemove = static_cast<const core::topology::EdgesRemoved*>(eventTopo);
            applyEdgeDestruction(edgeRemove->getArray());
        });
    }

    if constexpr (DataTypes::spatial_dimensions >= 3)
    {
        if (m_massTopologyType == sofa::geometry::ElementType::TETRAHEDRON)
        {
            d_vertexMass.linkToTetrahedronDataArray();
            d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TETRAHEDRAADDED, [this](const core::topology::TopologyChange* eventTopo) {
                const core::topology::TetrahedraAdded* tAdd = static_cast<const core::topology::TetrahedraAdded*>(eventTopo);
                applyTetrahedronCreation(tAdd->getIndexArray(), tAdd->getElementArray(), tAdd->ancestorsList, tAdd->coefs);
                });
            d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TETRAHEDRAREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
                const core::topology::TetrahedraRemoved* tRemove = static_cast<const core::topology::TetrahedraRemoved*>(eventTopo);
                applyTetrahedronDestruction(tRemove->getArray());
                });
        }
        if (m_massTopologyType == sofa::geometry::ElementType::HEXAHEDRON)
        {
            d_vertexMass.linkToHexahedronDataArray();
            d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::HEXAHEDRAADDED, [this](const core::topology::TopologyChange* eventTopo) {
                const core::topology::HexahedraAdded* hAdd = static_cast<const core::topology::HexahedraAdded*>(eventTopo);
                applyHexahedronCreation(hAdd->getIndexArray(), hAdd->getElementArray(), hAdd->ancestorsList, hAdd->coefs);
                });
            d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::HEXAHEDRAREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
                const core::topology::HexahedraRemoved* hRemove = static_cast<const core::topology::HexahedraRemoved*>(eventTopo);
                applyHexahedronDestruction(hRemove->getArray());
                });
        }
    }

    if constexpr (DataTypes::spatial_dimensions >= 2)
    {
        if (m_massTopologyType == sofa::geometry::ElementType::TRIANGLE)
        {
            d_vertexMass.linkToTriangleDataArray();
            d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TRIANGLESADDED, [this](const core::topology::TopologyChange* eventTopo) {
                const core::topology::TrianglesAdded* tAdd = static_cast<const core::topology::TrianglesAdded*>(eventTopo);
                applyTriangleCreation(tAdd->getIndexArray(), tAdd->getElementArray(), tAdd->ancestorsList, tAdd->coefs);
            });
            d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::TRIANGLESREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
                const core::topology::TrianglesRemoved* tRemove = static_cast<const core::topology::TrianglesRemoved*>(eventTopo);
                applyTriangleDestruction(tRemove->getArray());
            });
        }
        if (m_massTopologyType == sofa::geometry::ElementType::QUAD)
        {
            d_vertexMass.linkToQuadDataArray();
            d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::QUADSADDED, [this](const core::topology::TopologyChange* eventTopo) {
                const core::topology::QuadsAdded* qAdd = static_cast<const core::topology::QuadsAdded*>(eventTopo);
                applyQuadCreation(qAdd->getIndexArray(), qAdd->getElementArray(), qAdd->ancestorsList, qAdd->coefs);
            });
            d_vertexMass.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::QUADSREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
                const core::topology::QuadsRemoved* qRemove = static_cast<const core::topology::QuadsRemoved*>(eventTopo);
                applyQuadDestruction(qRemove->getArray());
            });
        }
    }
}

template <class DataTypes, class GeometricalTypes>
bool DiagonalMass<DataTypes, GeometricalTypes>::checkTopology()
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
        return false;
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
            return false;
        }
        else
        {
            typename sofa::core::behavior::MechanicalState<GeometricalTypes>::SPtr geometryState = boost::dynamic_pointer_cast<sofa::core::behavior::MechanicalState<GeometricalTypes>>(baseState);
            if (geometryState == nullptr)
            {
                msg_error() << "A state associated with the topology has been found but is incompatible with the definition of the mass (templates mismatch).";
                return false;
            }
            else
            {
                l_geometryState.set(geometryState);
                msg_info() << "Topology is associated with the state: '" << l_geometryState->getPathName() << "'";
            }
        }
    }

    if (l_topology)
    {
        if (l_topology->getNbHexahedra() > 0)
        {
            msg_info() << "Hexahedral topology found.";
            m_massTopologyType = sofa::geometry::ElementType::HEXAHEDRON;
            return true;
        }
        else if (l_topology->getNbTetrahedra() > 0)
        {
            msg_info() << "Tetrahedral topology found.";
            m_massTopologyType = sofa::geometry::ElementType::TETRAHEDRON;
            return true;
        }
        else if (l_topology->getNbQuads() > 0)
        {
            msg_info() << "Quad topology found.";
            m_massTopologyType = sofa::geometry::ElementType::QUAD;
            return true;
        }
        else if (l_topology->getNbTriangles() > 0)
        {
            msg_info() << "Triangular topology found."; 
            m_massTopologyType = sofa::geometry::ElementType::TRIANGLE;
            return true;
        }
        else if (l_topology->getNbEdges() > 0)
        {
            msg_info() << "Edge topology found.";
            m_massTopologyType = sofa::geometry::ElementType::EDGE;
            return true;
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

template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::init()
{
    this->d_componentState.setValue(ComponentState::Invalid);

    if (!d_fileMass.getValue().empty())
    {
        if(!load(d_fileMass.getFullPath().c_str())){
            return;
        }
        msg_warning() << "File given as input for DiagonalMass, in this a case:" << msgendl
                      << "the topology won't be used to compute the mass" << msgendl
                      << "the update, the coherency and the tracking of mass information data are disable (listening = false)";
        Inherited::init();
    }
    else
    {
        Inherited::init();
        if(!checkTopology())
        {
            return;
        }
        initTopologyHandlers();

        // TODO(dmarchal 2018-11-10): this code is duplicated with the one in RigidImpl we should factor it (remove in 1 year if not done or update the dates)
        if (this->mstate && d_vertexMass.getValue().size() > 0 && d_vertexMass.getValue().size() < unsigned(this->mstate->getSize()))
        {
            helper::WriteAccessor<Data<MassVector> > masses = d_vertexMass;
            size_t i = masses.size()-1;
            size_t n = size_t(this->mstate->getSize());
            masses.reserve(n);
            while (masses.size() < n)
                masses.push_back(masses[i]);
        }

        massInitialization();

        this->trackInternalData(d_vertexMass);
        this->trackInternalData(d_massDensity);
        this->trackInternalData(d_totalMass);
    }
    this->d_componentState.setValue(ComponentState::Valid);
}


template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::massInitialization()
{
    //Mass initialization process
    if(d_vertexMass.isSet() || d_massDensity.isSet() || d_totalMass.isSet() )
    {
        //totalMass data has priority on vertexMass and massDensity
        if (d_totalMass.isSet())
        {
            if(d_vertexMass.isSet() || d_massDensity.isSet())
            {
                msg_warning() << "totalMass value overriding other mass information (vertexMass or massDensity).\n"
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
                msg_warning() << "massDensity value overriding the value of the attribute vertexMass.\n"
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
void DiagonalMass<DataTypes, GeometricalTypes>::printMass()
{
    if (this->f_printLog.getValue() == false)
        return;

    const MassVector &vertexM = d_vertexMass.getValue();

    Real average_vertex = 0.0;
    Real min_vertex = std::numeric_limits<Real>::max();
    Real max_vertex = 0.0;

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

    msg_info() << "mass information computed :" << msgendl
               << "totalMass   = " << d_totalMass.getValue() << msgendl
               << "massDensity = " << d_massDensity.getValue() << msgendl
               << "mean vertexMass [min,max] = " << average_vertex << " [" << min_vertex << "," <<  max_vertex <<"]";
}

template <class DataTypes, class GeometricalTypes>
typename DiagonalMass<DataTypes, GeometricalTypes>::Real DiagonalMass<DataTypes, GeometricalTypes>::computeVertexMass(Real density)
{
    Real total_mass = Real(0);

    if (l_topology == nullptr)
    {
        msg_warning() << "No topology set. DiagonalMass can't computeMass.";
        return total_mass;
    }
    
    core::ConstVecCoordId posid = d_computeMassOnRest.getValue() ? core::vec_id::read_access::restPosition : core::vec_id::read_access::position;
    const auto& positions = l_geometryState->read(posid)->getValue();

    Real mass = Real(0);
    helper::WriteAccessor<Data<MassVector> > masses = d_vertexMass;
    // resize array
    masses.clear();
    masses.resize(this->mstate->getSize());

    if constexpr (GeometricalTypes::spatial_dimensions >= 1)
    {
        if (m_massTopologyType == sofa::geometry::ElementType::EDGE)
        {
            for (Topology::EdgeID i = 0; i < l_topology->getNbEdges(); ++i)
            {
                const Edge& e = l_topology->getEdge(i);

                const auto& pos0 = GeometricalTypes::getCPos(positions[e[0]]);
                const auto& pos1 = GeometricalTypes::getCPos(positions[e[1]]);

                const auto edgeLength = sofa::geometry::Edge::length(pos0, pos1);
                mass = (density * edgeLength) / (Real(2.0));

                for (unsigned int j = 0; j < e.size(); j++)
                {
                    masses[e[j]] += mass;
                    total_mass += mass;
                }
            }
        }
    }

    if constexpr (GeometricalTypes::spatial_dimensions >= 2)
    {
        if (m_massTopologyType == sofa::geometry::ElementType::TRIANGLE)
        {
            for (Topology::TriangleID i = 0; i < l_topology->getNbTriangles(); ++i)
            {
                const Triangle& t = l_topology->getTriangle(i);

                const auto& pos0 = GeometricalTypes::getCPos(positions[t[0]]);
                const auto& pos1 = GeometricalTypes::getCPos(positions[t[1]]);
                const auto& pos2 = GeometricalTypes::getCPos(positions[t[2]]);

                const auto triangleArea = sofa::geometry::Triangle::area(pos0, pos1, pos2);
                mass = (density * triangleArea) / (Real(3.0));

                for (unsigned int j = 0; j < t.size(); j++)
                {
                    masses[t[j]] += mass;
                    total_mass += mass;
                }
            }
        }

        if (m_massTopologyType == sofa::geometry::ElementType::QUAD)
        {
            for (Topology::QuadID i = 0; i < l_topology->getNbQuads(); ++i)
            {
                const Quad& q = l_topology->getQuad(i);

                const auto& pos0 = GeometricalTypes::getCPos(positions[q[0]]);
                const auto& pos1 = GeometricalTypes::getCPos(positions[q[1]]);
                const auto& pos2 = GeometricalTypes::getCPos(positions[q[2]]);
                const auto& pos3 = GeometricalTypes::getCPos(positions[q[3]]);

                const auto quadArea = sofa::geometry::Quad::area(pos0, pos1, pos2, pos3);
                mass = (density * quadArea) / (Real(4.0));
                for (unsigned int j = 0; j < q.size(); j++)
                {
                    masses[q[j]] += mass;
                    total_mass += mass;
                }
            }
        }
    }

    if constexpr (GeometricalTypes::spatial_dimensions >= 3)
    {
        if (m_massTopologyType == sofa::geometry::ElementType::TETRAHEDRON)
        {
            for (Topology::TetrahedronID i = 0; i < l_topology->getNbTetrahedra(); ++i)
            {
                const Tetrahedron& t = l_topology->getTetrahedron(i);

                /// compute its mass based on the mass density and the tetrahedron volume
                const auto& rpos0 = GeometricalTypes::getCPos(positions[t[0]]);
                const auto& rpos1 = GeometricalTypes::getCPos(positions[t[1]]);
                const auto& rpos2 = GeometricalTypes::getCPos(positions[t[2]]);
                const auto& rpos3 = GeometricalTypes::getCPos(positions[t[3]]);

                const auto tetraVolume = sofa::geometry::Tetrahedron::volume(rpos0, rpos1, rpos2, rpos3);
                mass = (density * tetraVolume) / (typename DataTypes::Real(4.0));
                for (unsigned int j = 0; j < t.size(); j++)
                {
                    masses[t[j]] += mass;
                    total_mass += mass;
                }
            }
        }
        if (m_massTopologyType == sofa::geometry::ElementType::HEXAHEDRON)
        {
            for (Topology::HexahedronID i = 0; i < l_topology->getNbHexahedra(); ++i)
            {
                const Hexahedron& h = l_topology->getHexahedron(i);

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
                mass = (density * hexaVolume) / (typename DataTypes::Real(8.0));

                for (unsigned int j = 0; j < h.size(); j++)
                {
                    masses[h[j]] += mass;
                    total_mass += mass;
                }
            }
        }
    }

    return total_mass;
}

template <class DataTypes, class GeometricalTypes>
bool DiagonalMass<DataTypes, GeometricalTypes>::checkTotalMass()
{
    //Check for negative or null value, if wrongly set use the default value totalMass = 1.0
    if(d_totalMass.getValue() < 0.0)
    {
        msg_warning() << "totalMass data can not have a negative value.\n"
                      << "To remove this warning, you need to set a strictly positive value to the totalMass data";
        return false;
    }
    else
    {
        return true;
    }
}


template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::checkTotalMassInit()
{
    //Check for negative or null value, if wrongly set use the default value totalMass = 1.0
    if(!checkTotalMass())
    {
        d_totalMass.setValue(1.0) ;
        msg_warning() << "Switching back to default values: totalMass = 1.0\n";
        this->d_componentState.setValue(ComponentState::Invalid);
    }
}


template <class DataTypes, class GeometricalTypes>
bool DiagonalMass<DataTypes, GeometricalTypes>::checkVertexMass()
{
    const MassVector &vertexMass = d_vertexMass.getValue();

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
void DiagonalMass<DataTypes, GeometricalTypes>::initFromVertexMass()
{
    msg_info() << "vertexMass information is used";

    // saving a copy of vertexMass vector to recover the input values after init methods which are overwritten those values.
    const MassVector vertexMass = d_vertexMass.getValue();
    const Real totalMassSave = std::accumulate(vertexMass.begin(), vertexMass.end(), Real(0));

    // set total mass
    d_totalMass.setValue(totalMassSave);

    // compute vertexMass vector with density == 1
    const Real sumMass = computeVertexMass(1.0);

    // Set real density from sumMass found
    if (sumMass < std::numeric_limits<typename DataTypes::Real>::epsilon())
        setMassDensity(1.0);
    else
        setMassDensity(Real(totalMassSave / sumMass));

    // restore input vertexMass vector
    helper::WriteAccessor<Data<MassVector> > vertexMassWrite = d_vertexMass;
    for(size_t i=0; i<vertexMassWrite.size(); i++)
    {
        vertexMassWrite[i] = vertexMass[i];
    }
}


template <class DataTypes, class GeometricalTypes>
bool DiagonalMass<DataTypes, GeometricalTypes>::checkMassDensity()
{
    const Real &massDensity = d_massDensity.getValue();

    //Check that the massDensity is strictly positive
    if(massDensity < 0.0)
    {
        msg_warning() << "Negative value of massDensity: massDensity = " << massDensity;
        return false;
    }
    else
    {
        return true;
    }
}


template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::initFromMassDensity()
{
    msg_info() << "massDensity information is used";

    // Compute Mass per vertex using mesh topology
    const Real& md = d_massDensity.getValue();
    const Real sumMass = computeVertexMass(md);

    // sum of mass per vertex give total mass
    d_totalMass.setValue(sumMass);
}


template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::initFromTotalMass()
{
    msg_info() << "totalMass information is used";

    const Real totalMass = d_totalMass.getValue();
    
    // compute vertexMass vector with density == 1
    const Real sumMass = computeVertexMass(1.0);

    // Set real density from sumMass found
    if (sumMass < std::numeric_limits<typename DataTypes::Real>::epsilon())
        setMassDensity(1.0);
    else
        setMassDensity(Real(totalMass / sumMass));
    
    // Update vertex mass using real density
    helper::WriteAccessor<Data<MassVector> > vertexMass = d_vertexMass;
    const Real& density = d_massDensity.getValue();
    for (auto& vm : vertexMass)
    {
        vm *= density;
    }
}


template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::setVertexMass(sofa::type::vector< Real > vertexMass)
{
    const MassVector currentVertexMass = d_vertexMass.getValue();
    helper::WriteAccessor<Data<MassVector> > vertexMassWrite = d_vertexMass;
    vertexMassWrite.resize(vertexMass.size());
    for(size_t i=0; i<vertexMass.size(); i++)
    {
        vertexMassWrite[i] = vertexMass[i];
    }

    if(!checkVertexMass())
    {
        msg_warning() << "Given values to setVertexMass() are not correct.\n"
                      << "Previous values are used.";
        d_vertexMass.setValue(currentVertexMass);
    }
}


template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::setMassDensity(Real massDensityValue)
{
    const Real currentMassDensity = d_massDensity.getValue();
    d_massDensity.setValue(massDensityValue);
    if(!checkMassDensity())
    {
        msg_warning() << "Given values to setMassDensity() are not correct.\n"
                      << "Previous values are used.";
        d_massDensity.setValue(currentMassDensity);
    }
}


template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::setTotalMass(Real totalMass)
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
const typename DiagonalMass<DataTypes, GeometricalTypes>::Real &DiagonalMass<DataTypes, GeometricalTypes>::getMassDensity()
{
    return d_massDensity.getValue();
}


template <class DataTypes, class GeometricalTypes>
const typename DiagonalMass<DataTypes, GeometricalTypes>::Real &DiagonalMass<DataTypes, GeometricalTypes>::getTotalMass()
{
    return d_totalMass.getValue();
}


template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::doUpdateInternal()
{
    if (this->hasDataChanged(d_totalMass))
    {
        if(checkTotalMass())
        {
            initFromTotalMass();
            this->d_componentState.setValue(ComponentState::Valid);
        }
        else
        {
            msg_error() << "doUpdateInternal: incorrect update from totalMass";
            this->d_componentState.setValue(ComponentState::Invalid);
        }
    }
    else if(this->hasDataChanged(d_massDensity))
    {
        if(checkMassDensity())
        {
            initFromMassDensity();
            this->d_componentState.setValue(ComponentState::Valid);
        }
        else
        {
            msg_error() << "doUpdateInternal: incorrect update from massDensity";
            this->d_componentState.setValue(ComponentState::Invalid);
        }
    }
    else if(this->hasDataChanged(d_vertexMass))
    {
        if(checkVertexMass())
        {
            initFromVertexMass();
            this->d_componentState.setValue(ComponentState::Valid);
        }
        else
        {
            msg_error() << "doUpdateInternal: incorrect update from vertexMass";
            this->d_componentState.setValue(ComponentState::Invalid);
        }
    }

    //Info post-init
    msg_info() << "mass information updated";
    printMass();
}


template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if(mparams)
    {
        VecDeriv& v = *d_v.beginEdit();
        // gravity
        sofa::type::Vec3d g ( this->getContext()->getGravity() );
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2]);
        Deriv hg = theGravity * typename DataTypes::Real(sofa::core::mechanicalparams::dt(mparams));

        for (unsigned int i=0; i<v.size(); i++)
        {
            v[i] += hg;
        }
        d_v.endEdit();
    }
}


template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& , const DataVecDeriv& )
{
    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if(this->m_separateGravity.getValue())
        return;

    const MassVector &masses= d_vertexMass.getValue();
    helper::WriteOnlyAccessor< DataVecDeriv > _f = f;

    // gravity
    sofa::type::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);


    // add weight and inertia force
    for (unsigned int i=0; i<masses.size(); i++)
    {
        _f[i] += theGravity*masses[i];
    }
}

template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    const MassVector &masses= d_vertexMass.getValue();
    if (masses.empty())
        return;

    const auto& x = l_geometryState->read(core::vec_id::read_access::position)->getValue();
    type::Vec3 gravityCenter;
    Real totalMass=0.0;

    std::vector<  sofa::type::Vec3 > points;

    for (unsigned int i = 0; i < x.size(); i++)
    {
        const sofa::type::Vec3 p = toVec3(GeometricalTypes::getCPos(x[i]));

        points.push_back(p);
        gravityCenter += p * masses[i];
        totalMass += masses[i];
    }

    if ( d_showCenterOfGravity.getValue() )
    {
        gravityCenter /= totalMass;

        Real axisSize = d_showAxisSize.getValue();
        sofa::type::Vec3 temp;

        for ( unsigned int i=0 ; i<3 ; i++ )
            if(i < Coord::spatial_dimensions )
                temp[i] = gravityCenter[i];

        vparams->drawTool()->drawCross(temp, float(axisSize), sofa::type::RGBAColor::yellow());
    }
}

template <class DataTypes, class GeometricalTypes>
class DiagonalMass<DataTypes, GeometricalTypes>::Loader : public helper::io::XspLoaderDataHook
{
public:
    DiagonalMass<DataTypes, GeometricalTypes>* dest;
    Loader(DiagonalMass<DataTypes, GeometricalTypes>* dest) : dest(dest) {}
    void addMass(SReal /*px*/, SReal /*py*/, SReal /*pz*/, SReal /*vx*/, SReal /*vy*/, SReal /*vz*/, SReal mass, SReal /*elastic*/, bool /*fixed*/, bool /*surface*/) override
    {
        dest->addMass(MassType(Real(mass)));
    }
};

template <class DataTypes, class GeometricalTypes>
bool DiagonalMass<DataTypes, GeometricalTypes>::load(const char *filename)
{
    clear();
    if (filename!=nullptr && filename[0]!='\0')
    {
        Loader loader(this);
        return helper::io::XspLoader::Load(filename, loader);
    }
    return false;
}


template <class DataTypes, class GeometricalTypes>
void DiagonalMass<DataTypes, GeometricalTypes>::handleEvent(sofa::core::objectmodel::Event *event)
{
     SOFA_UNUSED(event);
}

} // namespace sofa::component::mass
