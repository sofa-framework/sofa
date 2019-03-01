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
#ifndef SOFA_COMPONENT_MASS_DIAGONALMASS_INL
#define SOFA_COMPONENT_MASS_DIAGONALMASS_INL

#include <SofaBaseMechanics/DiagonalMass.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/io/XspLoader.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaBaseMechanics/AddMToMatrixFunctor.h>
#include <sofa/simulation/AnimateEndEvent.h>


namespace sofa
{

namespace component
{

namespace mass
{

using sofa::core::objectmodel::ComponentState;
using namespace sofa::core::topology;

template <class DataTypes, class MassType>
DiagonalMass<DataTypes, MassType>::DiagonalMass()
    : d_vertexMass( initData(&d_vertexMass, "vertexMass", "Specify a vector giving the mass of each vertex. \n"
                                                          "If unspecified or wrongly set, the massDensity or totalMass information is used.") )
    , d_massDensity( initData(&d_massDensity, (Real)1.0,"massDensity","Specify one single real and positive value for the mass density. \n"
                                                                      "If unspecified or wrongly set, the totalMass information is used.") )
    , d_computeMassOnRest(initData(&d_computeMassOnRest, false, "computeMassOnRest", "If true, the mass of every element is computed based on the rest position rather than the position"))
    , d_totalMass(initData(&d_totalMass, (Real)1.0, "totalMass", "Specify the total mass resulting from all particles. \n"
                                                                  "If unspecified or wrongly set, the default value is used: totalMass = 1.0"))
    , d_showCenterOfGravity( initData(&d_showCenterOfGravity, false, "showGravityCenter", "Display the center of gravity of the system" ) )
    , d_showAxisSize( initData(&d_showAxisSize, 1.0f, "showAxisSizeFactor", "Factor length of the axis displayed (only used for rigids)" ) )
    , d_fileMass( initData(&d_fileMass,  "fileMass", "Xsp3.0 file to specify the mass parameters" ) )
    , m_pointHandler(NULL)
    , m_topologyType(TOPOLOGY_UNKNOWN)
{
    this->addAlias(&d_fileMass,"filename");
}

template <class DataTypes, class MassType>
DiagonalMass<DataTypes, MassType>::~DiagonalMass()
{
    if (m_pointHandler)
        delete m_pointHandler;
}


template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::applyCreateFunction(unsigned int, MassType &m, const Point &, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    m=0;
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::applyEdgeCreation(const sofa::helper::vector< unsigned int >& edgeAdded,
        const sofa::helper::vector< Edge >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    if (dm->getMassTopologyType()==DiagonalMass<DataTypes, MassType>::TOPOLOGY_EDGESET)
    {

        helper::WriteAccessor<Data<MassVector> > masses(this->m_topologyData);
        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass=(typename DataTypes::Real) 0;
        unsigned int i;

        for (i=0; i<edgeAdded.size(); ++i)
        {
            /// get the edge to be added
            const Edge &e=dm->_topology->getEdge(edgeAdded[i]);
            // compute its mass based on the mass density and the edge length
            if(dm->edgeGeo)
            {
                mass=(md*dm->edgeGeo->computeRestEdgeLength(edgeAdded[i]))/(typename DataTypes::Real)2.0;
            }
            // added mass on its two vertices
            masses[e[0]]+=mass;
            masses[e[1]]+=mass;
        }
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::applyEdgeDestruction(const sofa::helper::vector<unsigned int> & edgeRemoved)
{
    if (dm->getMassTopologyType()==DiagonalMass<DataTypes, MassType>::TOPOLOGY_EDGESET)
    {
        helper::WriteAccessor<Data<MassVector> > masses(*this->m_topologyData);

        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass=(typename DataTypes::Real) 0;
        unsigned int i;

        for (i=0; i<edgeRemoved.size(); ++i)
        {
            /// get the edge to be added
            const Edge &e=dm->_topology->getEdge(edgeRemoved[i]);
            // compute its mass based on the mass density and the edge length
            if(dm->edgeGeo)
            {
                mass=(md*dm->edgeGeo->computeRestEdgeLength(edgeRemoved[i]))/(typename DataTypes::Real)2.0;
            }
            // removed mass on its two vertices
            masses[e[0]]-=mass;
            masses[e[1]]-=mass;
        }
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::ApplyTopologyChange(const core::topology::EdgesAdded* e)
{
    const sofa::helper::vector< unsigned int >& edgeIndex = e->getIndexArray();
    const sofa::helper::vector< Edge >& edges = e->getArray();
    const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = e->ancestorsList;
    const sofa::helper::vector< sofa::helper::vector< double > >& coeffs = e->coefs;

    applyEdgeCreation(edgeIndex, edges, ancestors, coeffs);
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::ApplyTopologyChange(const core::topology::EdgesRemoved* e)
{
    const sofa::helper::vector<unsigned int> & edgeRemoved = e->getArray();

    applyEdgeDestruction(edgeRemoved);
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::applyTriangleCreation(const sofa::helper::vector< unsigned int >& triangleAdded,
        const sofa::helper::vector< Triangle >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    if (dm->getMassTopologyType()==DiagonalMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET)
    {
        helper::WriteAccessor<Data<MassVector> > masses(*this->m_topologyData);

        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass=(typename DataTypes::Real) 0;
        unsigned int i;

        for (i=0; i<triangleAdded.size(); ++i)
        {
            /// get the triangle to be added
            const Triangle &t=dm->_topology->getTriangle(triangleAdded[i]);
            // compute its mass based on the mass density and the triangle area
            if(dm->triangleGeo)
            {
                mass=(md*dm->triangleGeo->computeRestTriangleArea(triangleAdded[i]))/(typename DataTypes::Real)3.0;
            }
            // added mass on its three vertices
            masses[t[0]]+=mass;
            masses[t[1]]+=mass;
            masses[t[2]]+=mass;
        }
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::applyTriangleDestruction(const sofa::helper::vector<unsigned int> & triangleRemoved)
{
    if (dm->getMassTopologyType()==DiagonalMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET)
    {
        helper::WriteAccessor<Data<MassVector> > masses(*this->m_topologyData);

        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass=(typename DataTypes::Real) 0;
        unsigned int i;

        for (i=0; i<triangleRemoved.size(); ++i)
        {
            /// get the triangle to be added
            const Triangle &t=dm->_topology->getTriangle(triangleRemoved[i]);

            /// compute its mass based on the mass density and the triangle area
            if(dm->triangleGeo)
            {
                mass=(md*dm->triangleGeo->computeRestTriangleArea(triangleRemoved[i]))/(typename DataTypes::Real)3.0;
            }

            /// removed  mass on its three vertices
            masses[t[0]]-=mass;
            masses[t[1]]-=mass;
            masses[t[2]]-=mass;
        }
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::ApplyTopologyChange(const core::topology::TrianglesAdded* e)
{
    const sofa::helper::vector< unsigned int >& triangleAdded = e->getIndexArray();
    const sofa::helper::vector< Triangle >& elems = e->getElementArray();
    const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = e->ancestorsList;
    const sofa::helper::vector< sofa::helper::vector< double > >& coefs = e->coefs;

    applyTriangleCreation(triangleAdded,elems,ancestors,coefs);
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::ApplyTopologyChange(const core::topology::TrianglesRemoved* e)
{
    const sofa::helper::vector<unsigned int> & triangleRemoved = e->getArray();

    applyTriangleDestruction(triangleRemoved);
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::applyTetrahedronCreation(const sofa::helper::vector< unsigned int >& tetrahedronAdded,
        const sofa::helper::vector< Tetrahedron >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    if (dm->getMassTopologyType()==DiagonalMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET)
    {
        helper::WriteAccessor<Data<MassVector> > masses(*this->m_topologyData);

        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass=(typename DataTypes::Real) 0;
        unsigned int i;

        for (i=0; i<tetrahedronAdded.size(); ++i)
        {
            /// get the tetrahedron to be added
            const Tetrahedron &t=dm->_topology->getTetrahedron(tetrahedronAdded[i]);

            /// compute its mass based on the mass density and the tetrahedron volume
            if(dm->tetraGeo)
            {
                mass=(md*dm->tetraGeo->computeRestTetrahedronVolume(tetrahedronAdded[i]))/(typename DataTypes::Real)4.0;
            }

            /// added  mass on its four vertices
            masses[t[0]]+=mass;
            masses[t[1]]+=mass;
            masses[t[2]]+=mass;
            masses[t[3]]+=mass;

        }

    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::applyTetrahedronDestruction(const sofa::helper::vector<unsigned int> & tetrahedronRemoved)
{
    if (dm->getMassTopologyType()==DiagonalMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET)
    {
        helper::WriteAccessor<Data<MassVector> > masses(*this->m_topologyData);

        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass=(typename DataTypes::Real) 0;
        unsigned int i;

        for (i=0; i<tetrahedronRemoved.size(); ++i)
        {
            /// get the tetrahedron to be added
            const Tetrahedron &t=dm->_topology->getTetrahedron(tetrahedronRemoved[i]);
            if(dm->tetraGeo)
            {
                // compute its mass based on the mass density and the tetrahedron volume
                mass=(md*dm->tetraGeo->computeRestTetrahedronVolume(tetrahedronRemoved[i]))/(typename DataTypes::Real)4.0;
            }
            // removed  mass on its four vertices
            masses[t[0]]-=mass;
            masses[t[1]]-=mass;
            masses[t[2]]-=mass;
            masses[t[3]]-=mass;
        }

    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::ApplyTopologyChange(const core::topology::TetrahedraAdded* e)
{
    const sofa::helper::vector< unsigned int >& tetrahedronAdded = e->getIndexArray();
    const sofa::helper::vector< Tetrahedron >& elems = e->getElementArray();
    const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = e->ancestorsList;
    const sofa::helper::vector< sofa::helper::vector< double > >& coefs = e->coefs;

    applyTetrahedronCreation(tetrahedronAdded, elems, ancestors, coefs);
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::ApplyTopologyChange(const core::topology::TetrahedraRemoved* e)
{
    const sofa::helper::vector<unsigned int> & tetrahedronRemoved = e->getArray();

    applyTetrahedronDestruction(tetrahedronRemoved);
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::applyHexahedronCreation(const sofa::helper::vector< unsigned int >& hexahedronAdded,
        const sofa::helper::vector< Hexahedron >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    if (dm->getMassTopologyType()==DiagonalMass<DataTypes, MassType>::TOPOLOGY_HEXAHEDRONSET)
    {
        helper::WriteAccessor<Data<MassVector> > masses(*this->m_topologyData);

        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass=(typename DataTypes::Real) 0;
        unsigned int i;

        for (i=0; i<hexahedronAdded.size(); ++i)
        {
            /// get the tetrahedron to be added
            const Hexahedron &t=dm->_topology->getHexahedron(hexahedronAdded[i]);
            // compute its mass based on the mass density and the tetrahedron volume
            if(dm->hexaGeo)
            {
                mass=(md*dm->hexaGeo->computeRestHexahedronVolume(hexahedronAdded[i]))/(typename DataTypes::Real)8.0;
            }
            // added  mass on its four vertices
            for (unsigned int j=0; j<8; ++j)
                masses[t[j]]+=mass;
        }

    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::applyHexahedronDestruction(const sofa::helper::vector<unsigned int> & hexahedronRemoved)
{
    if (dm->getMassTopologyType()==DiagonalMass<DataTypes, MassType>::TOPOLOGY_HEXAHEDRONSET)
    {
        helper::WriteAccessor<Data<MassVector> > masses(*this->m_topologyData);

        typename DataTypes::Real md=dm->getMassDensity();
        typename DataTypes::Real mass=(typename DataTypes::Real) 0;
        unsigned int i;

        for (i=0; i<hexahedronRemoved.size(); ++i)
        {
            /// get the tetrahedron to be added
            const Hexahedron &t=dm->_topology->getHexahedron(hexahedronRemoved[i]);
            if(dm->hexaGeo)
            {
                // compute its mass based on the mass density and the tetrahedron volume
                mass=(md*dm->hexaGeo->computeRestHexahedronVolume(hexahedronRemoved[i]))/(typename DataTypes::Real)8.0;
            }
            // removed  mass on its four vertices
            for (unsigned int j=0; j<8; ++j)
                masses[t[j]]-=mass;
        }

    }
}
template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::ApplyTopologyChange(const core::topology::HexahedraAdded* e)
{
    const sofa::helper::vector< unsigned int >& hexahedronAdded = e->getIndexArray();
    const sofa::helper::vector< Hexahedron >& elems = e->getElementArray();
    const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = e->ancestorsList;
    const sofa::helper::vector< sofa::helper::vector< double > >& coefs = e->coefs;

    applyHexahedronCreation(hexahedronAdded,elems,ancestors,coefs);
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes,MassType>::DMassPointHandler::ApplyTopologyChange(const core::topology::HexahedraRemoved* e)
{
    const sofa::helper::vector<unsigned int> & hexahedronRemoved = e->getArray();

    applyHexahedronDestruction(hexahedronRemoved);
}



template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::clear()
{
    MassVector& masses = *d_vertexMass.beginEdit();
    masses.clear();
    d_vertexMass.endEdit();
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::addMass(const MassType& m)
{
    MassVector& masses = *d_vertexMass.beginEdit();
    masses.push_back(m);
    d_vertexMass.endEdit();
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::resize(int vsize)
{
    MassVector& masses = *d_vertexMass.beginEdit();
    masses.resize(vsize);
    d_vertexMass.endEdit();
}

// -- Mass interface
template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::addMDx(const core::MechanicalParams* /*mparams*/, DataVecDeriv& res, const DataVecDeriv& dx, SReal factor)
{
    const MassVector &masses= d_vertexMass.getValue();
    helper::WriteAccessor< DataVecDeriv > _res = res;
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
            _res[i] += (_dx[i] * masses[i]) * (Real)factor;
        }
    }
}



template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::accFromF(const core::MechanicalParams* /*mparams*/, DataVecDeriv& a, const DataVecDeriv& f)
{

    const MassVector &masses= d_vertexMass.getValue();
    helper::WriteOnlyAccessor< DataVecDeriv > _a = a;
    const VecDeriv& _f = f.getValue();

    for (unsigned int i=0; i<masses.size(); i++)
    {
        _a[i] = _f[i] / masses[i];
    }
}

template <class DataTypes, class MassType>
SReal DiagonalMass<DataTypes, MassType>::getKineticEnergy( const core::MechanicalParams* /*mparams*/, const DataVecDeriv& v ) const
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

template <class DataTypes, class MassType>
SReal DiagonalMass<DataTypes, MassType>::getPotentialEnergy( const core::MechanicalParams* /*mparams*/, const DataVecCoord& x ) const
{

    const MassVector &masses= d_vertexMass.getValue();
    helper::ReadAccessor< DataVecCoord > _x = x;
    SReal e = 0;
    // gravity
    defaulttype::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<masses.size(); i++)
    {
        e -= theGravity*masses[i]*_x[i];
    }
    return e;
}

// does nothing by default, need to be specialized in .cpp
template <class DataTypes, class MassType>
defaulttype::Vector6
DiagonalMass<DataTypes, MassType>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& /*vx*/, const DataVecDeriv& /*vv*/  ) const
{
    return defaulttype::Vector6();
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    const MassVector &masses= d_vertexMass.getValue();
    const int N = defaulttype::DataTypeInfo<Deriv>::size();
    AddMToMatrixFunctor<Deriv,MassType> calc;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    Real mFactor = (Real)mparams->mFactorIncludingRayleighDamping(this->rayleighMass.getValue());
    for (unsigned int i=0; i<masses.size(); i++)
        calc(r.matrix, masses[i], r.offset + N*i, mFactor);
}


template <class DataTypes, class MassType>
SReal DiagonalMass<DataTypes, MassType>::getElementMass(unsigned int index) const
{
    return (SReal)(d_vertexMass.getValue()[index]);
}


//TODO: special case for Rigid Mass
template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const
{
    static const defaulttype::BaseMatrix::Index dimension = (defaulttype::BaseMatrix::Index) defaulttype::DataTypeInfo<Deriv>::size();
    if (m->rowSize() != dimension || m->colSize() != dimension) m->resize(dimension,dimension);

    m->clear();
    AddMToMatrixFunctor<Deriv,MassType>()(m, d_vertexMass.getValue()[index], 0, 1);
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::reinit()
{
    if (m_dataTrackerTotal.hasChanged() || m_dataTrackerDensity.hasChanged() || m_dataTrackerVertex.hasChanged())
    {
        update();
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::initTopologyHandlers()
{
    // add the functions to handle topology changes.
    m_pointHandler = new DMassPointHandler(this, &d_vertexMass);
    d_vertexMass.createTopologicalEngine(_topology, m_pointHandler);
    if (edgeGeo)
        d_vertexMass.linkToEdgeDataArray();
    if (triangleGeo)
        d_vertexMass.linkToTriangleDataArray();
    if (quadGeo)
        d_vertexMass.linkToQuadDataArray();
    if (tetraGeo)
        d_vertexMass.linkToTetrahedronDataArray();
    if (hexaGeo)
        d_vertexMass.linkToHexahedronDataArray();
    d_vertexMass.registerTopologicalData();
}

template <class DataTypes, class MassType>
bool DiagonalMass<DataTypes, MassType>::checkTopology()
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
                msg_error() << "Hexahedron topology found but geometry algorithms are missing. Add the component HexahedronSetGeometryAlgorithms.";
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
                msg_error() << "Tetrahedron topology found but geometry algorithms are missing. Add the component TetrahedronSetGeometryAlgorithms.";
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
                msg_error() << "Quad topology found but geometry algorithms are missing. Add the component QuadSetGeometryAlgorithms.";
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
                msg_error() << "Triangle topology found but geometry algorithms are missing. Add the component TriangleSetGeometryAlgorithms.";
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
                msg_error() << "Edge topology found but geometry algorithms are missing. Add the component EdgeSetGeometryAlgorithms.";
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
void DiagonalMass<DataTypes, MassType>::init()
{
    if (!d_fileMass.getValue().empty())
    {
        if(!load(d_fileMass.getFullPath().c_str())){
            m_componentstate = ComponentState::Invalid;
            return;
        }
        msg_warning() << "File given as input for DiagonalMass, in this a case:" << msgendl
                      << "the topology won't be used to compute the mass" << msgendl
                      << "the update, the coherency and the tracking of mass information data are disable (listening = false)";
        this->f_listening.setValue(false);
        Inherited::init();
    }
    else
    {
        m_dataTrackerVertex.trackData(d_vertexMass);
        m_dataTrackerDensity.trackData(d_massDensity);
        m_dataTrackerTotal.trackData(d_totalMass);

        if(!checkTopology())
        {
            m_componentstate = ComponentState::Invalid;
            return;
        }
        Inherited::init();
        initTopologyHandlers();

        // TODO(dmarchal 2018-11-10): this code is duplicated with the one in RigidImpl we should factor it (remove in 1 year if not done or update the dates)
        if (this->mstate && d_vertexMass.getValue().size() > 0 && d_vertexMass.getValue().size() < (unsigned)this->mstate->getSize())
        {
            MassVector &masses= *d_vertexMass.beginEdit();
            size_t i = masses.size()-1;
            size_t n = (size_t)this->mstate->getSize();
            masses.reserve(n);
            while (masses.size() < n)
                masses.push_back(masses[i]);
            d_vertexMass.endEdit();
        }

        massInitialization();
    }
     m_componentstate = ComponentState::Valid;
}


template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::massInitialization()
{
    //Mass initialization process
    if(d_vertexMass.isSet() || d_massDensity.isSet() || d_totalMass.isSet() )
    {
        //totalMass data is prioritary on vertexMass and massDensity
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


template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::printMass()
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
        average_vertex /= (Real)(vertexM.size());
    }

    msg_info() << "mass information computed :" << msgendl
               << "totalMass   = " << d_totalMass.getValue() << msgendl
               << "massDensity = " << d_massDensity.getValue() << msgendl
               << "mean vertexMass [min,max] = " << average_vertex << " [" << min_vertex << "," <<  max_vertex <<"]";
}


template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::computeMass()
{
    if (_topology && (d_massDensity.getValue() > 0 || d_vertexMass.getValue().size() == 0))
    {
        if (_topology->getNbHexahedra()>0 && hexaGeo)
        {

            MassVector& masses = *d_vertexMass.beginEdit();
            m_topologyType=TOPOLOGY_HEXAHEDRONSET;

            masses.resize(this->mstate->getSize());
            for(unsigned int i=0; i<masses.size(); ++i)
              masses[i]=(Real)0;

            Real md=d_massDensity.getValue();
            Real mass=(Real)0;
            Real total_mass=(Real)0;

            for (Topology::HexahedronID i=0; i<_topology->getNbHexahedra(); ++i)
            {
                const Hexahedron &h=_topology->getHexahedron(i);
                if (hexaGeo)
                {
                    if (d_computeMassOnRest.getValue())
                        mass=(md*hexaGeo->computeRestHexahedronVolume(i))/(Real)8.0;
                    else
                        mass=(md*hexaGeo->computeHexahedronVolume(i))/(Real)8.0;

                    for (unsigned int j = 0 ; j < h.size(); j++)
                    {
                        masses[h[j]] += mass;
                        total_mass += mass;
                    }
                }
            }

            d_totalMass.setValue(total_mass);
            d_vertexMass.endEdit();

        }
        else if (_topology->getNbTetrahedra()>0 && tetraGeo)
        {

            MassVector& masses = *d_vertexMass.beginEdit();
            m_topologyType=TOPOLOGY_TETRAHEDRONSET;

            // resize array
            clear();
            masses.resize(this->mstate->getSize());

            for(unsigned int i=0; i<masses.size(); ++i)
                masses[i]=(Real)0;

            Real md=d_massDensity.getValue();
            Real mass=(Real)0;
            Real total_mass=(Real)0;

            for (Topology::TetrahedronID i=0; i<_topology->getNbTetrahedra(); ++i)
            {
                const Tetrahedron &t=_topology->getTetrahedron(i);
                if(tetraGeo)
                {
                    if (d_computeMassOnRest.getValue())
                        mass=(md*tetraGeo->computeRestTetrahedronVolume(i))/(Real)4.0;
                    else
                        mass=(md*tetraGeo->computeTetrahedronVolume(i))/(Real)4.0;
                }
                for (unsigned int j = 0 ; j < t.size(); j++)
                {
                    masses[t[j]] += mass;
                    total_mass += mass;
                }
            }
            d_totalMass.setValue(total_mass);
            d_vertexMass.endEdit();
        }
        else if (_topology->getNbQuads()>0 && quadGeo) {
            MassVector& masses = *d_vertexMass.beginEdit();
            m_topologyType=TOPOLOGY_QUADSET;

            // resize array
            clear();
            masses.resize(this->mstate->getSize());

            for(unsigned int i=0; i<masses.size(); ++i)
                masses[i]=(Real)0;

            Real md=d_massDensity.getValue();
            Real mass=(Real)0;
            Real total_mass=(Real)0;

            for (Topology::QuadID i=0; i<_topology->getNbQuads(); ++i)
            {
                const Quad &t=_topology->getQuad(i);
                if(quadGeo)
                {
                    if (d_computeMassOnRest.getValue())
                        mass=(md*quadGeo->computeRestQuadArea(i))/(Real)4.0;
                    else
                        mass=(md*quadGeo->computeQuadArea(i))/(Real)4.0;
                }
                for (unsigned int j = 0 ; j < t.size(); j++)
                {
                    masses[t[j]] += mass;
                    total_mass += mass;
                }
            }
            d_totalMass.setValue(total_mass);
            d_vertexMass.endEdit();
        }
        else if (_topology->getNbTriangles()>0 && triangleGeo)
        {
            MassVector& masses = *d_vertexMass.beginEdit();
            m_topologyType=TOPOLOGY_TRIANGLESET;

            // resize array
            clear();
            masses.resize(this->mstate->getSize());

            for(unsigned int i=0; i<masses.size(); ++i)
                masses[i]=(Real)0;

            Real md=d_massDensity.getValue();
            Real mass=(Real)0;
            Real total_mass=(Real)0;

            for (Topology::TriangleID i=0; i<_topology->getNbTriangles(); ++i)
            {
                const Triangle &t=_topology->getTriangle(i);
                if(triangleGeo)
                {
                    if (d_computeMassOnRest.getValue())
                        mass=(md*triangleGeo->computeRestTriangleArea(i))/(Real)3.0;
                    else
                        mass=(md*triangleGeo->computeTriangleArea(i))/(Real)3.0;
                }
                for (unsigned int j = 0 ; j < t.size(); j++)
                {
                    masses[t[j]] += mass;
                    total_mass += mass;
                }
            }
            d_totalMass.setValue(total_mass);
            d_vertexMass.endEdit();
        }
        else if (_topology->getNbEdges()>0 && edgeGeo)
        {

            MassVector& masses = *d_vertexMass.beginEdit();
            m_topologyType=TOPOLOGY_EDGESET;

            // resize array
            clear();
            masses.resize(this->mstate->getSize());

            for(unsigned int i=0; i<masses.size(); ++i)
                masses[i]=(Real)0;

            Real md=d_massDensity.getValue();
            Real mass=(Real)0;
            Real total_mass=(Real)0;

            for (Topology::EdgeID i=0; i<_topology->getNbEdges(); ++i)
            {
                const Edge &e=_topology->getEdge(i);
                if(edgeGeo)
                {
                    if (d_computeMassOnRest.getValue())
                        mass=(md*edgeGeo->computeRestEdgeLength(i))/(Real)2.0;
                    else
                        mass=(md*edgeGeo->computeEdgeLength(i))/(Real)2.0;
                }
                for (unsigned int j = 0 ; j < e.size(); j++)
                {
                    masses[e[j]] += mass;
                    total_mass += mass;
                }
            }
            d_totalMass.setValue(total_mass);
            d_vertexMass.endEdit();
        }
    }
}

template <class DataTypes, class MassType>
bool DiagonalMass<DataTypes, MassType>::checkTotalMass()
{
    //Check for negative or null value, if wrongly set use the default value totalMass = 1.0
    if(d_totalMass.getValue() <= 0.0)
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


template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::checkTotalMassInit()
{
    //Check for negative or null value, if wrongly set use the default value totalMass = 1.0
    if(!checkTotalMass())
    {
        msg_warning() << "Switching back to default values: totalMass = 1.0\n";
        d_totalMass.setValue(1.0) ;
    }
}


template <class DataTypes, class MassType>
bool DiagonalMass<DataTypes, MassType>::checkVertexMass()
{
    const MassVector &vertexMass = d_vertexMass.getValue();

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
void DiagonalMass<DataTypes, MassType>::initFromVertexMass()
{
    msg_info() << "vertexMass information is used";

    const MassVector& vertexMass = d_vertexMass.getValue();
    Real totalMassSave = 0.0;
    for(size_t i=0; i<vertexMass.size(); i++)
    {
        totalMassSave += vertexMass[i];
    }
    //Compute the volume
    setMassDensity(1.0);

    computeMass();

    helper::WriteAccessor<Data<MassVector> > vertexMassWrite = d_vertexMass;
    //Compute volume = mass since massDensity = 1.0
    Real volume = 0.0;
    for(size_t i=0; i<vertexMassWrite.size(); i++)
    {
        volume += vertexMass[i];
        vertexMassWrite[i] = vertexMass[i];
    }
    //Update all computed values
    setMassDensity((Real)totalMassSave/volume);
    d_totalMass.setValue(totalMassSave);
}


template <class DataTypes, class MassType>
bool DiagonalMass<DataTypes, MassType>::checkMassDensity()
{
    const Real &massDensity = d_massDensity.getValue();

    //Check that the massDensity is strictly positive
    if(massDensity <= 0.0)
    {
        msg_warning() << "Negative value of massDensity: massDensity = " << massDensity;
        return false;
    }
    else
    {
        return true;
    }
}


template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::initFromMassDensity()
{
    msg_info() << "massDensity information is used";

    computeMass();

    const MassVector &vertexMass = d_vertexMass.getValue();
    Real sumMass = 0.0;
    for (size_t i=0; i<(size_t)_topology->getNbPoints(); i++)
    {
        sumMass += vertexMass[i];
    }
    d_totalMass.setValue(sumMass);
}


template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::initFromTotalMass()
{
    msg_info() << "totalMass information is used";

    const Real totalMassTemp = d_totalMass.getValue();

    Real sumMass = 0.0;
    setMassDensity(1.0);

    computeMass();

    const MassVector &vertexMass = d_vertexMass.getValue();
    for (size_t i=0; i<(size_t)_topology->getNbPoints(); i++)
    {
        sumMass += vertexMass[i];
    }

    setMassDensity((Real)totalMassTemp/sumMass);

    computeMass();
}


template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::setVertexMass(sofa::helper::vector< Real > vertexMass)
{
    const MassVector currentVertexMass = d_vertexMass.getValue();
    helper::WriteAccessor<Data<MassVector> > vertexMassWrite = d_vertexMass;
    vertexMassWrite.resize(vertexMass.size());
    for(int i=0; i<(int)vertexMass.size(); i++)
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


template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::setMassDensity(Real massDensityValue)
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


template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::setTotalMass(Real totalMass)
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
const typename DiagonalMass<DataTypes, MassType>::Real &DiagonalMass<DataTypes, MassType>::getMassDensity()
{
    return d_massDensity.getValue();
}


template <class DataTypes, class MassType>
const typename DiagonalMass<DataTypes, MassType>::Real &DiagonalMass<DataTypes, MassType>::getTotalMass()
{
    return d_totalMass.getValue();
}


template <class DataTypes, class MassType>
bool DiagonalMass<DataTypes, MassType>::update()
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
        if(checkVertexMass())
        {
            initFromVertexMass();
            update = true;
        }
        m_dataTrackerVertex.clean();
    }

    if(update)
    {
        //Info post-init
        msg_info() << "mass information updated";
        printMass();
    }

    return update;
}


template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if(mparams)
    {
        VecDeriv& v = *d_v.beginEdit();
        // gravity
        sofa::defaulttype::Vec3d g ( this->getContext()->getGravity() );
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2]);
        Deriv hg = theGravity * (typename DataTypes::Real)mparams->dt();

        for (unsigned int i=0; i<v.size(); i++)
        {
            v[i] += hg;
        }
        d_v.endEdit();
    }
}


template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& , const DataVecDeriv& )
{
    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if(this->m_separateGravity.getValue())
        return;

    const MassVector &masses= d_vertexMass.getValue();
    helper::WriteAccessor< DataVecDeriv > _f = f;

    // gravity
    sofa::defaulttype::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);


    // add weight and inertia force
    for (unsigned int i=0; i<masses.size(); i++)
    {
        _f[i] += theGravity*masses[i];
    }
}

template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    const MassVector &masses= d_vertexMass.getValue();
    if (masses.empty())
        return;

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    Coord gravityCenter;
    Real totalMass=0.0;

    std::vector<  sofa::defaulttype::Vector3 > points;

    for (unsigned int i=0; i<x.size(); i++)
    {
        sofa::defaulttype::Vector3 p;
        p = DataTypes::getCPos(x[i]);

        points.push_back(p);
        gravityCenter += x[i]*masses[i];
        totalMass += masses[i];
    }

    if ( d_showCenterOfGravity.getValue() )
    {
        gravityCenter /= totalMass;
        const sofa::defaulttype::Vec4f color(1.0,1.0,0.0,1.0);

        Real axisSize = d_showAxisSize.getValue();
        sofa::defaulttype::Vector3 temp;

        for ( unsigned int i=0 ; i<3 ; i++ )
            if(i < Coord::spatial_dimensions )
                temp[i] = gravityCenter[i];

        vparams->drawTool()->drawCross(temp, axisSize, color);
    }
}

template <class DataTypes, class MassType>
class DiagonalMass<DataTypes, MassType>::Loader : public helper::io::XspLoaderDataHook
{
public:
    DiagonalMass<DataTypes, MassType>* dest;
    Loader(DiagonalMass<DataTypes, MassType>* dest) : dest(dest) {}
    void addMass(SReal /*px*/, SReal /*py*/, SReal /*pz*/, SReal /*vx*/, SReal /*vy*/, SReal /*vz*/, SReal mass, SReal /*elastic*/, bool /*fixed*/, bool /*surface*/) override
    {
        dest->addMass(MassType((Real)mass));
    }
};

template <class DataTypes, class MassType>
bool DiagonalMass<DataTypes, MassType>::load(const char *filename)
{
    clear();
    if (filename!=NULL && filename[0]!='\0')
    {
        Loader loader(this);
        return helper::io::XspLoader::Load(filename, loader);
    }
    return false;
}


template <class DataTypes, class MassType>
void DiagonalMass<DataTypes, MassType>::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::simulation::AnimateEndEvent::checkEventType(event))
    {
        update();
    }
}


} // namespace mass

} // namespace component

} // namespace sofa

#endif
