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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTBARYCENTRICMAPPING_INL
#define SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTBARYCENTRICMAPPING_INL

#include "PersistentContactBarycentricMapping.h"

#include <sofa/component/mapping/BarycentricMapping.inl>

#include <sofa/simulation/common/AnimateEndEvent.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class In, class Out>
int PersistentContactBarycentricMapperMeshTopology<In,Out>::addContactPointFromInputMapping(const InVecDeriv& in, const sofa::defaulttype::Vector3& _pos, std::vector< std::pair<int, double> > & /*baryCoords*/)
{
    this->updateJ = true;
    int retValue = 0;

    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = this->fromTopology->getTetrahedra();
#ifdef SOFA_NEW_HEXA
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& cubes = this->fromTopology->getHexahedra();
#else
    const sofa::core::topology::BaseMeshTopology::SeqCubes& cubes = this->fromTopology->getCubes();
#endif
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& triangles = this->fromTopology->getTriangles();
    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = this->fromTopology->getQuads();

    sofa::helper::vector<Matrix3> bases;
    sofa::helper::vector<Vector3> centers;

    if ( tetrahedra.empty() && cubes.empty() )
    {
        if ( triangles.empty() && quads.empty() )
        {
            //no 3D elements, nor 2D elements -> map on 1D elements

            const sofa::core::topology::BaseMeshTopology::SeqEdges& edges = this->fromTopology->getEdges();
            if ( edges.empty() )
                return retValue;

            sofa::helper::vector< SReal >   lengthEdges;
            sofa::helper::vector< Vector3 > unitaryVectors;

            unsigned int e;
            for ( e=0; e<edges.size(); e++ )
            {
                lengthEdges.push_back ( ( in[edges[e][1]]-in[edges[e][0]] ).norm() );

                Vector3 V12 = ( in[edges[e][1]]-in[edges[e][0]] );
                V12.normalize();
                unitaryVectors.push_back ( V12 );
            }

            SReal coef=0;
            for ( e=0; e<edges.size(); e++ )
            {
                SReal lengthEdge = lengthEdges[e];
                Vector3 V12 = unitaryVectors[e];

                coef = ( V12 ) *Vector3 ( _pos - in[edges[e][0]] ) / lengthEdge;
                if ( coef >= 0 && coef <= 1 )
                {
                    retValue = this->addPointInLine ( e,&coef );
                    break;
                }
            }
            //If no good coefficient has been found, we add to the last element
            if ( e == edges.size() )
                retValue = this->addPointInLine ( edges.size()-1,&coef );
        }
        else
        {
            // no 3D elements -> map on 2D elements
            int c0 = triangles.size();
            bases.resize ( triangles.size() + quads.size() );
            centers.resize ( triangles.size() + quads.size() );

            for ( unsigned int t = 0; t < triangles.size(); t++ )
            {
                Mat3x3d m,mt;
                m[0] = in[triangles[t][1]]-in[triangles[t][0]];
                m[1] = in[triangles[t][2]]-in[triangles[t][0]];
                m[2] = cross ( m[0],m[1] );
                mt.transpose ( m );
                bases[t].invert ( mt );
                centers[t] = ( in[triangles[t][0]]+in[triangles[t][1]]+in[triangles[t][2]] ) /3;
            }

            for ( unsigned int c = 0; c < quads.size(); c++ )
            {
                Mat3x3d m,mt;
                m[0] = in[quads[c][1]]-in[quads[c][0]];
                m[1] = in[quads[c][3]]-in[quads[c][0]];
                m[2] = cross ( m[0],m[1] );
                mt.transpose ( m );
                bases[c0+c].invert ( mt );
                centers[c0+c] = ( in[quads[c][0]]+in[quads[c][1]]+in[quads[c][2]]+in[quads[c][3]] ) *0.25;
            }

            Vector3 coefs;
            int index = -1;
            double distance = 1e10;

            for ( unsigned int t = 0; t < triangles.size(); t++ )
            {
                Vec3d v = bases[t] * ( _pos - in[triangles[t][0]] );
                double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( ( v[2]<0?-v[2]:v[2] )-0.01,v[0]+v[1]-1 ) );
                if ( d>0 ) d = ( _pos-centers[t] ).norm2();
                if ( d<distance ) { coefs = v; distance = d; index = t; }
            }

            for ( unsigned int c = 0; c < quads.size(); c++ )
            {
                Vec3d v = bases[c0+c] * ( _pos - in[quads[c][0]] );
                double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( v[1]-1,v[0]-1 ),std::max ( v[2]-0.01,-v[2]-0.01 ) ) );
                if ( d>0 ) d = ( _pos-centers[c0+c] ).norm2();
                if ( d<distance ) { coefs = v; distance = d; index = c0+c; }
            }

            if ( index < c0 )
                retValue = this->addPointInTriangle ( index, coefs.ptr() );
            else
                retValue = this->addPointInQuad ( index-c0, coefs.ptr() );
        }
    }
    else
    {
        int c0 = tetrahedra.size();
        bases.resize ( tetrahedra.size() +cubes.size() );
        centers.resize ( tetrahedra.size() +cubes.size() );

        for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
        {
            Mat3x3d m,mt;
            m[0] = in[tetrahedra[t][1]]-in[tetrahedra[t][0]];
            m[1] = in[tetrahedra[t][2]]-in[tetrahedra[t][0]];
            m[2] = in[tetrahedra[t][3]]-in[tetrahedra[t][0]];
            mt.transpose ( m );
            bases[t].invert ( mt );
            centers[t] = ( in[tetrahedra[t][0]]+in[tetrahedra[t][1]]+in[tetrahedra[t][2]]+in[tetrahedra[t][3]] ) *0.25;
        }

        for ( unsigned int c = 0; c < cubes.size(); c++ )
        {
            Mat3x3d m,mt;
            m[0] = in[cubes[c][1]]-in[cubes[c][0]];
#ifdef SOFA_NEW_HEXA
            m[1] = in[cubes[c][3]]-in[cubes[c][0]];
#else
            m[1] = in[cubes[c][2]]-in[cubes[c][0]];
#endif
            m[2] = in[cubes[c][4]]-in[cubes[c][0]];
            mt.transpose ( m );
            bases[c0+c].invert ( mt );
            centers[c0+c] = ( in[cubes[c][0]]+in[cubes[c][1]]+in[cubes[c][2]]+in[cubes[c][3]]+in[cubes[c][4]]+in[cubes[c][5]]+in[cubes[c][6]]+in[cubes[c][7]] ) *0.125;
        }

        Vector3 coefs;
        int index = -1;
        double distance = 1e10;

        for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
        {
            Vector3 v = bases[t] * ( _pos - in[tetrahedra[t][0]] );
            double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( -v[2],v[0]+v[1]+v[2]-1 ) );
            if ( d>0 )
                d = ( _pos-centers[t] ).norm2();
            if ( d<distance )
            {
                coefs = v;
                distance = d;
                index = t;
            }
        }

        for ( unsigned int c = 0; c < cubes.size(); c++ )
        {
            Vector3 v = bases[c0+c] * ( _pos - in[cubes[c][0]] );
            double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( -v[2],v[0]-1 ),std::max ( v[1]-1,v[2]-1 ) ) );
            if ( d>0 )
                d = ( _pos-centers[c0+c] ).norm2();
            if ( d<distance )
            {
                coefs = v;
                distance = d;
                index = c0+c;
            }
        }

        if ( index < c0 )
            retValue = this->addPointInTetra ( index, coefs.ptr() );
        else
            retValue = this->addPointInCube ( index-c0, coefs.ptr() );
    }

    return retValue;
}


template <class In, class Out>
int PersistentContactBarycentricMapperSparseGridTopology<In,Out>::addContactPointFromInputMapping(const InVecDeriv& in, const sofa::defaulttype::Vector3& pos, std::vector< std::pair<int, double> > & /*baryCoords*/)
{
    this->updateJ = true;

#ifdef SOFA_NEW_HEXA
    const sofa::core::topology::BaseMeshTopology::SeqHexahedra& cubes = this->fromTopology->getHexahedra();
#else
    const sofa::core::topology::BaseMeshTopology::SeqCubes& cubes = this->fromTopology->getCubes();
#endif

    sofa::helper::vector<Matrix3> bases;
    sofa::helper::vector<Vector3> centers;

    bases.resize ( cubes.size() );
    centers.resize ( cubes.size() );

    for ( unsigned int c = 0; c < cubes.size(); c++ )
    {
        Mat3x3d m,mt;
        m[0] = in[cubes[c][1]]-in[cubes[c][0]];
#ifdef SOFA_NEW_HEXA
        m[1] = in[cubes[c][3]]-in[cubes[c][0]];
#else
        m[1] = in[cubes[c][2]]-in[cubes[c][0]];
#endif
        m[2] = in[cubes[c][4]]-in[cubes[c][0]];
        mt.transpose ( m );
        bases[c].invert ( mt );
        centers[c] = ( in[cubes[c][0]]+in[cubes[c][1]]+in[cubes[c][2]]+in[cubes[c][3]]+in[cubes[c][4]]+in[cubes[c][5]]+in[cubes[c][6]]+in[cubes[c][7]] ) *0.125;
    }

    Vector3 coefs;
    int index = -1;
    double distance = 1e10;

    for ( unsigned int c = 0; c < cubes.size(); c++ )
    {
        Vector3 v = bases[c] * ( pos - in[cubes[c][0]] );
        double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( -v[2],v[0]-1 ),std::max ( v[1]-1,v[2]-1 ) ) );
        if (d > 0)
            d = (pos - centers[c]).norm2();
        if (d < distance)
        {
            coefs = v;
            distance = d;
            index = c;
        }
    }

    return this->addPointInCube(index, coefs.ptr());
}


template <class In, class Out>
int PersistentContactBarycentricMapperSparseGridTopology<In,Out>::keepContactPointFromInputMapping(const int index)
{
    if ((const unsigned int)index < m_storedMap.size())
    {
        CubeData &c_data = m_storedMap[index];
        return this->addPointInCube(c_data.in_index, c_data.baryCoords);
    }

    serr << "Warning! PersistentContactBarycentricMapperSparseGridTopology keepContactPointFromInputMapping method refers to an unstored index" << sendl;
    return 0;
}


template <class In, class Out>
void PersistentContactBarycentricMapperSparseGridTopology<In,Out>::storeBarycentricData()
{
    m_storedMap = this->map;
}


template <class In, class Out>
int PersistentContactBarycentricMapperTetrahedronSetTopology<In,Out>::addContactPointFromInputMapping(const InVecDeriv& in, const sofa::defaulttype::Vector3& pos
        , std::vector< std::pair<int, double> > & /*baryCoords*/)
{
    const sofa::helper::vector<topology::Tetrahedron>& tetrahedra = this->fromTopology->getTetrahedra();

    sofa::helper::vector<Matrix3> bases;
    sofa::helper::vector<Vector3> centers;

    bases.resize ( tetrahedra.size() );
    centers.resize ( tetrahedra.size() );
    for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
    {
        Mat3x3d m,mt;
        m[0] = in[tetrahedra[t][1]]-in[tetrahedra[t][0]];
        m[1] = in[tetrahedra[t][2]]-in[tetrahedra[t][0]];
        m[2] = in[tetrahedra[t][3]]-in[tetrahedra[t][0]];
        mt.transpose ( m );
        bases[t].invert ( mt );
        centers[t] = ( in[tetrahedra[t][0]]+in[tetrahedra[t][1]]+in[tetrahedra[t][2]]+in[tetrahedra[t][3]] ) *0.25;
    }

    Vector3 coefs;
    int index = -1;
    double distance = 1e10;
    for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
    {
        Vec3d v = bases[t] * ( pos - in[tetrahedra[t][0]] );
        double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( -v[2],v[0]+v[1]+v[2]-1 ) );

        if (d > 0)
            d = (pos - centers[t]).norm2();

        if (d < distance)
        {
            coefs = v;
            distance = d;
            index = t;
        }
    }

    this->addPointInTetra (index, coefs.ptr() );

    return this->map.getValue().size() - 1;
}



template <class TIn, class TOut>
void PersistentContactBarycentricMapping<TIn, TOut>::beginAddContactPoint()
{
    if (!m_init)
    {
        if (this->mapper)
        {
            this->mapper->clear(0);
        }

        this->toModel->resize(0);

        m_init = true;
    }
}


template <class TIn, class TOut>
int PersistentContactBarycentricMapping<TIn, TOut>::addContactPointFromInputMapping(const sofa::defaulttype::Vector3& pos, std::vector< std::pair<int, double> > & baryCoords)
{
    if (m_persistentMapper)
    {
        const InVecCoord& xfrom = *this->fromModel->getX();

        int index = m_persistentMapper->addContactPointFromInputMapping(xfrom, pos, baryCoords);
        this->toModel->resize(index+1);
        return index;
    }

    return 0;
}


template <class TIn, class TOut>
int PersistentContactBarycentricMapping<TIn, TOut>::keepContactPointFromInputMapping(const int prevIndex)
{
    if (m_persistentMapper)
    {
        int index = m_persistentMapper->keepContactPointFromInputMapping(prevIndex);
        this->toModel->resize(index+1);
        return index;
    }

    serr << "PersistentContactBarycentricMapping::keepContactPointFromInputMapping : no persistent mapper found" << sendl;
    return 0;
}


template <class TIn, class TOut>
void PersistentContactBarycentricMapping<TIn, TOut>::init()
{
    this->f_listening.setValue(true);
    m_init = false;

    BaseMeshTopology *topo_from = this->fromModel->getContext()->getMeshTopology();

    if (this->mapper == NULL)
    {
        if (topo_from != NULL)
        {
            createPersistentMapperFromTopology(topo_from);
        }
    }

    Inherit::init();
}


template <class TIn, class TOut>
void PersistentContactBarycentricMapping<TIn, TOut>::createPersistentMapperFromTopology(BaseMeshTopology *topology)
{
    using sofa::core::behavior::BaseMechanicalState;

    m_persistentMapper = NULL;

    topology::PointSetTopologyContainer* toTopoCont;
    this->toModel->getContext()->get(toTopoCont);

    core::topology::TopologyContainer* fromTopoCont;
    this->fromModel->getContext()->get(fromTopoCont);

    BaseMechanicalState *dofFrom = static_cast< simulation::Node* >(this->fromModel->getContext())->mechanicalState;
    BaseMechanicalState *dofTo = static_cast< simulation::Node* >(this->toModel->getContext())->mechanicalState;

    helper::ParticleMask *maskFrom = &dofFrom->forceMask;
    helper::ParticleMask *maskTo = NULL;

    if (dofTo)
        maskTo = &dofTo->forceMask;

    if (fromTopoCont != NULL)
    {
        topology::TetrahedronSetTopologyContainer* t1 = dynamic_cast< topology::TetrahedronSetTopologyContainer* >(fromTopoCont);
        if (t1 != NULL)
        {
            typedef PersistentContactBarycentricMapperTetrahedronSetTopology<InDataTypes, OutDataTypes> TetrahedronSetMapper;
            m_persistentMapper = sofa::core::objectmodel::New< TetrahedronSetMapper >(t1, toTopoCont, maskFrom, maskTo);
        }
    }
    else
    {
        using sofa::component::topology::SparseGridTopology;

        SparseGridTopology* sgt = dynamic_cast< SparseGridTopology* >(topology);
        if (sgt != NULL && sgt->isVolume())
        {
            typedef PersistentContactBarycentricMapperSparseGridTopology< InDataTypes, OutDataTypes > SparseGridMapper;
            m_persistentMapper = sofa::core::objectmodel::New< SparseGridMapper >(sgt, toTopoCont, maskFrom, maskTo);
        }
        else
        {
            using sofa::core::topology::BaseMeshTopology;

            typedef PersistentContactBarycentricMapperMeshTopology< InDataTypes, OutDataTypes > MeshMapper;
            BaseMeshTopology* bmt = dynamic_cast< BaseMeshTopology* >(topology);
            m_persistentMapper = sofa::core::objectmodel::New< MeshMapper >(bmt, toTopoCont, maskFrom, maskTo);
        }
    }

    if (m_persistentMapper)
    {
        m_persistentMapper->setName("mapper");
        this->addSlave(m_persistentMapper.get());

        TopologyBarycentricMapper<InDataTypes, OutDataTypes> *tmp = dynamic_cast< TopologyBarycentricMapper<InDataTypes, OutDataTypes> * >(m_persistentMapper.get());
        this->mapper = tmp;
        this->addSlave(this->mapper.get());
    }
    else
    {
        serr << "PersistentContactBarycentricMapping not yet compatible with its input topology" << sendl;
    }
}


template <class TIn, class TOut>
void PersistentContactBarycentricMapping<TIn, TOut>::applyPositionAndFreePosition()
{
    core::Mapping<TIn, TOut>::apply(0, sofa::core::VecCoordId::position(), sofa::core::ConstVecCoordId::position());
    core::Mapping<TIn, TOut>::apply(0, sofa::core::VecCoordId::freePosition(), sofa::core::ConstVecCoordId::freePosition());
    core::Mapping<TIn, TOut>::applyJ(0, sofa::core::VecDerivId::velocity(), sofa::core::ConstVecDerivId::velocity());
    core::Mapping<TIn, TOut>::applyJ(0, sofa::core::VecDerivId::freeVelocity(), sofa::core::ConstVecDerivId::freeVelocity());
}


template <class TIn, class TOut>
void PersistentContactBarycentricMapping<TIn, TOut>::storeBarycentricData()
{
    if (m_persistentMapper)
    {
        m_persistentMapper->storeBarycentricData();
    }
}


template <class TIn, class TOut>
void PersistentContactBarycentricMapping<TIn, TOut>::handleEvent(sofa::core::objectmodel::Event* ev)
{
    if (dynamic_cast< simulation::AnimateEndEvent* >(ev))
    {
        storeBarycentricData();

        m_init = false;
    }
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTBARYCENTRICMAPPING_INL
