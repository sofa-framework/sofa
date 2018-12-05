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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_INL
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_INL

#include <SofaBaseMechanics/BarycentricMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/behavior/MechanicalState.h>

#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaBaseTopology/SparseGridTopology.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/QuadSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.h>

#include<SofaBaseMechanics/BarycentricMappers/BarycentricMapperMeshTopology.h>
#include<SofaBaseMechanics/BarycentricMappers/BarycentricMapperRegularGridTopology.h>
#include<SofaBaseMechanics/BarycentricMappers/BarycentricMapperSparseGridTopology.h>
#include<SofaBaseMechanics/BarycentricMappers/BarycentricMapperEdgeSetTopology.h>
#include<SofaBaseMechanics/BarycentricMappers/BarycentricMapperTriangleSetTopology.h>
#include<SofaBaseMechanics/BarycentricMappers/BarycentricMapperQuadSetTopology.h>
#include<SofaBaseMechanics/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.h>
#include<SofaBaseMechanics/BarycentricMappers/BarycentricMapperHexahedronSetTopology.h>

#include <sofa/helper/vector.h>
#include <sofa/helper/system/config.h>

#include <sofa/simulation/Simulation.h>

#include <algorithm>
#include <iostream>

namespace sofa
{

namespace component
{

namespace mapping
{

using sofa::defaulttype::Vector3;
using sofa::defaulttype::Matrix3;
using sofa::defaulttype::Mat3x3d;
using sofa::defaulttype::Vec3d;
// 10/18 E.Coevoet: what's the difference between edge/line, tetra/tetrahedron, hexa/hexahedron?
typedef typename sofa::core::topology::BaseMeshTopology::Line Edge;
typedef typename sofa::core::topology::BaseMeshTopology::Edge Edge;
typedef typename sofa::core::topology::BaseMeshTopology::Triangle Triangle;
typedef typename sofa::core::topology::BaseMeshTopology::Quad Quad;
typedef typename sofa::core::topology::BaseMeshTopology::Tetra Tetra;
typedef typename sofa::core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;
typedef typename sofa::core::topology::BaseMeshTopology::Hexa Hexa;
typedef typename sofa::core::topology::BaseMeshTopology::Hexahedron Hexahedron;
typedef typename sofa::core::topology::BaseMeshTopology::SeqLines SeqLines;
typedef typename sofa::core::topology::BaseMeshTopology::SeqEdges SeqEdges;
typedef typename sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
typedef typename sofa::core::topology::BaseMeshTopology::SeqQuads SeqQuads;
typedef typename sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
typedef typename sofa::core::topology::BaseMeshTopology::SeqHexahedra SeqHexahedra;

template <class TIn, class TOut>
BarycentricMapping<TIn, TOut>::BarycentricMapping(core::State<In>* from, core::State<Out>* to, typename Mapper::SPtr mapper)
    : Inherit1 ( from, to )
    , m_mapper(initLink("mapper","Internal mapper created depending on the type of topology"), mapper)

{
    if (mapper)
        this->addSlave(mapper.get());
}

template <class TIn, class TOut>
BarycentricMapping<TIn, TOut>::BarycentricMapping (core::State<In>* from, core::State<Out>* to, BaseMeshTopology * topology )
    : Inherit1 ( from, to )
    , m_mapper (initLink("mapper","Internal mapper created depending on the type of topology"))
{
    if (topology)
    {
        createMapperFromTopology ( topology );
    }
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::createMapperFromTopology ( BaseMeshTopology * topology )
{
    using sofa::core::behavior::BaseMechanicalState;

    m_mapper = nullptr;

    topology::PointSetTopologyContainer* toTopoCont;
    this->toModel->getContext()->get(toTopoCont);

    core::topology::TopologyContainer* fromTopoCont = nullptr;

    if (dynamic_cast< core::topology::TopologyContainer* >(topology) != nullptr)
    {
        fromTopoCont = dynamic_cast< core::topology::TopologyContainer* >(topology);
    }
    else if (topology == nullptr)
    {
        this->fromModel->getContext()->get(fromTopoCont);
    }

    if (fromTopoCont != nullptr)
    {
        topology::HexahedronSetTopologyContainer* t1 = dynamic_cast< topology::HexahedronSetTopologyContainer* >(fromTopoCont);
        if (t1 != nullptr)
        {
            typedef BarycentricMapperHexahedronSetTopology<InDataTypes, OutDataTypes> HexahedronSetMapper;
            m_mapper = sofa::core::objectmodel::New<HexahedronSetMapper>(t1, toTopoCont);
        }
        else
        {
            topology::TetrahedronSetTopologyContainer* t2 = dynamic_cast<topology::TetrahedronSetTopologyContainer*>(fromTopoCont);
            if (t2 != nullptr)
            {
                typedef BarycentricMapperTetrahedronSetTopology<InDataTypes, OutDataTypes> TetrahedronSetMapper;
                m_mapper = sofa::core::objectmodel::New<TetrahedronSetMapper>(t2, toTopoCont);
            }
            else
            {
                topology::QuadSetTopologyContainer* t3 = dynamic_cast<topology::QuadSetTopologyContainer*>(fromTopoCont);
                if (t3 != nullptr)
                {
                    typedef BarycentricMapperQuadSetTopology<InDataTypes, OutDataTypes> QuadSetMapper;
                    m_mapper = sofa::core::objectmodel::New<QuadSetMapper>(t3, toTopoCont);
                }
                else
                {
                    topology::TriangleSetTopologyContainer* t4 = dynamic_cast<topology::TriangleSetTopologyContainer*>(fromTopoCont);
                    if (t4 != nullptr)
                    {
                        typedef BarycentricMapperTriangleSetTopology<InDataTypes, OutDataTypes> TriangleSetMapper;
                        m_mapper = sofa::core::objectmodel::New<TriangleSetMapper>(t4, toTopoCont);
                    }
                    else
                    {
                        topology::EdgeSetTopologyContainer* t5 = dynamic_cast<topology::EdgeSetTopologyContainer*>(fromTopoCont);
                        if ( t5 != nullptr )
                        {
                            typedef BarycentricMapperEdgeSetTopology<InDataTypes, OutDataTypes> EdgeSetMapper;
                            m_mapper = sofa::core::objectmodel::New<EdgeSetMapper>(t5, toTopoCont);
                        }
                    }
                }
            }
        }
    }
    else
    {
        using sofa::component::topology::RegularGridTopology;

        RegularGridTopology* rgt = dynamic_cast< RegularGridTopology* >(topology);

        if (rgt != nullptr && rgt->isVolume())
        {
            typedef BarycentricMapperRegularGridTopology< InDataTypes, OutDataTypes > RegularGridMapper;

            m_mapper = sofa::core::objectmodel::New<RegularGridMapper>(rgt, toTopoCont);
        }
        else
        {
            using sofa::component::topology::SparseGridTopology;

            SparseGridTopology* sgt = dynamic_cast< SparseGridTopology* >(topology);
            if (sgt != nullptr && sgt->isVolume())
            {
                typedef BarycentricMapperSparseGridTopology< InDataTypes, OutDataTypes > SparseGridMapper;
                m_mapper = sofa::core::objectmodel::New<SparseGridMapper>(sgt, toTopoCont);
            }
            else // generic MeshTopology
            {
                using sofa::core::topology::BaseMeshTopology;

                typedef BarycentricMapperMeshTopology< InDataTypes, OutDataTypes > MeshMapper;
                BaseMeshTopology* bmt = dynamic_cast< BaseMeshTopology* >(topology);
                m_mapper = sofa::core::objectmodel::New<MeshMapper>(bmt, toTopoCont);
            }
        }
    }
    if (m_mapper)
    {
        m_mapper->setName("mapper");
        this->addSlave(m_mapper.get());
        m_mapper->maskFrom = this->maskFrom;
        m_mapper->maskTo = this->maskTo;
    }
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::init()
{
    topology_from = this->fromModel->getContext()->getMeshTopology();
    topology_to = this->toModel->getContext()->getMeshTopology();

    Inherit1::init();

    if ( m_mapper == NULL ) // try to create a mapper according to the topology of the In model
    {
        if ( topology_from!=NULL )
        {
            createMapperFromTopology ( topology_from );
        }
    }

    if ( m_mapper != NULL )
    {
        if (useRestPosition.getValue())
            m_mapper->init ( ((const core::State<Out> *)this->toModel)->read(core::ConstVecCoordId::restPosition())->getValue(), ((const core::State<In> *)this->fromModel)->read(core::ConstVecCoordId::restPosition())->getValue() );
        else
            m_mapper->init (((const core::State<Out> *)this->toModel)->read(core::ConstVecCoordId::position())->getValue(), ((const core::State<In> *)this->fromModel)->read(core::ConstVecCoordId::position())->getValue() );
    }
    else
    {
        msg_error() << "Barycentric mapping does not understand topology.";
    }

}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::reinit()
{
    if ( m_mapper != NULL )
    {
        m_mapper->clear();
        m_mapper->init (((const core::State<Out> *)this->toModel)->read(core::ConstVecCoordId::position())->getValue(), ((const core::State<In> *)this->fromModel)->read(core::ConstVecCoordId::position())->getValue() );
    }
}


template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::apply(const core::MechanicalParams * mparams, Data< typename Out::VecCoord >& out, const Data< typename In::VecCoord >& in)
{
    SOFA_UNUSED(mparams);

    if (m_mapper != NULL)
    {
        m_mapper->resize( this->toModel );
        m_mapper->apply(*out.beginWriteOnly(), in.getValue());
        out.endEdit();
    }
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::applyJ (const core::MechanicalParams * mparams, Data< typename Out::VecDeriv >& _out, const Data< typename In::VecDeriv >& in)
{
    SOFA_UNUSED(mparams);

    typename Out::VecDeriv* out = _out.beginEdit();
    if (m_mapper != NULL)
    {
        m_mapper->applyJ(*out, in.getValue());
    }
    _out.endEdit();
}


template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::applyJT (const core::MechanicalParams * mparams, Data< typename In::VecDeriv >& out, const Data< typename Out::VecDeriv >& in)
{
    SOFA_UNUSED(mparams);

    if (m_mapper != NULL)
    {
        m_mapper->applyJT(*out.beginEdit(), in.getValue());
        out.endEdit();
    }
}


template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* BarycentricMapping<TIn, TOut>::getJ()
{
    if (m_mapper!=NULL )
    {
        const size_t outStateSize = this->toModel->getSize();
        const size_t inStateSize = this->fromModel->getSize();
        const sofa::defaulttype::BaseMatrix* matJ = m_mapper->getJ((int)outStateSize, (int)inStateSize);

        return matJ;
    }
    else
        return nullptr;
}



template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if ( !vparams->displayFlags().getShowMappings() ) return;

    // Draw model (out) points
    const OutVecCoord& out = this->toModel->read(core::ConstVecCoordId::position())->getValue();
    std::vector< Vector3 > points;
    for ( unsigned int i=0; i<out.size(); i++ )
    {
        points.push_back ( OutDataTypes::getCPos(out[i]) );
    }
    vparams->drawTool()->drawPoints ( points, 7, sofa::defaulttype::Vec<4,float> ( 1,1,0,1 ) );

    // Draw mapping line between models
    const InVecCoord& in = this->fromModel->read(core::ConstVecCoordId::position())->getValue();
    if ( m_mapper!=NULL )
        m_mapper->draw(vparams,out,in);

}


template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::applyJT(const core::ConstraintParams * cparams, Data< typename In::MatrixDeriv >& out, const Data< typename Out::MatrixDeriv >& in)
{
    SOFA_UNUSED(cparams);

    if (m_mapper!=NULL )
    {
        m_mapper->applyJT(*out.beginEdit(), in.getValue());
        out.endEdit();
    }
}


template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::handleTopologyChange ( core::topology::Topology* t )
{
    SOFA_UNUSED(t);
    reinit(); // we now recompute the entire mapping when there is a topologychange
}

#ifdef BARYCENTRIC_MAPPER_TOPOCHANGE_REINIT
template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::handleTopologyChange(core::topology::Topology* t)
{
    using core::topology::TopologyChange;

    if (t != this->fromTopology) return;

    if ( this->fromTopology->beginChange() == this->fromTopology->endChange() )
        return;

    MechanicalState< In >* mStateFrom = NULL;
    MechanicalState< Out >* mStateTo = NULL;

    this->fromTopology->getContext()->get(mStateFrom);
    this->toTopology->getContext()->get(mStateTo);

    if ((mStateFrom == NULL) || (mStateTo == NULL))
        return;

    const typename MechanicalState< In >::VecCoord& in = *(mStateFrom->getX0());
    const typename MechanicalState< Out >::VecCoord& out = *(mStateTo->getX0());

	for (std::list< const TopologyChange *>::const_iterator it = this->fromTopology->beginChange(), itEnd = this->fromTopology->endChange(); it != itEnd; ++it)
	{
		const core::topology::TopologyChangeType& changeType = (*it)->getChangeType();

		switch ( changeType )
		{
        case core::topology::ENDING_EVENT :
        {
            const helper::vector< topology::Triangle >& triangles = this->fromTopology->getTriangles();
            helper::vector< Mat3x3d > bases;
            helper::vector< Vector3 > centers;

            // clear and reserve space for 2D mapping
            this->clear(out.size());
            bases.resize(triangles.size());
            centers.resize(triangles.size());

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

            for ( unsigned int i=0; i<out.size(); i++ )
            {
                Vec3d pos = Out::getCPos(out[i]);
                Vector3 coefs;
                int index = -1;
                double distance = 1e10;
                for ( unsigned int t = 0; t < triangles.size(); t++ )
                {
                    Vec3d v = bases[t] * ( pos - in[triangles[t][0]] );
                    double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( ( v[2]<0?-v[2]:v[2] )-0.01,v[0]+v[1]-1 ) );
                    if ( d>0 ) d = ( pos-centers[t] ).norm2();
                    if ( d<distance ) { coefs = v; distance = d; index = t; }
                }

                this->addPointInTriangle ( index, coefs.ptr() );
            }
            break;
        }
		default:
			break;
		}
	}
}

#endif // BARYCENTRIC_MAPPER_TOPOCHANGE_REINIT

template<class TIn, class TOut>
const helper::vector< defaulttype::BaseMatrix*>* BarycentricMapping<TIn, TOut>::getJs()
{
    typedef typename Mapper::MatrixType mat_type;
    const sofa::defaulttype::BaseMatrix* matJ = getJ();

    const mat_type* mat = dynamic_cast<const mat_type*>(matJ);
    assert( mat );

    eigen.copyFrom( *mat );   // woot

    js.resize( 1 );
    js[0] = &eigen;
    return &js;
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::updateForceMask()
{
    if( m_mapper )
        m_mapper->updateForceMask();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
