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
#include <sofa/component/mapping/linear/BarycentricMapping.h>

#include <sofa/component/topology/container/grid/RegularGridTopology.h>
#include <sofa/component/topology/container/grid/SparseGridTopology.h>

#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperMeshTopology.h>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperRegularGridTopology.h>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperSparseGridTopology.h>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperEdgeSetTopology.h>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperTriangleSetTopology.h>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperQuadSetTopology.h>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.h>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperHexahedronSetTopology.h>

#include <sofa/linearalgebra/EigenSparseMatrix.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/type/vector.h>
#include <sofa/simulation/Simulation.h>

namespace sofa::component::mapping::linear
{

using namespace topology;
using sofa::type::Vec3;
using sofa::type::Matrix3;
using sofa::core::objectmodel::ComponentState;
using sofa::linearalgebra::EigenSparseMatrix;

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

template <typename T, typename V>
static bool is_a(const V * topology) {
    return dynamic_cast<const T *>(topology) != nullptr;
}

template <class TIn, class TOut>
BarycentricMapping<TIn, TOut>::BarycentricMapping(core::State<In>* from, core::State<Out>* to, typename Mapper::SPtr mapper)
    : Inherit1 ( from, to )
    , d_useRestPosition(core::objectmodel::Base::initData(&d_useRestPosition, false, "useRestPosition", "Use the rest position of the input and output models to initialize the mapping"))
    , d_mapper(initLink("mapper","Internal mapper created depending on the type of topology"), mapper)
    , d_input_topology(initLink("input_topology", "Input topology container (usually the surrounding domain)."))
    , d_output_topology(initLink("output_topology", "Output topology container (usually the immersed domain)."))


{
    if (mapper)
        this->addSlave(mapper.get());
    internalMatrix = new EigenSparseMatrix<InDataTypes, OutDataTypes>;
}

template <class TIn, class TOut>
BarycentricMapping<TIn, TOut>::BarycentricMapping (core::State<In>* from, core::State<Out>* to, BaseMeshTopology * input_topology )
    : Inherit1 ( from, to )
    , d_useRestPosition(core::objectmodel::Base::initData(&d_useRestPosition, false, "useRestPosition", "Use the rest position of the input and output models to initialize the mapping"))
    , d_mapper (initLink("mapper","Internal mapper created depending on the type of topology"))
    , d_input_topology(initLink("input_topology", "Input topology container (usually the surrounding domain)."))
    , d_output_topology(initLink("output_topology", "Output topology container (usually the immersed domain)."))
{
    if (input_topology) {
        d_input_topology.set(input_topology);
        populateTopologies();
        createMapperFromTopology ();
    }
    internalMatrix = new EigenSparseMatrix<InDataTypes, OutDataTypes>;
}

template <class TIn, class TOut>
BarycentricMapping<TIn, TOut>::~BarycentricMapping()
{
    delete internalMatrix;
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::populateTopologies()
{
    if (! d_input_topology.get()) {
        if (! this->fromModel) {
            msg_error() << "No input mechanical state found. Consider setting the '" << this->fromModel.getName() << "' attribute.";
        } else {
            BaseMeshTopology * topology = nullptr;
            this->fromModel->getContext()->get(topology);
            if (topology) {
                d_input_topology.set(topology);
            } else {
                msg_error() << "No input topology found. Consider setting the '" << d_input_topology.getName() << "' attribute. ";
            }
        }
    }

    if (! d_output_topology.get()) {
        if (! this->toModel) {
            msg_error() << "No output mechanical state found. Consider setting the '" << this->toModel.getName() << "' attribute.";
        } else {
            BaseMeshTopology * topology = nullptr;
            this->toModel->getContext()->get(topology);
            if (topology) {
                d_output_topology.set(topology);
            }
        }
    }
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::createMapperFromTopology ()
{
    using sofa::core::behavior::BaseMechanicalState;
    using sofa::core::topology::TopologyContainer;
    using sofa::component::topology::container::grid::SparseGridTopology;

    using RegularGridMapper = BarycentricMapperRegularGridTopology< InDataTypes, OutDataTypes >;
    using SparseGridMapper =  BarycentricMapperSparseGridTopology< InDataTypes, OutDataTypes >;
    using MeshMapper = BarycentricMapperMeshTopology< InDataTypes, OutDataTypes >;
    using HexahedronSetMapper =  BarycentricMapperHexahedronSetTopology<InDataTypes, OutDataTypes>;
    using TetrahedronSetMapper = BarycentricMapperTetrahedronSetTopology<InDataTypes, OutDataTypes>;
    using QuadSetMapper = BarycentricMapperQuadSetTopology<InDataTypes, OutDataTypes>;
    using TriangleSetMapper = BarycentricMapperTriangleSetTopology<InDataTypes, OutDataTypes>;
    using EdgeSetMapper = BarycentricMapperEdgeSetTopology<InDataTypes, OutDataTypes>;

    auto input_topology_container = d_input_topology.get();
    auto output_topology_container = d_output_topology.get();

    if (!input_topology_container)
        return;

    // Output topology container could be null, as most of the mappers do not need it.

    d_mapper = nullptr;
    RegularGridTopology* rgt = nullptr;
    SparseGridTopology* sgt = nullptr;

    // Regular Grid Topology
    if (is_a<RegularGridTopology>(input_topology_container) ) {
        rgt = dynamic_cast<RegularGridTopology*>(input_topology_container);
        if (rgt->hasVolume()) {
            msg_info() << "Creating RegularGridMapper";
            d_mapper = sofa::core::objectmodel::New<RegularGridMapper>(rgt, output_topology_container);
        }
        goto end;
    }

    // Sparse Grid Topology
    if (is_a<SparseGridTopology>(input_topology_container) ) {
        sgt = dynamic_cast<SparseGridTopology*>(input_topology_container);
        if (sgt->hasVolume()) {
            msg_info() << "Creating SparseGridMapper";
            d_mapper = sofa::core::objectmodel::New<SparseGridMapper>(sgt, output_topology_container);
        }
        goto end;
    }

    //TopologyContainer topologies
    if(is_a<sofa::core::topology::TopologyContainer>(input_topology_container))
    {
        auto topoContainer = dynamic_cast<sofa::core::topology::TopologyContainer*>(input_topology_container);
        // Hexahedron Topology
        if (input_topology_container->getNbHexahedra() > 0) {
            msg_info() << "Creating HexahedronSetMapper";
            d_mapper = sofa::core::objectmodel::New<HexahedronSetMapper>(topoContainer, output_topology_container);
            goto end;
        }

        // Tetrahedron Topology
        if (input_topology_container->getNbTetrahedra() > 0) {
            msg_info() << "Creating TetrahedronSetMapper";
            d_mapper = sofa::core::objectmodel::New<TetrahedronSetMapper >(topoContainer, output_topology_container);
            goto end;
        }

        // Quad Topology
        if (input_topology_container->getNbQuads() > 0) {
            msg_info() << "Creating QuadSetMapper";
            d_mapper = sofa::core::objectmodel::New<QuadSetMapper >(topoContainer, output_topology_container);
            goto end;
        }

        // Triangle Topology
        if (input_topology_container->getNbTriangles() > 0) {
            msg_info() << "Creating TriangleSetMapper";
            d_mapper = sofa::core::objectmodel::New<TriangleSetMapper >(topoContainer, output_topology_container);
            goto end;
        }

        // Edge Topology
        if (input_topology_container->getNbEdges() > 0) {
            msg_info() << "Creating EdgeSetMapper";
            d_mapper = sofa::core::objectmodel::New<EdgeSetMapper >(topoContainer, output_topology_container);
            goto end;
        }
    }

    // Generic Mesh Topology
    d_mapper = sofa::core::objectmodel::New<MeshMapper>(input_topology_container, output_topology_container);

end:
    if (d_mapper) 
    {
        this->addSlave(d_mapper.get());
    }
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::init()
{
    this->d_componentState.setValue(ComponentState::Invalid) ;

    Inherit1::init();

    populateTopologies();

    if ( d_mapper == nullptr ) // try to create a mapper according to the topology of the In model
    {
        createMapperFromTopology ( );
    }

    if ( d_mapper == nullptr)
    {
        if (d_input_topology.get()) {
            msg_error() << "No compatible input topology found. Make sure the input topology ('" << d_input_topology.getPath()
                        << "') is a class derived from BaseMeshTopology.";
        }

        return;
    }

    if (!this->toModel)
        return;

    initMapper();

    this->d_componentState.setValue(ComponentState::Valid) ;
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::reinit()
{
    if ( d_mapper != nullptr )
    {
        d_mapper->clear();
        initMapper();
    }
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::initMapper()
{
    if (d_mapper != nullptr && this->toModel != nullptr && this->fromModel != nullptr)
    {
        if (d_useRestPosition.getValue())
            d_mapper->init (((const core::State<Out> *)this->toModel)->read(core::ConstVecCoordId::restPosition())->getValue(), ((const core::State<In> *)this->fromModel)->read(core::ConstVecCoordId::restPosition())->getValue() );
        else
            d_mapper->init (((const core::State<Out> *)this->toModel)->read(core::ConstVecCoordId::position())->getValue(), ((const core::State<In> *)this->fromModel)->read(core::ConstVecCoordId::position())->getValue() );
    }
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::apply(const core::MechanicalParams * mparams, Data< typename Out::VecCoord >& out, const Data< typename In::VecCoord >& in)
{
    SOFA_UNUSED(mparams);

    if (d_mapper != nullptr)
    {
        d_mapper->resize( this->toModel );
        d_mapper->apply(*out.beginWriteOnly(), in.getValue());
        out.endEdit();
    }
}

template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::applyJ (const core::MechanicalParams * mparams, Data< typename Out::VecDeriv >& _out, const Data< typename In::VecDeriv >& in)
{
    SOFA_UNUSED(mparams);

    if (d_mapper != nullptr)
    {
        auto outWriteAccessor = sofa::helper::getWriteAccessor(_out);
        d_mapper->applyJ(outWriteAccessor.wref(), in.getValue());
    }
}


template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::applyJT (const core::MechanicalParams * mparams, Data< typename In::VecDeriv >& out, const Data< typename Out::VecDeriv >& in)
{
    SOFA_UNUSED(mparams);

    if (d_mapper != nullptr)
    {
        auto outWriteAccessor = sofa::helper::getWriteAccessor(out);
        d_mapper->applyJT(outWriteAccessor.wref(), in.getValue());
    }
}


template <class TIn, class TOut>
const sofa::linearalgebra::BaseMatrix* BarycentricMapping<TIn, TOut>::getJ()
{
    if (d_mapper!=nullptr )
    {
        const size_t outStateSize = this->toModel->getSize();
        const size_t inStateSize = this->fromModel->getSize();
        const sofa::linearalgebra::BaseMatrix* matJ = d_mapper->getJ((int)outStateSize, (int)inStateSize);

        return matJ;
    }
    else
        return nullptr;
}



template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if ( !vparams->displayFlags().getShowMappings() ) return;
    if (this->d_componentState.getValue() != ComponentState::Valid ) return;

    // Draw model (out) points
    const OutVecCoord& out = this->toModel->read(core::ConstVecCoordId::position())->getValue();
    std::vector< Vec3 > points;
    for ( unsigned int i=0; i<out.size(); i++ )
    {
        points.push_back ( OutDataTypes::getCPos(out[i]) );
    }
    vparams->drawTool()->drawPoints ( points, 7, sofa::type::RGBAColor::yellow());

    // Draw mapping line between models
    const InVecCoord& in = this->fromModel->read(core::ConstVecCoordId::position())->getValue();
    if ( d_mapper!=nullptr )
        d_mapper->draw(vparams,out,in);

}


template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::applyJT(const core::ConstraintParams * cparams, Data< typename In::MatrixDeriv >& out, const Data< typename Out::MatrixDeriv >& in)
{
    SOFA_UNUSED(cparams);

    if (d_mapper!=nullptr )
    {
        auto outWriteAccessor = sofa::helper::getWriteAccessor(out);
        d_mapper->applyJT(outWriteAccessor.wref(), in.getValue());
    }
}


template <class TIn, class TOut>
void BarycentricMapping<TIn, TOut>::handleTopologyChange ( core::topology::Topology* t )
{
    //foward topological modifications to the mapper
    if (this->d_mapper.get()){
        this->d_mapper->processTopologicalChanges(((const core::State<Out> *)this->toModel)->read(core::ConstVecCoordId::position())->getValue(),
                                                  ((const core::State<In> *)this->fromModel)->read(core::ConstVecCoordId::position())->getValue(),
                                                  t);
    }
}

#ifdef BARYCENTRIC_MAPPER_TOPOCHANGE_REINIT
template <class In, class Out>
void BarycentricMapperTriangleSetTopology<In,Out>::handleTopologyChange(core::topology::Topology* t)
{
    using core::topology::TopologyChange;

    auto input_topology = d_input_topology.get();
    auto output_topology = d_output_topology.get();

    if (t != input_topology) return;

    if ( input_topology->beginChange() == input_topology->endChange() )
        return;

    MechanicalState< In >* mStateFrom = nullptr;
    MechanicalState< Out >* mStateTo = nullptr;

    input_topology->getContext()->get(mStateFrom);
    output_topology->getContext()->get(mStateTo);

    if ((mStateFrom == nullptr) || (mStateTo == nullptr))
        return;

    const typename MechanicalState< In >::VecCoord& in = *(mStateFrom->getX0());
    const typename MechanicalState< Out >::VecCoord& out = *(mStateTo->getX0());

    for (std::list< const TopologyChange *>::const_iterator it = input_topology->beginChange(), itEnd = input_topology->endChange(); it != itEnd; ++it)
    {
        const core::topology::TopologyChangeType& changeType = (*it)->getChangeType();

        switch ( changeType )
        {
        case core::topology::ENDING_EVENT :
        {
            const type::vector< topology::Triangle >& triangles = input_topology->getTriangles();
            type::vector< Mat3x3d > bases;
            type::vector< Vec3 > centers;

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
                Vec3 coefs;
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
const type::vector< linearalgebra::BaseMatrix*>* BarycentricMapping<TIn, TOut>::getJs()
{
    typedef typename Mapper::MatrixType mat_type;
    const sofa::linearalgebra::BaseMatrix* matJ = getJ();

    const auto * mat = dynamic_cast<const mat_type*>(matJ);
    if(mat==nullptr)
        throw std::runtime_error("Unable to downcast the matrix");

    static_cast<EigenSparseMatrix<InDataTypes, OutDataTypes>*>(internalMatrix)->copyFrom(*mat);

    js.resize( 1 );
    js[0] = internalMatrix;
    return &js;
}

} // namespace sofa::component::mapping::linear
