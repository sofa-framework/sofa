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
#ifndef SOFA_COMPONENT_MAPPING_RelativeRigidMapping_INL
#define SOFA_COMPONENT_MAPPING_RelativeRigidMapping_INL

#include "../deformationMapping/RelativeRigidMapping.h"
#include <sofa/core/visual/VisualParams.h>
#include <iostream>

#include <sofa/core/Mapping.inl>

using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;


template <class TIn, class TOut>
RelativeRigidMapping<TIn, TOut>::RelativeRigidMapping()
    : Inherit(),
      edges( initData(&edges, "edges", "edges: (p, c) -> inv(p) * c"))

{
}

template <class TIn, class TOut>
RelativeRigidMapping<TIn, TOut>::~RelativeRigidMapping()
{
}


template <class TIn, class TOut>
void RelativeRigidMapping<TIn, TOut>::init()
{
    // TODO wrap this somehow ?
    baseMatrices.resize( 1 );
    baseMatrices[0] = &jacobian;

    this->Inherit::init();  // applies the mapping, so after the Data init
}


template <class TIn, class TOut>
void RelativeRigidMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteAccessor< Data<OutVecCoord> >  out = dOut;
    helper::ReadAccessor< Data<InVecCoord> >  in = dIn;
    assert( Nout == Nin );

    // TODO is this needed on each apply ?
    this->jacobian.resizeBlocks(out.size(), in.size());

    se3_type se3;

    const edges_type& edges = this->edges.getValue();
    assert(out.size() >= edges.size() );


    // for each edge in the topology
    index_type off = 0;
    for(typename edges_type::const_iterator e = edges.begin(), end = edges.end();
        e != end; ++e, ++off)
    {

        // parent/child indices
        index_type p = e->data()[0];
        index_type c = e->data()[1];

        assert( p != c );

        // mapping output
        out[ off ] = se3.prod( se3.inv(in[ p ]), in[ c ] );

        // ordered matrix chunks
        typedef std::map<index_type, mat66> chunk_type;
        chunk_type chunk;

        blocks(chunk[ p ], chunk[ c ], in[ p ], in[ c ]);

        // fills sparse row
        for(index_type i = 0; i < Nout; ++i)
        {
            index_type row = Nout * off + i;

            jacobian.beginRow( row );

            for(typename chunk_type::iterator c = chunk.begin(); c != chunk.end(); ++c)
            {

                for(index_type j = 0; j < Nin; ++j)
                {
                    index_type col = Nin * c->first + j;
                    jacobian.insertBack( row, col, c->second(i, j) );
                }
            }
        }
    }

    jacobian.compress();

}

template <class TIn, class TOut>
void RelativeRigidMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    if( jacobian.rowSize() > 0 )
        jacobian.mult(dOut,dIn);
}

template <class TIn, class TOut>
void RelativeRigidMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
{
    if( jacobian.rowSize() > 0 )
        jacobian.addMultTranspose(dIn,dOut);
}

template <class TIn, class TOut>
void RelativeRigidMapping<TIn, TOut>::applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& )
{
    //    cerr<<"RelativeRigidMapping<TIn, TOut>::applyJT does nothing " << endl;
}


template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* RelativeRigidMapping<TIn, TOut>::getJ()
{
    return &jacobian;
}

template <class TIn, class TOut>
const vector<sofa::defaulttype::BaseMatrix*>* RelativeRigidMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}

template <class TIn, class TOut>
void RelativeRigidMapping<TIn, TOut>::draw(const core::visual::VisualParams* /*vparams*/)
{
    // typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();
    // SeqEdges links = edgeContainer->getEdges();

    // vector< Vec3d > points;

    // for(unsigned i=0; i<links.size(); i++ ){
    //     points.push_back(pos[links[i][0]]);
    //     points.push_back(pos[links[i][1]]);
    // }
    // vparams->drawTool()->drawLines ( points, 1, Vec<4,float> ( 1,1,0,1 ) );
}


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
