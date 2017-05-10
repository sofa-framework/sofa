/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_SquareDistanceMapping_INL
#define SOFA_COMPONENT_MAPPING_SquareDistanceMapping_INL

#include "SquareDistanceMapping.h"
#include <sofa/core/visual/VisualParams.h>
#include <iostream>
#include <sofa/simulation/Node.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace mapping
{


static const SReal s_null_distance_epsilon = 1e-8;


template <class TIn, class TOut>
SquareDistanceMapping<TIn, TOut>::SquareDistanceMapping()
    : Inherit()
    , d_pairs(initData(&d_pairs, "pairs", "couple of indices to compute distance in-between"))
//    , f_computeDistance(initData(&f_computeDistance, false, "computeDistance", "if no restLengths are given and if 'computeDistance = true', then rest length of each element equal 0, otherwise rest length is the initial lenght of each of them"))
//    , f_restLengths(initData(&f_restLengths, "restLengths", "Rest lengths of the connections"))
    , d_showObjectScale(initData(&d_showObjectScale, Real(0), "showObjectScale", "Scale for object display"))
    , d_color(initData(&d_color, defaulttype::Vec4f(1,1,0,1), "showColor", "Color for object display"))
    , d_geometricStiffness(initData(&d_geometricStiffness, (unsigned)2, "geometricStiffness", "0 -> no GS, 1 -> exact GS, 2 -> stabilized GS (default)"))
{
}

template <class TIn, class TOut>
SquareDistanceMapping<TIn, TOut>::~SquareDistanceMapping()
{
}


template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::init()
{
    // backward compatibility
    if( !d_pairs.isSet() )
    {
        topology::EdgeSetTopologyContainer* edgeContainer = dynamic_cast<topology::EdgeSetTopologyContainer*>( this->getContext()->getMeshTopology() );
        if( edgeContainer )
        {
            serr<<helper::logging::Message::Deprecated<<"Giving pairs from a EdgeSetTopologyContainer is deprecated, please use the Data 'pairs'"<<sendl;

            const topology::EdgeSetTopologyContainer::SeqEdges& edges = edgeContainer->getEdges();
            const size_t size = edges.size();

            VecPair& pairs = *d_pairs.beginWriteOnly();
            pairs.resize( size );
            for( size_t i=0 ; i<size ; ++i ) { pairs[i].set(edges[i][0],edges[i][1]); }
            d_pairs.endEdit();
        }
    }


    const VecPair& pairs = d_pairs.getValue();

    this->getToModel()->resize( pairs.size() );

    // only used for warning message
    bool compliance = ((simulation::Node*)(this->getContext()))->forceField.size() && ((simulation::Node*)(this->getContext()))->forceField[0]->isCompliance.getValue();
    if( compliance ) serr<<"Null rest Lengths cannot be used for stable compliant constraint, prefer to use a DifferenceMapping if those dofs are used with a compliance"<<sendl;

    // compute the rest lengths if they are not known
//    if( f_restLengths.getValue().size() != pairs.size() )
//    {
//        helper::WriteOnlyAccessor< Data<helper::vector<Real> > > restLengths(f_restLengths);
//        typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();
//        restLengths.resize( pairs.size() );
//        if(!(f_computeDistance.getValue()))
//        {
//            for(unsigned i=0; i<pairs.size(); i++ )
//            {
//                restLengths[i] = (pos[pairs[i][0]] - pos[pairs[i][1]]).norm();

//                if( restLengths[i]<=s_null_distance_epsilon && compliance ) serr<<"Null rest Length cannot be used for stable compliant constraint, prefer to use a DifferenceMapping for this dof "<<i<<" if used with a compliance"<<sendl;
//            }
//        }
//        else
//        {
//            if( compliance ) serr<<"Null rest Lengths cannot be used for stable compliant constraint, prefer to use a DifferenceMapping if those dofs are used with a compliance"<<sendl;
//            for(unsigned i=0; i<pairs.size(); i++ )
//                restLengths[i] = (Real)0.;
//        }
//    }
//    else // manually set
//        if( compliance ) // for warning message
//        {
//            helper::ReadAccessor< Data<helper::vector<Real> > > restLengths(f_restLengths);
//            for(unsigned i=0; i<pairs.size(); i++ )
//                if( restLengths[i]<=s_null_distance_epsilon ) serr<<"Null rest Length cannot be used for stable compliant constraint, prefer to use a DifferenceMapping for this dof "<<i<<" if used with a compliance"<<sendl;
//        }

    baseMatrices.resize( 1 );
    baseMatrices[0] = &jacobian;

    this->Inherit::init();  // applies the mapping, so after the Data init
}


template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::computeCoordPositionDifference( Direction& r, const InCoord& a, const InCoord& b )
{
    r = TIn::getCPos(b)-TIn::getCPos(a);
}

template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteOnlyAccessor< Data<OutVecCoord> >  out = dOut;
    helper::ReadAccessor< Data<InVecCoord> >  in = dIn;
//    helper::ReadAccessor<Data<helper::vector<Real> > > restLengths(f_restLengths);

    const VecPair& pairs = d_pairs.getValue();

    //    jacobian.clear();
    jacobian.resizeBlocks(out.size(),in.size());


    Direction gap;

    for(unsigned i=0; i<pairs.size(); i++ )
    {
        const InCoord& p0 = in[pairs[i][0]];
        const InCoord& p1 = in[pairs[i][1]];

        // gap = in[pairs[i][1]] - in[pairs[i][0]] (only for position)
        computeCoordPositionDifference( gap, p0, p1 );

        // if d = N - R  ==> d² = N² + R² - 2.N.R
        const Real gapNorm = gap.norm2();
        out[i][0] = gapNorm; // d = N²


        // insert in increasing column order
        gap *= 2; // 2*p[1]-2*p[0]

//        if( restLengths[i] )
//        {
//            out[i][0] -= ( 2*sqrt(gapNorm) + restLengths[i] ) * restLengths[i]; // d = N² + R² - 2.N.R

//            // TODO implement Jacobian when restpos != 0
//            // gap -=  d2NR/dx
//        }


        jacobian.beginRow(i);
        if( pairs[i][1]<pairs[i][0] )
        {
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
                jacobian.insertBack( i, pairs[i][1]*Nin+k, gap[k] );
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
                jacobian.insertBack( i, pairs[i][0]*Nin+k, -gap[k] );
        }
        else
        {
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
                jacobian.insertBack( i, pairs[i][0]*Nin+k, -gap[k] );
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
                jacobian.insertBack( i, pairs[i][1]*Nin+k, gap[k] );
        }
    }

    jacobian.finalize();
}


template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    if( jacobian.rowSize() )
        jacobian.mult(dOut,dIn);
}

template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
{
    if( jacobian.rowSize() )
        jacobian.addMultTranspose(dIn,dOut);
}

template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId )
{
    const unsigned& geometricStiffness = d_geometricStiffness.getValue();
    if( !geometricStiffness ) return;

    helper::WriteAccessor<Data<InVecDeriv> > parentForce (*parentDfId[this->fromModel.get(mparams)].write());
    helper::ReadAccessor<Data<InVecDeriv> > parentDisplacement (*mparams->readDx(this->fromModel));  // parent displacement
    const SReal& kfactor = mparams->kFactor();
    helper::ReadAccessor<Data<OutVecDeriv> > childForce (*mparams->readF(this->toModel));

    if( K.compressedMatrix.nonZeros() )
    {
        K.addMult( parentForce.wref(), parentDisplacement.ref(), (typename In::Real)kfactor );
    }
    else
    {
        const VecPair& pairs = d_pairs.getValue();

        for(unsigned i=0; i<pairs.size(); i++ )
        {
            // force in compression (>0) can lead to negative eigen values in geometric stiffness
            // this results in a undefinite implicit matrix that causes instabilies
            // if stabilized GS (geometricStiffness==2) -> keep only force in extension
            if( childForce[i][0] < 0 || geometricStiffness==1 )
            {

                const SReal tmp = 2*childForce[i][0]*kfactor;

                const InDeriv df = (parentDisplacement[pairs[i][0]]-parentDisplacement[pairs[i][1]])*tmp;
                // it is symmetric so    -df  = (parentDisplacement[pairs[i][1]]-parentDisplacement[pairs[i][0]])*tmp;

                parentForce[pairs[i][0]] += df;
                parentForce[pairs[i][1]] -= df;
            }
        }
    }
}

template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& )
{
    //    cerr<<"SquareDistanceMapping<TIn, TOut>::applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& ) does nothing " << endl;
}


template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* SquareDistanceMapping<TIn, TOut>::getJ()
{
    return &jacobian;
}

template <class TIn, class TOut>
const helper::vector<sofa::defaulttype::BaseMatrix*>* SquareDistanceMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}



template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::updateK(const core::MechanicalParams *mparams, core::ConstMultiVecDerivId childForceId )
{
    const unsigned& geometricStiffness = d_geometricStiffness.getValue();
    if( !geometricStiffness ) { K.resize(0,0); return; }


    helper::ReadAccessor<Data<OutVecDeriv> > childForce( *childForceId[this->toModel.get(mparams)].read() );
    const VecPair& pairs = d_pairs.getValue();

    unsigned int size = this->fromModel->getSize();
    K.resizeBlocks(size,size);
    for(size_t i=0; i<pairs.size(); i++)
    {
        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in a undefinite implicit matrix that causes instabilies
        // if stabilized GS (geometricStiffness==2) -> keep only force in extension
        if( childForce[i][0] < 0 || geometricStiffness==1 )
        {
            const SReal tmp = 2*childForce[i][0];

            for(unsigned k=0; k<In::spatial_dimensions; k++)
            {
                K.add( pairs[i][0]*Nin+k, pairs[i][0]*Nin+k, tmp );
                K.add( pairs[i][0]*Nin+k, pairs[i][1]*Nin+k, -tmp );
                K.add( pairs[i][1]*Nin+k, pairs[i][1]*Nin+k, tmp );
                K.add( pairs[i][1]*Nin+k, pairs[i][0]*Nin+k, -tmp );
            }
        }
    }
    K.compress();
}

template <class TIn, class TOut>
const defaulttype::BaseMatrix* SquareDistanceMapping<TIn, TOut>::getK()
{
    return &K;
}

template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if( !vparams->displayFlags().getShowMechanicalMappings() ) return;
#ifndef SOFA_NO_OPENGL
    glPushAttrib(GL_LIGHTING_BIT);

    typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();
    const VecPair& pairs = d_pairs.getValue();



    if( d_showObjectScale.getValue() == 0 )
    {
        glDisable(GL_LIGHTING);
        helper::vector< defaulttype::Vector3 > points;
        for(unsigned i=0; i<pairs.size(); i++ )
        {
            points.push_back( sofa::defaulttype::Vector3( TIn::getCPos(pos[pairs[i][0]]) ) );
            points.push_back( sofa::defaulttype::Vector3( TIn::getCPos(pos[pairs[i][1]]) ));
        }
        vparams->drawTool()->drawLines ( points, 1, d_color.getValue() );
    }
    else
    {
        glEnable(GL_LIGHTING);
        for(unsigned i=0; i<pairs.size(); i++ )
        {
            defaulttype::Vector3 p0 = TIn::getCPos(pos[pairs[i][0]]);
            defaulttype::Vector3 p1 = TIn::getCPos(pos[pairs[i][1]]);
            vparams->drawTool()->drawCylinder( p0, p1, (float)d_showObjectScale.getValue(), d_color.getValue() );
        }
    }

    glPopAttrib();
#endif // SOFA_NO_OPENGL
}



template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::updateForceMask()
{
    const VecPair& pairs = d_pairs.getValue();

    for(size_t i=0; i<pairs.size(); i++ )
    {
        if (this->maskTo->getEntry( i ) )
        {
            this->maskFrom->insertEntry( pairs[i][0] );
            this->maskFrom->insertEntry( pairs[i][1] );
        }
    }
}





///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////




template <class TIn, class TOut>
SquareDistanceMultiMapping<TIn, TOut>::SquareDistanceMultiMapping()
    : Inherit()
    , d_pairs(initData(&d_pairs, "pairs", "couple of indices (themselves couple of mstate index + dof index in mstate) to compute distance in-between"))
//    , f_computeDistance(initData(&f_computeDistance, false, "computeDistance", "if 'computeDistance = true', then rest length of each element equal 0, otherwise rest length is the initial lenght of each of them"))
//    , f_restLengths(initData(&f_restLengths, "restLengths", "Rest lengths of the connections"))
    , d_showObjectScale(initData(&d_showObjectScale, Real(0), "showObjectScale", "Scale for object display"))
    , d_color(initData(&d_color, defaulttype::Vec4f(1,1,0,1), "showColor", "Color for object display"))
    , d_geometricStiffness(initData(&d_geometricStiffness, (unsigned)2, "geometricStiffness", "0 -> no GS, 1 -> exact GS, 2 -> stabilized GS (default)"))
{
}

template <class TIn, class TOut>
SquareDistanceMultiMapping<TIn, TOut>::~SquareDistanceMultiMapping()
{
    release();
}



template <class TIn, class TOut>
void SquareDistanceMultiMapping<TIn, TOut>::init()
{
    const VecPair& pairs = d_pairs.getValue();

    if( pairs.empty() && this->getFromModels().size()==2 && this->getFromModels()[0]->getSize()==this->getFromModels()[1]->getSize()) // if no pair is defined-> map all dofs
    {
        helper::WriteOnlyAccessor<Data<VecPair> > p(d_pairs);
        p.resize(this->getFromModels()[0]->getSize());
        for( unsigned i = 0; i < p.size(); ++i)
        {
            p[i][0][0] = 0;
            p[i][0][1] = i;
            p[i][1][0] = 1;
            p[i][1][1] = i;
        }
    }

    this->getToModels()[0]->resize( pairs.size() );

//    const helper::vector<defaulttype::Vec2i>& pairs = d_indexPairs.getValue();

    // only used for warning message
    bool compliance = ((simulation::Node*)(this->getContext()))->forceField.size() && ((simulation::Node*)(this->getContext()))->forceField[0]->isCompliance.getValue();
    if( compliance ) serr<<"Null rest Lengths cannot be used for stable compliant constraint, prefer to use a DifferenceMapping if those dofs are used with a compliance"<<sendl;

//    // compute the rest lengths if they are not known
//    if( f_restLengths.getValue().size() != pairs.size() )
//    {
//        helper::WriteAccessor< Data<helper::vector<Real> > > restLengths(f_restLengths);
//        restLengths.resize( pairs.size() );
//        if(!(f_computeDistance.getValue()))
//        {
//            for(unsigned i=0; i<pairs.size(); i++ )
//            {
//                const defaulttype::Vec2i& pair[0] = pairs[ pairs[i][0] ];
//                const defaulttype::Vec2i& pair[1] = pairs[ pairs[i][1] ];

//                const InCoord& pos0 = this->getFromModels()[pair[0][0]]->readPositions()[pair[0][1]];
//                const InCoord& pos1 = this->getFromModels()[pair[1][0]]->readPositions()[pair[1][1]];

//                restLengths[i] = (pos0 - pos1).norm();

//                if( restLengths[i]==0 && compliance ) serr<<"Null rest Length cannot be used for stable compliant constraint, prefer to use a DifferenceMapping for this dof "<<i<<" if used with a compliance"<<sendl;
//            }
//        }
//        else
//        {
//            if( compliance ) serr<<"Null rest Lengths cannot be used for stable compliant constraint, prefer to use a DifferenceMapping if those dofs are used with a compliance"<<sendl;
//            for(unsigned i=0; i<pairs.size(); i++ )
//                restLengths[i] = (Real)0.;
//        }
//    }
//    else // manually set
//        if( compliance ) // for warning message
//        {
//            helper::ReadAccessor< Data<helper::vector<Real> > > restLengths(f_restLengths);
//            for(unsigned i=0; i<pairs.size(); i++ )
//                if( restLengths[i]<=s_null_distance_epsilon ) serr<<"Null rest Length cannot be used for stable compliant constraint, prefer to use a DifferenceMapping for this dof "<<i<<" if used with a compliance"<<sendl;
//        }

    alloc();

    this->Inherit::init();  // applies the mapping, so after the Data init
}


template <class TIn, class TOut>
void SquareDistanceMultiMapping<TIn, TOut>::parse( sofa::core::objectmodel::BaseObjectDescription* arg )
{
    Inherit1::parse(arg);

    // to be backward compatible with previous link description
    if( !d_pairs.isSet() )
    {
        const char* indexPairsChar = arg->getAttribute("indexPairs");
        if( indexPairsChar )
        {
            serr<<helper::logging::Message::Deprecated<<"parse: You are using a deprecated Data 'indexPairs' associated with a EdgeSetTopology, please use the new structure data 'pairs'"<<sendl;

            topology::EdgeSetTopologyContainer* edgeContainer = dynamic_cast<topology::EdgeSetTopologyContainer*>( this->getContext()->getMeshTopology() );
            if( edgeContainer )
            {
                helper::vector<defaulttype::Vec2i> indexPairs;
                std::istringstream ss( indexPairsChar );
                indexPairs.read( ss );

                const topology::EdgeSetTopologyContainer::SeqEdges& edges = edgeContainer->getEdges();
                const size_t size = edges.size();

                VecPair& pairs = *d_pairs.beginWriteOnly();
                pairs.resize( size );
                for( size_t i=0 ; i<size ; ++i )
                {
                    pairs[i][0][0] = indexPairs[edges[i][0]][0];
                    pairs[i][0][1] = indexPairs[edges[i][0]][1];
                    pairs[i][1][0] = indexPairs[edges[i][1]][0];
                    pairs[i][1][1] = indexPairs[edges[i][1]][1];
                }
                d_pairs.endEdit();
            }
            else
            {
                serr<<helper::logging::Message::Error<<"No EdgeSetTopologyContainer"<<sendl;
            }
        }
    }
}

template <class TIn, class TOut>
void SquareDistanceMultiMapping<TIn, TOut>::computeCoordPositionDifference( Direction& r, const InCoord& a, const InCoord& b )
{
    r = TIn::getCPos(b)-TIn::getCPos(a);
}

template <class TIn, class TOut>
void SquareDistanceMultiMapping<TIn, TOut>::apply(const helper::vector<OutVecCoord*>& outPos, const vecConstInVecCoord& inPos)
{
    OutVecCoord& out = *outPos[0];

//    helper::ReadAccessor<Data<helper::vector<Real> > > restLengths(f_restLengths);
    const VecPair& pairs = d_pairs.getValue();

    unsigned size = this->getFromModels().size();

    unsigned totalInSize = 0;
    for( unsigned i=0 ; i<size ; ++i )
    {
        size_t insize = inPos[i]->size();
        static_cast<SparseMatrixEigen*>(baseMatrices[i])->resizeBlocks(out.size(),insize);
        totalInSize += insize;
    }
//    fullJ.resizeBlocks( out.size(), totalInSize  );
    K.resizeBlocks( totalInSize, totalInSize  );

    directions.resize(out.size());
    invlengths.resize(out.size());

    for(unsigned i=0; i<pairs.size(); i++ )
    {
        Direction& gap = directions[i];

        const Pair& p = pairs[i];

        const InCoord& pos0 = (*inPos[p[0][0]])[p[0][1]];
        const InCoord& pos1 = (*inPos[p[1][0]])[p[1][1]];

        // gap = pos1-pos0 (only for position)
        computeCoordPositionDifference( gap, pos0, pos1 );

        Real gapNorm2 = gap.norm2();
        out[i] = gapNorm2;  // output

        gap *= 2;

        SparseMatrixEigen* J0 = static_cast<SparseMatrixEigen*>(baseMatrices[p[0][0]]);
        SparseMatrixEigen* J1 = static_cast<SparseMatrixEigen*>(baseMatrices[p[1][0]]);

        J0->beginRowSafe(i);
        J1->beginRowSafe(i);

        for(unsigned k=0; k<In::spatial_dimensions; k++ )
        {
            J0->insertBack( i, p[0][1]*Nin+k, -gap[k] );
            J1->insertBack( i, p[1][1]*Nin+k,  gap[k] );
        }
    }


    for( unsigned i=0 ; i<size ; ++i )
    {
        static_cast<SparseMatrixEigen*>(baseMatrices[i])->finalize();
    }

}


template <class TIn, class TOut>
void SquareDistanceMultiMapping<TIn, TOut>::applyJ(const helper::vector<OutVecDeriv*>& outDeriv, const helper::vector<const  InVecDeriv*>& inDeriv)
{
    unsigned n = baseMatrices.size();
    unsigned i = 0;

    // let the first valid jacobian set its contribution    out = J_0 * in_0
    for( ; i < n ; ++i ) {
        const SparseMatrixEigen& J = *static_cast<SparseMatrixEigen*>(baseMatrices[i]);
        if( J.rowSize() > 0 ) {
            J.mult(*outDeriv[0], *inDeriv[i]);
            break;
        }
    }

    ++i;

    // the next valid jacobians will add their contributions    out += J_i * in_i
    for( ; i < n ; ++i ) {
        const SparseMatrixEigen& J = *static_cast<SparseMatrixEigen*>(baseMatrices[i]);
        if( J.rowSize() > 0 )
            J.addMult(*outDeriv[0], *inDeriv[i]);
    }
}

template <class TIn, class TOut>
void SquareDistanceMultiMapping<TIn, TOut>::applyJT(const helper::vector< InVecDeriv*>& outDeriv, const helper::vector<const OutVecDeriv*>& inDeriv)
{
    for( unsigned i = 0, n = baseMatrices.size(); i < n; ++i) {
        const SparseMatrixEigen& J = *static_cast<SparseMatrixEigen*>(baseMatrices[i]);
        if( J.rowSize() > 0 )
            J.addMultTranspose(*outDeriv[i], *inDeriv[0]);
    }
}

template <class TIn, class TOut>
void SquareDistanceMultiMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId)
{
    // NOT OPTIMIZED AT ALL, but will do the job for now

    const unsigned& geometricStiffness = d_geometricStiffness.getValue();
    if( !geometricStiffness ) return;

    const SReal kfactor = mparams->kFactor();
    const OutVecDeriv& childForce = this->getToModels()[0]->readForces().ref();
    const VecPair& pairs = d_pairs.getValue();

    unsigned size = this->getFromModels().size();

    helper::vector<InVecDeriv*> parentForce( size );
    helper::vector<const InVecDeriv*> parentDisplacement( size );
    for( unsigned i=0; i< size ; i++ )
    {
        core::State<In>* fromModel = this->getFromModels()[i];
        parentForce[i] = parentDfId[fromModel].write()->beginEdit();
        parentDisplacement[i] = &mparams->readDx(fromModel)->getValue();
    }


    for(unsigned i=0; i<pairs.size(); i++ )
    {
        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in a undefinite implicit matrix that causes instabilies
        // if stabilized GS (geometricStiffness==2) -> keep only force in extension
        if( childForce[i][0] < 0 || geometricStiffness==1 )
        {
            const Pair& pair = pairs[i];

            InVecDeriv& parentForce0 = *parentForce[pair[0][0]];
            InVecDeriv& parentForce1 = *parentForce[pair[1][0]];

            const InVecDeriv& parentDisplacement0 = *parentDisplacement[pair[0][0]];
            const InVecDeriv& parentDisplacement1 = *parentDisplacement[pair[1][0]];

            const SReal tmp = 2*childForce[i][0]*kfactor;

            Direction dx = TIn::getDPos(parentDisplacement1[pair[1][1]]) - TIn::getDPos(parentDisplacement0[pair[0][1]]);
            InDeriv df;
            TIn::setDPos(df,dx*tmp);

            parentForce0[pair[0][1]] -= df;
            parentForce1[pair[1][1]] += df;
        }
    }

    for( unsigned i=0; i< size ; i++ )
    {
        core::State<In>* fromModel = this->getFromModels()[i];
        parentDfId[fromModel].write()->endEdit();
    }
}




template <class TIn, class TOut>
const helper::vector<sofa::defaulttype::BaseMatrix*>* SquareDistanceMultiMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}

template <class TIn, class TOut>
void SquareDistanceMultiMapping<TIn, TOut>::updateK(const core::MechanicalParams* /*mparams*/, core::ConstMultiVecDerivId childForceId )
{
    const unsigned& geometricStiffness = d_geometricStiffness.getValue();
    if( !geometricStiffness ) { K.resize(0,0); return; }

    helper::ReadAccessor<Data<OutVecDeriv> > childForce( *childForceId[(const core::State<TOut>*)this->getToModels()[0]].read() );
    const VecPair& pairs = d_pairs.getValue();

    for(size_t i=0; i<pairs.size(); i++)
    {
        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in a undefinite implicit matrix that causes instabilies
        // if stabilized GS (geometricStiffness==2) -> keep only force in extension
        if( childForce[i][0] < 0 || geometricStiffness==1 )
        {
            const SReal tmp = 2*childForce[i][0];

            const Pair& pair = pairs[i];

            // TODO optimize (precompute base Index per mechanicalobject)
            size_t globalIndex0 = 0;
            for( unsigned i=0 ; i<pair[0][0] ; ++i )
            {
                size_t insize = this->getFromModels()[i]->getSize();
                globalIndex0 += insize;
            }
            globalIndex0 += pair[0][1];

            size_t globalIndex1 = 0;
            for( unsigned i=0 ; i<pair[1][0] ; ++i )
            {
                size_t insize = this->getFromModels()[i]->getSize();
                globalIndex1 += insize;
            }
            globalIndex1 += pair[1][1];


            for(unsigned k=0; k<In::spatial_dimensions; k++)
            {
                K.add( globalIndex0*Nin+k, globalIndex0*Nin+k, tmp );
                K.add( globalIndex0*Nin+k, globalIndex1*Nin+k, -tmp );
                K.add( globalIndex1*Nin+k, globalIndex1*Nin+k, tmp );
                K.add( globalIndex1*Nin+k, globalIndex0*Nin+k, -tmp );
            }
        }
    }
    K.compress();
}


template <class TIn, class TOut>
const defaulttype::BaseMatrix* SquareDistanceMultiMapping<TIn, TOut>::getK()
{
    return &K;
}

template <class TIn, class TOut>
void SquareDistanceMultiMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if( !vparams->displayFlags().getShowMechanicalMappings() ) return;

    const VecPair& pairs = d_pairs.getValue();


    if( d_showObjectScale.getValue() == 0 )
    {
        helper::vector< defaulttype::Vector3 > points;
        for(unsigned i=0; i<pairs.size(); i++ )
        {
            const Pair& pair = pairs[i];

            const InCoord& pos0 = this->getFromModels()[pair[0][0]]->readPositions()[pair[0][1]];
            const InCoord& pos1 = this->getFromModels()[pair[1][0]]->readPositions()[pair[1][1]];

            points.push_back( defaulttype::Vector3( TIn::getCPos(pos0) ) );
            points.push_back( defaulttype::Vector3( TIn::getCPos(pos1) ) );
        }
        vparams->drawTool()->drawLines ( points, 1, d_color.getValue() );
    }
    else
    {
        for(unsigned i=0; i<pairs.size(); i++ )
        {
            const Pair& pair = pairs[i];

            const InCoord& pos0 = this->getFromModels()[pair[0][0]]->readPositions()[pair[0][1]];
            const InCoord& pos1 = this->getFromModels()[pair[1][0]]->readPositions()[pair[1][1]];

            defaulttype::Vector3 p0 = TIn::getCPos(pos0);
            defaulttype::Vector3 p1 = TIn::getCPos(pos1);
            vparams->drawTool()->drawCylinder( p0, p1, (float)d_showObjectScale.getValue(), d_color.getValue() );
        }
    }
}


template <class TIn, class TOut>
void SquareDistanceMultiMapping<TIn, TOut>::updateForceMask()
{
    const VecPair& pairs = d_pairs.getValue();

    for(size_t i=0; i<pairs.size(); i++ )
    {
        if( this->maskTo[0]->getEntry(i) )
        {
            const Pair& pair = pairs[i];

            this->maskFrom[pair[0][0]]->insertEntry( pair[0][1] );
            this->maskFrom[pair[1][0]]->insertEntry( pair[1][1] );
        }
    }
}




} // namespace mapping

} // namespace component

} // namespace sofa

#endif
