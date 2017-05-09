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
    edgeContainer = dynamic_cast<topology::EdgeSetTopologyContainer*>( this->getContext()->getMeshTopology() );
    if( !edgeContainer ) serr<<"No EdgeSetTopologyContainer found ! "<<sendl;

    SeqEdges links = edgeContainer->getEdges();

    this->getToModel()->resize( links.size() );

    // only used for warning message
    bool compliance = ((simulation::Node*)(this->getContext()))->forceField.size() && ((simulation::Node*)(this->getContext()))->forceField[0]->isCompliance.getValue();
    if( compliance ) serr<<"Null rest Lengths cannot be used for stable compliant constraint, prefer to use a DifferenceMapping if those dofs are used with a compliance"<<sendl;

    // compute the rest lengths if they are not known
//    if( f_restLengths.getValue().size() != links.size() )
//    {
//        helper::WriteOnlyAccessor< Data<helper::vector<Real> > > restLengths(f_restLengths);
//        typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();
//        restLengths.resize( links.size() );
//        if(!(f_computeDistance.getValue()))
//        {
//            for(unsigned i=0; i<links.size(); i++ )
//            {
//                restLengths[i] = (pos[links[i][0]] - pos[links[i][1]]).norm();

//                if( restLengths[i]<=s_null_distance_epsilon && compliance ) serr<<"Null rest Length cannot be used for stable compliant constraint, prefer to use a DifferenceMapping for this dof "<<i<<" if used with a compliance"<<sendl;
//            }
//        }
//        else
//        {
//            if( compliance ) serr<<"Null rest Lengths cannot be used for stable compliant constraint, prefer to use a DifferenceMapping if those dofs are used with a compliance"<<sendl;
//            for(unsigned i=0; i<links.size(); i++ )
//                restLengths[i] = (Real)0.;
//        }
//    }
//    else // manually set
//        if( compliance ) // for warning message
//        {
//            helper::ReadAccessor< Data<helper::vector<Real> > > restLengths(f_restLengths);
//            for(unsigned i=0; i<links.size(); i++ )
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
    SeqEdges links = edgeContainer->getEdges();

    //    jacobian.clear();
    jacobian.resizeBlocks(out.size(),in.size());


    Direction gap;

    for(unsigned i=0; i<links.size(); i++ )
    {
        const InCoord& p0 = in[links[i][0]];
        const InCoord& p1 = in[links[i][1]];

        // gap = in[links[i][1]] - in[links[i][0]] (only for position)
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
        if( links[i][1]<links[i][0] )
        {
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
                jacobian.insertBack( i, links[i][1]*Nin+k, gap[k] );
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
                jacobian.insertBack( i, links[i][0]*Nin+k, -gap[k] );
        }
        else
        {
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
                jacobian.insertBack( i, links[i][0]*Nin+k, -gap[k] );
            for(unsigned k=0; k<In::spatial_dimensions; k++ )
                jacobian.insertBack( i, links[i][1]*Nin+k, gap[k] );
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
        const SeqEdges& links = edgeContainer->getEdges();

        for(unsigned i=0; i<links.size(); i++ )
        {
            // force in compression (>0) can lead to negative eigen values in geometric stiffness
            // this results in a undefinite implicit matrix that causes instabilies
            // if stabilized GS (geometricStiffness==2) -> keep only force in extension
            if( childForce[i][0] < 0 || geometricStiffness==1 )
            {

                const SReal tmp = 2*childForce[i][0]*kfactor;

                const InDeriv df = (parentDisplacement[links[i][0]]-parentDisplacement[links[i][1]])*tmp;
                // it is symmetric so    -df  = (parentDisplacement[links[i][1]]-parentDisplacement[links[i][0]])*tmp;

                parentForce[links[i][0]] += df;
                parentForce[links[i][1]] -= df;
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
    const SeqEdges& links = edgeContainer->getEdges();

    unsigned int size = this->fromModel->getSize();
    K.resizeBlocks(size,size);
    for(size_t i=0; i<links.size(); i++)
    {
        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in a undefinite implicit matrix that causes instabilies
        // if stabilized GS (geometricStiffness==2) -> keep only force in extension
        if( childForce[i][0] < 0 || geometricStiffness==1 )
        {
            const SReal tmp = 2*childForce[i][0];

            for(unsigned k=0; k<In::spatial_dimensions; k++)
            {
                K.add( links[i][0]*Nin+k, links[i][0]*Nin+k, tmp );
                K.add( links[i][0]*Nin+k, links[i][1]*Nin+k, -tmp );
                K.add( links[i][1]*Nin+k, links[i][1]*Nin+k, tmp );
                K.add( links[i][1]*Nin+k, links[i][0]*Nin+k, -tmp );
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
    const SeqEdges& links = edgeContainer->getEdges();



    if( d_showObjectScale.getValue() == 0 )
    {
        glDisable(GL_LIGHTING);
        helper::vector< defaulttype::Vector3 > points;
        for(unsigned i=0; i<links.size(); i++ )
        {
            points.push_back( sofa::defaulttype::Vector3( TIn::getCPos(pos[links[i][0]]) ) );
            points.push_back( sofa::defaulttype::Vector3( TIn::getCPos(pos[links[i][1]]) ));
        }
        vparams->drawTool()->drawLines ( points, 1, d_color.getValue() );
    }
    else
    {
        glEnable(GL_LIGHTING);
        for(unsigned i=0; i<links.size(); i++ )
        {
            defaulttype::Vector3 p0 = TIn::getCPos(pos[links[i][0]]);
            defaulttype::Vector3 p1 = TIn::getCPos(pos[links[i][1]]);
            vparams->drawTool()->drawCylinder( p0, p1, (float)d_showObjectScale.getValue(), d_color.getValue() );
        }
    }

    glPopAttrib();
#endif // SOFA_NO_OPENGL
}



template <class TIn, class TOut>
void SquareDistanceMapping<TIn, TOut>::updateForceMask()
{
    const SeqEdges& links = edgeContainer->getEdges();

    for(size_t i=0; i<links.size(); i++ )
    {
        if (this->maskTo->getEntry( i ) )
        {
            this->maskFrom->insertEntry( links[i][0] );
            this->maskFrom->insertEntry( links[i][1] );
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
//    , f_computeDistance(initData(&f_computeDistance, false, "computeDistance", "if 'computeDistance = true', then rest length of each element equal 0, otherwise rest length is the initial lenght of each of them"))
//    , f_restLengths(initData(&f_restLengths, "restLengths", "Rest lengths of the connections"))
    , d_showObjectScale(initData(&d_showObjectScale, Real(0), "showObjectScale", "Scale for object display"))
    , d_color(initData(&d_color, defaulttype::Vec4f(1,1,0,1), "showColor", "Color for object display"))
    , d_indexPairs(initData(&d_indexPairs, "indexPairs", "list of couples (parent index + index in the parent)"))
    , d_geometricStiffness(initData(&d_geometricStiffness, (unsigned)2, "geometricStiffness", "0 -> no GS, 1 -> exact GS, 2 -> stabilized GS (default)"))
{
}

template <class TIn, class TOut>
SquareDistanceMultiMapping<TIn, TOut>::~SquareDistanceMultiMapping()
{
    release();
}


template <class TIn, class TOut>
void SquareDistanceMultiMapping<TIn, TOut>::addPoint( const core::BaseState* from, int index)
{

    // find the index of the parent state
    unsigned i;
    for(i=0; i<this->fromModels.size(); i++)
        if(this->fromModels.get(i)==from )
            break;
    if(i==this->fromModels.size())
    {
        serr<<"addPoint, parent "<<from->getName()<<" not found !"<< sendl;
        assert(0);
    }

    addPoint(i, index);
}

template <class TIn, class TOut>
void SquareDistanceMultiMapping<TIn, TOut>::addPoint( int from, int index)
{
    assert((size_t)from<this->fromModels.size());
    helper::vector<defaulttype::Vec2i>& indexPairsVector = *d_indexPairs.beginEdit();
    indexPairsVector.push_back(defaulttype::Vec2i(from,index));
    d_indexPairs.endEdit();
}


template <class TIn, class TOut>
void SquareDistanceMultiMapping<TIn, TOut>::init()
{
    edgeContainer = dynamic_cast<topology::EdgeSetTopologyContainer*>( this->getContext()->getMeshTopology() );
    if( !edgeContainer ) serr<<"No EdgeSetTopologyContainer found ! "<<sendl;

    const SeqEdges& links = edgeContainer->getEdges();

    this->getToModels()[0]->resize( links.size() );

//    const helper::vector<defaulttype::Vec2i>& pairs = d_indexPairs.getValue();

    // only used for warning message
    bool compliance = ((simulation::Node*)(this->getContext()))->forceField.size() && ((simulation::Node*)(this->getContext()))->forceField[0]->isCompliance.getValue();
    if( compliance ) serr<<"Null rest Lengths cannot be used for stable compliant constraint, prefer to use a DifferenceMapping if those dofs are used with a compliance"<<sendl;

//    // compute the rest lengths if they are not known
//    if( f_restLengths.getValue().size() != links.size() )
//    {
//        helper::WriteAccessor< Data<helper::vector<Real> > > restLengths(f_restLengths);
//        restLengths.resize( links.size() );
//        if(!(f_computeDistance.getValue()))
//        {
//            for(unsigned i=0; i<links.size(); i++ )
//            {
//                const defaulttype::Vec2i& pair0 = pairs[ links[i][0] ];
//                const defaulttype::Vec2i& pair1 = pairs[ links[i][1] ];

//                const InCoord& pos0 = this->getFromModels()[pair0[0]]->readPositions()[pair0[1]];
//                const InCoord& pos1 = this->getFromModels()[pair1[0]]->readPositions()[pair1[1]];

//                restLengths[i] = (pos0 - pos1).norm();

//                if( restLengths[i]==0 && compliance ) serr<<"Null rest Length cannot be used for stable compliant constraint, prefer to use a DifferenceMapping for this dof "<<i<<" if used with a compliance"<<sendl;
//            }
//        }
//        else
//        {
//            if( compliance ) serr<<"Null rest Lengths cannot be used for stable compliant constraint, prefer to use a DifferenceMapping if those dofs are used with a compliance"<<sendl;
//            for(unsigned i=0; i<links.size(); i++ )
//                restLengths[i] = (Real)0.;
//        }
//    }
//    else // manually set
//        if( compliance ) // for warning message
//        {
//            helper::ReadAccessor< Data<helper::vector<Real> > > restLengths(f_restLengths);
//            for(unsigned i=0; i<links.size(); i++ )
//                if( restLengths[i]<=s_null_distance_epsilon ) serr<<"Null rest Length cannot be used for stable compliant constraint, prefer to use a DifferenceMapping for this dof "<<i<<" if used with a compliance"<<sendl;
//        }

    alloc();

    this->Inherit::init();  // applies the mapping, so after the Data init
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

    const helper::vector<defaulttype::Vec2i>& pairs = d_indexPairs.getValue();
//    helper::ReadAccessor<Data<helper::vector<Real> > > restLengths(f_restLengths);
    const SeqEdges& links = edgeContainer->getEdges();

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

    for(unsigned i=0; i<links.size(); i++ )
    {
        Direction& gap = directions[i];

        const defaulttype::Vec2i& pair0 = pairs[ links[i][0] ];
        const defaulttype::Vec2i& pair1 = pairs[ links[i][1] ];

        const InCoord& pos0 = (*inPos[pair0[0]])[pair0[1]];
        const InCoord& pos1 = (*inPos[pair1[0]])[pair1[1]];

        // gap = pos1-pos0 (only for position)
        computeCoordPositionDifference( gap, pos0, pos1 );

        Real gapNorm2 = gap.norm2();
        out[i] = gapNorm2;  // output

        gap *= 2;

        SparseMatrixEigen* J0 = static_cast<SparseMatrixEigen*>(baseMatrices[pair0[0]]);
        SparseMatrixEigen* J1 = static_cast<SparseMatrixEigen*>(baseMatrices[pair1[0]]);

        J0->beginRowSafe(i);
        J1->beginRowSafe(i);

        for(unsigned k=0; k<In::spatial_dimensions; k++ )
        {
            J0->insertBack( i, pair0[1]*Nin+k, -gap[k] );
            J1->insertBack( i, pair1[1]*Nin+k,  gap[k] );
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
    const SeqEdges& links = edgeContainer->getEdges();
    const helper::vector<defaulttype::Vec2i>& pairs = d_indexPairs.getValue();

    unsigned size = this->getFromModels().size();

    helper::vector<InVecDeriv*> parentForce( size );
    helper::vector<const InVecDeriv*> parentDisplacement( size );
    for( unsigned i=0; i< size ; i++ )
    {
        core::State<In>* fromModel = this->getFromModels()[i];
        parentForce[i] = parentDfId[fromModel].write()->beginEdit();
        parentDisplacement[i] = &mparams->readDx(fromModel)->getValue();
    }


    for(unsigned i=0; i<links.size(); i++ )
    {
        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in a undefinite implicit matrix that causes instabilies
        // if stabilized GS (geometricStiffness==2) -> keep only force in extension
        if( childForce[i][0] < 0 || geometricStiffness==1 )
        {
            const defaulttype::Vec2i& pair0 = pairs[ links[i][0] ];
            const defaulttype::Vec2i& pair1 = pairs[ links[i][1] ];
            InVecDeriv& parentForce0 = *parentForce[pair0[0]];
            InVecDeriv& parentForce1 = *parentForce[pair1[0]];

            const InVecDeriv& parentDisplacement0 = *parentDisplacement[pair0[0]];
            const InVecDeriv& parentDisplacement1 = *parentDisplacement[pair1[0]];

            const SReal tmp = 2*childForce[i][0]*kfactor;

            Direction dx = TIn::getDPos(parentDisplacement1[pair1[1]]) - TIn::getDPos(parentDisplacement0[pair0[1]]);
            InDeriv df;
            TIn::setDPos(df,dx*tmp);

            parentForce0[pair0[1]] -= df;
            parentForce1[pair1[1]] += df;
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
    const SeqEdges& links = edgeContainer->getEdges();
    const helper::vector<defaulttype::Vec2i>& pairs = d_indexPairs.getValue();

    for(size_t i=0; i<links.size(); i++)
    {
        // force in compression (>0) can lead to negative eigen values in geometric stiffness
        // this results in a undefinite implicit matrix that causes instabilies
        // if stabilized GS (geometricStiffness==2) -> keep only force in extension
        if( childForce[i][0] < 0 || geometricStiffness==1 )
        {
            const SReal tmp = 2*childForce[i][0];

            const defaulttype::Vec2i& pair0 = pairs[ links[i][0] ];
            const defaulttype::Vec2i& pair1 = pairs[ links[i][1] ];

            // TODO optimize (precompute base Index per mechanicalobject)
            size_t globalIndex0 = 0;
            for( int i=0 ; i<pair0[0] ; ++i )
            {
                size_t insize = this->getFromModels()[i]->getSize();
                globalIndex0 += insize;
            }
            globalIndex0 += pair0[1];

            size_t globalIndex1 = 0;
            for( int i=0 ; i<pair1[0] ; ++i )
            {
                size_t insize = this->getFromModels()[i]->getSize();
                globalIndex1 += insize;
            }
            globalIndex1 += pair1[1];


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

    const SeqEdges& links = edgeContainer->getEdges();

    const helper::vector<defaulttype::Vec2i>& pairs = d_indexPairs.getValue();

    if( d_showObjectScale.getValue() == 0 )
    {
        helper::vector< defaulttype::Vector3 > points;
        for(unsigned i=0; i<links.size(); i++ )
        {
            const defaulttype::Vec2i& pair0 = pairs[ links[i][0] ];
            const defaulttype::Vec2i& pair1 = pairs[ links[i][1] ];

            const InCoord& pos0 = this->getFromModels()[pair0[0]]->readPositions()[pair0[1]];
            const InCoord& pos1 = this->getFromModels()[pair1[0]]->readPositions()[pair1[1]];

            points.push_back( defaulttype::Vector3( TIn::getCPos(pos0) ) );
            points.push_back( defaulttype::Vector3( TIn::getCPos(pos1) ) );
        }
        vparams->drawTool()->drawLines ( points, 1, d_color.getValue() );
    }
    else
    {
        for(unsigned i=0; i<links.size(); i++ )
        {
            const defaulttype::Vec2i& pair0 = pairs[ links[i][0] ];
            const defaulttype::Vec2i& pair1 = pairs[ links[i][1] ];

            const InCoord& pos0 = this->getFromModels()[pair0[0]]->readPositions()[pair0[1]];
            const InCoord& pos1 = this->getFromModels()[pair1[0]]->readPositions()[pair1[1]];

            defaulttype::Vector3 p0 = TIn::getCPos(pos0);
            defaulttype::Vector3 p1 = TIn::getCPos(pos1);
            vparams->drawTool()->drawCylinder( p0, p1, (float)d_showObjectScale.getValue(), d_color.getValue() );
        }
    }
}


template <class TIn, class TOut>
void SquareDistanceMultiMapping<TIn, TOut>::updateForceMask()
{
    const SeqEdges& links = edgeContainer->getEdges();
    const helper::vector<defaulttype::Vec2i>& pairs = d_indexPairs.getValue();

    for(size_t i=0; i<links.size(); i++ )
    {
        if( this->maskTo[0]->getEntry(i) )
        {
            const defaulttype::Vec2i& pair0 = pairs[ links[i][0] ];
            const defaulttype::Vec2i& pair1 = pairs[ links[i][1] ];

            this->maskFrom[pair0[0]]->insertEntry( pair0[1] );
            this->maskFrom[pair1[0]]->insertEntry( pair1[1] );
        }
    }
}




} // namespace mapping

} // namespace component

} // namespace sofa

#endif
