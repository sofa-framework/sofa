/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
    , d_color(initData(&d_color, defaulttype::RGBAColor(1,1,0,1), "showColor", "Color for object display. (default=[1.0,1.0,0.0,1.0])"))
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

    jacobian.compress();
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

                SReal tmp = 2*childForce[i][0]*kfactor;

                InDeriv df = (parentDisplacement[links[i][0]]-parentDisplacement[links[i][1]])*tmp;
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
            SReal tmp = 2*childForce[i][0];

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




} // namespace mapping

} // namespace component

} // namespace sofa

#endif
