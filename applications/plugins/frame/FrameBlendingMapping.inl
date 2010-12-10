/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_INL
#define SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_INL

#include "FrameBlendingMapping.h"
#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <sofa/core/Mapping.inl>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/gl/glText.inl>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/DualQuat.h>
//#include <sofa/component/topology/DistanceOnGrid.inl>
//#include <sofa/component/topology/DynamicSparseGridTopologyContainer.h>
#include <sofa/simulation/common/Simulation.h>
#include <string>
#include <iostream>
#include <limits>

namespace sofa
{

namespace defaulttype
{

///////////////////////////////////////////////
//        Avoid Out specializations          //
///////////////////////////////////////////////
template<class _Real, int Dim>
inline const Vec<3,_Real>& center(const Vec<Dim,_Real>& c)
{
    return DeformationGradientTypes<3, 3, 2, _Real>::center(c);
}

template<class _Real, int Dim>
inline Vec<3,_Real>& center(Vec<Dim,_Real>& c)
{
    return DeformationGradientTypes<3, 3, 2, _Real>::center(c);
}

template<class _Real>
inline const Vec<3,_Real>& center(const Vec<3,_Real>& c)
{
    return c;
}

template<class _Real>
inline Vec<3,_Real>& center(Vec<3,_Real>& c)
{
    return c;
}

template<class _Real, int Dim>
inline Mat<Dim, Dim, _Real> covNN(const Vec<Dim,_Real>& v1, const Vec<Dim,_Real>& v2)
{
    Mat<Dim, Dim, _Real> res;
    for( unsigned int i = 0; i < Dim; ++i)
        for( unsigned int j = i; j < Dim; ++j)
        {
            res[i][j] = v1[i] * v2[j];
            res[j][i] = res[i][j];
        }
    return res;
}

template<class _Real, int Dim1, int Dim2>
inline Mat<Dim1, Dim2, _Real> covMN(const Vec<Dim1,_Real>& v1, const Vec<Dim2,_Real>& v2)
{
    Mat<Dim1, Dim2, _Real> res;
    for( unsigned int i = 0; i < Dim1; ++i)
        for( unsigned int j = i; j < Dim2; ++j)
        {
            res[i][j] = v1[i] * v2[j];
            res[j][i] = res[i][j];
        }
    return res;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
///////////////////////////////////////////////
}

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using sofa::component::topology::TriangleSetTopologyContainer;
using helper::WriteAccessor;
using helper::ReadAccessor;


template <class TIn, class TOut>
FrameBlendingMapping<TIn, TOut>::FrameBlendingMapping (core::State<In>* from, core::State<Out>* to )
    : Inherit ( from, to )
    , f_nbRefs ( initData ( &f_nbRefs, (unsigned)2, "nbRefs","number of parents for each child" ) )
    , f_index ( initData ( &f_index,"indices","parent indices for each child" ) )
    , weight ( initData ( &weight,"weights","influence weights of the Dofs" ) )
    , weightDeriv ( initData ( &weightDeriv,"weightGradients","weight gradients" ) )
    , weightDeriv2 ( initData ( &weightDeriv2,"weightHessians","weight Hessians" ) )
    , f_initPos ( initData ( &f_initPos,"initPos","initial child coordinates in the world reference frame" ) )
    //, f_initialInverseMatrices ( initData ( &f_initialInverseMatrices,"initialInverseMatrices","inverses of the initial parent matrices in the world reference frame" ) )
    , showBlendedFrame ( initData ( &showBlendedFrame, true, "showBlendedFrame","weights list for the influences of the references Dofs" ) )
    , showDefTensors ( initData ( &showDefTensors, true, "showDefTensors","show computed deformation tensors." ) )
    , showDefTensorsValues ( initData ( &showDefTensorsValues, true, "showDefTensorsValues","Show Deformation Tensors Values." ) )
    , showDefTensorScale ( initData ( &showDefTensorScale, 1.0, "showDefTensorScale","deformation tensor scale." ) )
    , showFromIndex ( initData ( &showFromIndex, ( unsigned ) 0, "showFromIndex","Displayed From Index." ) )
    , showDistancesValues ( initData ( &showDistancesValues, true, "showDistancesValues","Show dstances values." ) )
    , showWeights ( initData ( &showWeights, true, "showWeights","Show coeficients." ) )
    , showGammaCorrection ( initData ( &showGammaCorrection, 1.0, "showGammaCorrection","Correction of the Gamma by a power" ) )
    , showWeightsValues ( initData ( &showWeightsValues, true, "showWeightsValues","Show coeficients values." ) )
    , showReps ( initData ( &showReps, true, "showReps","Show repartition." ) )
    , showValuesNbDecimals ( initData ( &showValuesNbDecimals, 0, "showValuesNbDecimals","Multiply floating point by 10^n." ) )
    , showTextScaleFactor ( initData ( &showTextScaleFactor, 0.00005, "showTextScaleFactor","Text Scale Factor." ) )
    , showGradients ( initData ( &showGradients, true, "showGradients","Show gradients." ) )
    , showGradientsValues ( initData ( &showGradientsValues, true, "showGradientsValues","Show Gradients Values." ) )
    , showGradientsScaleFactor ( initData ( &showGradientsScaleFactor, 0.0001, "showGradientsScaleFactor","Gradients Scale Factor." ) )
    , useElastons ( initData ( &useElastons, false, "useElastons","Use Elastons to improve numerical integration" ) )
    , targetFrameNumber ( initData ( &targetFrameNumber, "targetFrameNumber","Target frames number" ) )
    , targetSampleNumber ( initData ( &targetSampleNumber, "targetSampleNumber","Target samples number" ) )
{
    maskFrom = NULL;
    if ( core::behavior::BaseMechanicalState *stateFrom = dynamic_cast< core::behavior::BaseMechanicalState *> ( from ) )
        maskFrom = &stateFrom->forceMask;
    maskTo = NULL;
    if ( core::behavior::BaseMechanicalState *stateTo = dynamic_cast< core::behavior::BaseMechanicalState *> ( to ) )
        maskTo = &stateTo->forceMask;
}

template <class TIn, class TOut>
FrameBlendingMapping<TIn, TOut>::~FrameBlendingMapping ()
{
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::init()
{
//                unsigned numParents = this->fromModel->getSize();
    unsigned numChildren = this->toModel->getSize();
    unsigned nbRef = f_nbRefs.getValue();
    ReadAccessor<Data<VecOutCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
    ReadAccessor<Data<VecInCoord> > in = *this->fromModel->read(core::ConstVecCoordId::position());
    ReadAccessor<Data<VecInCoord> > in0 = *this->fromModel->read(core::ConstVecCoordId::restPosition());
    ReadAccessor<Data<vector<unsigned> > > index = this->f_index;
    ReadAccessor<Data<vector<OutReal> > > weights(weight);
    ReadAccessor<Data<vector<MaterialCoord> > > dweights(weightDeriv);
    ReadAccessor<Data<vector<MaterialMat> > > ddweights(weightDeriv2);
    //WriteAccessor<Data<VecInCoord> > initialInverseMatrices(f_initialInverseMatrices);
    WriteAccessor<Data<VecOutCoord> > initPos(f_initPos);

    if( f_initPos.getValue().size() != numChildren )
    {
        initPos.resize(out.size());
        for(unsigned i=0; i<out.size(); i++ )
            initPos[i] = out[i];
    }

// now done in inout.init()
//                if( f_initialInverseMatrices.getValue().size() != numParents )
//                {
//                    initialInverseMatrices.resize(in0.size());
//                    for(unsigned i=0; i<initialInverseMatrices.size(); i++)
//                        initialInverseMatrices[i] = In::inverse(in0[i]);
////                    cerr<<"FrameBlendingMapping<TIn, TOut>::init(), matrices = "<< in << endl;
////                    cerr<<"FrameBlendingMapping<TIn, TOut>::init(), inverse matrices = "<< initialInverseMatrices << endl;
//                }


    gridMaterial=NULL;
    this->getContext()->get( gridMaterial, core::objectmodel::BaseContext::SearchRoot);
    if ( !gridMaterial )
    {
        serr << "GridMaterial component not found -> use model vertices as Gauss point and 1/d^2 as weights." << sendl;
    }
    else
    {
        //                    initFrames();
        initSamples();
    }
    updateWeights();

    inout.resize( out.size() * nbRef );
    for(unsigned i=0; i<out.size(); i++ )
        for( unsigned j=0; j<nbRef; j++ )
            inout[i*nbRef+j].init(
                in0[index[i*nbRef+j]],
                initPos[i],
                weight.getValue()[i*nbRef+j],
                weightDeriv.getValue()[i*nbRef+j],
                weightDeriv2.getValue()[i*nbRef+j]
            );

    /*	inout.init(
    			this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),
    			f_initPos.getValue(),
    			f_index.getValue(),
    			weight.getValue(),
    			weightDeriv.getValue(),
    			weightDeriv2.getValue()
    			);*/
    // compute J
    // now done in inout.init()
    //          J.resize( out.size() * nbRef );
    //          for(unsigned i=0; i<out.size(); i++ )
    //              for( unsigned j=0; j<nbRef; j++ )
    //inout[i*nbRef+j].computeJacobianBlock(in[index[i*nbRef+j]]);
    //                  //J[i*nbRef+j] = inout.computeJacobianBlock( in[index[i*nbRef+j]], initialInverseMatrices[index[i*nbRef+j]], initPos[i], weights[i*nbRef+j], dweights[i*nbRef+j], ddweights[i*nbRef+j]  );

    Inherit::init();



}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const unsigned int& nbRef = this->f_nbRefs.getValue();

    ReadAccessor<Data<VecOutCoord> >            initPos = this->f_initPos;
//                ReadAccessor<Data<VecInCoord> >             invmat = this->f_initialInverseMatrices;
//                ReadAccessor<Data<VecInCoord> >             initialInverseMatrices(f_initialInverseMatrices);
    ReadAccessor<Data<vector<unsigned> > >      index = this->f_index;
    ReadAccessor<Data<vector<OutReal> > >       weights = this->weight;
    ReadAccessor<Data<vector<MaterialCoord> > > dweights(weightDeriv);
    ReadAccessor<Data<vector<MaterialMat> > >   ddweights(weightDeriv2);

    //this->mm0.resize( in.size() );
    //for(unsigned i=0; i<mm0.size(); i++ )
    //    mm0[i] = In::mult(in[i],invmat[i]);

//                cerr<<"FrameBlendingMapping<TIn, TOut>::apply, in = "<< in << endl;
//                cerr<<"FrameBlendingMapping<TIn, TOut>::apply, invmat = "<< invmat << endl;
//                cerr<<"FrameBlendingMapping<TIn, TOut>::apply, mm0 = "<< mm0 << endl;


    for ( unsigned int i = 0 ; i < out.size(); i++ )
    {
        // Point transformation (apply)
        out[i] = OutCoord();
        for ( unsigned int j = 0 ; j < nbRef; ++j )
        {
            out[i] += inout[nbRef * i + j].mult( in[i] );
            //out[i] += inout.mult(mm0[index[nbRef * i + j]], initPos[i])  * weights[nbRef * i + j];
//                        cerr<<"  FrameBlendingMapping<TIn, TOut>::apply, initPos[i] = "<< initPos[i] <<",mm0[index[nbRef * i + j]]  = "<<mm0[index[nbRef * i + j]]<< endl;
//                        cerr<<"  FrameBlendingMapping<TIn, TOut>::apply, mult(mm0[index[nbRef * i + j]], initPos[i]) = "<< inout.mult(mm0[index[nbRef * i + j]], initPos[i]) << endl;
        }
//                    cerr<<"FrameBlendingMapping<TIn, TOut>::apply, initPos = "<<initPos[i]<<", out= "<< out[i] << endl;
    }
    // update J
    for(unsigned i=0; i<out.size(); i++ )
    {
        for( unsigned j=0; j<nbRef; j++ )
        {
            inout[i*nbRef+j].updateJacobian( in[index[i*nbRef+j]]);
            //J[i*nbRef+j] = inout.computeJacobianBlock( in[index[i*nbRef+j]], initialInverseMatrices[index[i*nbRef+j]],  initPos[i], weights[i*nbRef+j], dweights[i*nbRef+j], ddweights[i*nbRef+j]  );
        }
    }
}

template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    unsigned nbRef = this->f_nbRefs.getValue();
    const vector<unsigned> index = this->f_index.getValue();

    if ( ! ( this->maskTo->isInUse() ) )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            out[i] = OutDeriv();
            for ( unsigned int j=0 ; j<nbRef; j++ )
            {
                out[i] += inout[nbRef*i+j].mult( in[index[nbRef*i+j]] );
                //out[i] += inout.mult( this->J[nbRef*i+j] , in[index[nbRef*i+j]] );
            }
        }
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=this->maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;
        for ( it=indices.begin(); it!=indices.end(); it++ )
        {
            unsigned i= ( unsigned ) ( *it );
            out[i] = OutDeriv();
            for ( unsigned int j=0 ; j<nbRef; j++ )
            {
                out[i] += inout[nbRef*i+j].mult( in[index[nbRef*i+j]] );
                //out[i] += inout.mult( this->J[nbRef*i+j] , in[index[nbRef*i+j]] );
            }
        }
    }
}

template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    unsigned nbRef = this->f_nbRefs.getValue();
    const vector<unsigned> index = this->f_index.getValue();
    if ( ! ( this->maskTo->isInUse() ) )
    {
        this->maskFrom->setInUse ( false );
        for ( unsigned int i=0; i<in.size(); i++ ) // VecType
        {
            for ( unsigned int j=0 ; j<nbRef; j++ ) // AffineType
            {
                out[index[nbRef*i+j]] += inout[nbRef*i+j].multTranspose( in[i] );
                //out[index[nbRef*i+j]] += inout.multTranspose( this->J[nbRef*i+j], in[i] );
            }
        }
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=this->maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;
        for ( it=indices.begin(); it!=indices.end(); it++ ) // VecType
        {
            const int i= ( int ) ( *it );
            for ( unsigned int j=0 ; j<nbRef; j++ ) // AffineType
            {
                out[index[nbRef*i+j]] += inout[nbRef*i+j].multTranspose( in[i] );
                //out[index[nbRef*i+j]] += inout.multTranspose( this->J[nbRef*i+j], in[i] );
            }
        }
    }
}


template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::applyJT ( typename In::MatrixDeriv& /*out*/, const typename Out::MatrixDeriv& /*in*/ )
{
    cerr<<"WARNING ! FrameBlendingMapping<TIn, TOut>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) not implemented"<< endl;
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::initSamples()
{
    if( targetSampleNumber.getValue() == 0) return;

    // Get references
    WriteAccessor<Data<VecOutCoord> > xto0 = *this->toModel->write(core::VecCoordId::restPosition());
    WriteAccessor<Data<VecOutCoord> >  xto = *this->toModel->write(core::VecCoordId::position());
    WriteAccessor<Data<VecOutCoord> >  xtoReset = *this->toModel->write(core::VecCoordId::resetPosition());

    core::behavior::MechanicalState< Out >* mstateto = dynamic_cast<core::behavior::MechanicalState< Out >* >( this->toModel);
    if ( !mstateto)
    {
        serr << "Error: try to insert new samples, which are not mechanical states !" << sendl;
        return;
    }

    // Insert new samples
    std::cout<<"Inserting "<<targetSampleNumber.getValue()<<" gauss points..."<<std::endl;
    vector<MaterialCoord> points;
    gridMaterial->computeUniformSampling(points,targetSampleNumber.getValue());

    // copy
    this->toModel->resize(points.size());
    for ( unsigned int i=0; i<targetSampleNumber.getValue(); i++ )
        for ( unsigned int j=0; j<num_spatial_dimensions; j++ )
            xto[i][j] = xto0[i][j] = xtoReset[i][j]= points[i][j];
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::updateWeights ()
{
    std::cout<<"Lumping weights to gauss points..."<<std::endl;

    //                const VecOutCoord& xto = ( this->toModel->getX0()->size() == 0)?*this->toModel->getX():*this->toModel->getX0();
    //                const VecInCoord& xfrom = *this->fromModel->getX0();
    ReadAccessor<Data<VecOutCoord> > xto (f_initPos);
    ReadAccessor<Data<VecInCoord> > xfrom = *this->fromModel->read(core::ConstVecCoordId::restPosition());

    WriteAccessor<Data<vector<OutReal> > >       m_weights  ( weight );
    WriteAccessor<Data<vector<MaterialCoord> > > m_dweight  ( weightDeriv );
    WriteAccessor<Data<vector<MaterialMat> > >   m_ddweight ( weightDeriv2 );

    const unsigned int& nbRef = this->f_nbRefs.getValue();
    WriteAccessor<Data<vector<unsigned> > > index ( f_index );

    m_weights.resize ( xto.size() * nbRef );
    m_dweight.resize ( xto.size() * nbRef );
    m_ddweight.resize( xto.size() * nbRef );
    index.resize( xto.size() * nbRef );

    if(gridMaterial)
    {
        vector<MaterialCoord> points ( xto.size() );
        for(unsigned i=0; i<points.size(); i++ )
            points[i] = xto[i];

        vector<OutReal> w;
        vector<unsigned> reps;
        vector<MaterialCoord> dw;
        vector<MaterialMat> ddw;

        for (unsigned i=0; i<xto.size(); i++ )
        {
            if(gridMaterial->lumpWeightsRepartition(points[i],reps,w,&dw,&ddw))
            {
                for (unsigned j=0; j<nbRef; j++)
                {
                    m_weights[nbRef*i+j]=w[j];
                    index[nbRef*i+j]=reps[j];
                    for(unsigned k=0; k<num_spatial_dimensions; k++)
                    {
                        m_dweight[nbRef*i+j][k]=dw[j][k];
                    }
                    for(unsigned k=0; k<num_spatial_dimensions; k++)
                    {
                        for(unsigned m=0; m<num_spatial_dimensions; m++)
                        {
                            m_ddweight[nbRef*i+j][k][m]=ddw[j][k][m];
                        }
                    }
                }
            }
        }
    }
    else	// 1/d^2 weights with Euclidean distance
    {
        OutReal w,w2,w3,w4;
        SpatialCoord u;
        for (unsigned i=0; i<xto.size(); i++ )
        {
            // get the nbRef closest primitives
            for (unsigned j=0; j<nbRef; j++ )
            {
                m_weights[nbRef*i+j]=0;
                index[nbRef*i+j]=0;
            }
//                        cerr<<"FrameBlendingMapping<TIn, TOut>::updateWeights, xto = "<< xto[i] << endl;
            for (unsigned j=0; j<xfrom.size(); j++ )
            {
                Vec<3,OutReal> cto; Out::get( cto[0],cto[1],cto[2], xto[i] );
                Vec<3,OutReal> cfrom; In::get( cfrom[0],cfrom[1],cfrom[2], xfrom[j] );
                w=(cto-cfrom)*(cto-cfrom);
//                            cerr<<"  distance = "<< sqrt(w) << endl;
                if(w!=0)
                    w=1./w;
                else
                    w=std::numeric_limits<OutReal>::max();
                unsigned m=0;
                while (m!=nbRef && m_weights[nbRef*i+m]>w)
                    m++;
                if(m!=nbRef)
                {
                    for (unsigned k=nbRef-1; k>m; k--)
                    {
                        m_weights[nbRef*i+k]=m_weights[nbRef*i+k-1];
                        index[nbRef*i+k]=index[nbRef*i+k-1];
                    }
                    m_weights[nbRef*i+m]=w;
                    index[nbRef*i+m]=j;
                }
            }
            // compute weight gradients
            for (unsigned j=0; j<nbRef; j++ )
            {
                w=m_weights[i*nbRef+j];
//                            cerr<<"  weight = "<< w << endl;
                m_dweight[i*nbRef+j].fill(0);
                m_ddweight[i*nbRef+j].fill(0);
                if (w)
                {
                    w2=w*w; w3=w2*w; w4=w3*w;
                    for(unsigned k=0; k<num_spatial_dimensions; k++)
                        u[k]=(xto[i][k]-xfrom[j][k]);
                    m_dweight[i*nbRef+j] = - u * w2* 2.0;
                    for(unsigned k=0; k<num_spatial_dimensions; k++)
                        m_ddweight[i*nbRef+j][k][k]= - w2* 2.0;
                    for(unsigned k=0; k<num_spatial_dimensions; k++)
                        for(unsigned m=0; m<num_spatial_dimensions; m++)
                            m_ddweight[i*nbRef+j][k][m]+=u[k]*u[m]*w3* 8.0;
                }
            }
        }
    }

    normalizeWeights();
}

template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::normalizeWeights()
{
    const unsigned int xtoSize = this->toModel->getX()->size();
    const unsigned int& nbRef = this->f_nbRefs.getValue();
    WriteAccessor<Data<vector<OutReal> > >       m_weights  ( weight );
    WriteAccessor<Data<vector<MaterialCoord> > > m_dweight  ( weightDeriv );
    WriteAccessor<Data<vector<MaterialMat> > >   m_ddweight ( weightDeriv2 );

    for (unsigned int i = 0; i < xtoSize; ++i)
    {
        OutReal sumWeights = 0,wn;
        MaterialCoord sumGrad,dwn;			sumGrad.fill(0);
        MaterialMat sumGrad2,ddwn;				sumGrad2.fill(0);

        // Compute norm
        for (unsigned int j = 0; j < nbRef; ++j)
        {
            sumWeights += m_weights[i*nbRef+j];
            sumGrad += m_dweight[i*nbRef+j];
            sumGrad2 += m_ddweight[i*nbRef+j];
        }

        // Normalise
        if(sumWeights!=0)
        {
            for (unsigned int j = 0; j < nbRef; ++j)
            {
                wn=m_weights[i*nbRef+j]/sumWeights;
                dwn=(m_dweight[i*nbRef+j] - sumGrad*wn)/sumWeights;
                for(unsigned int o=0; o<num_material_dimensions; o++)
                {
                    for(unsigned int p=0; p<num_material_dimensions; p++)
                    {
                        ddwn[o][p]=(m_ddweight[i*nbRef+j][o][p] - wn*sumGrad2[o][p] - sumGrad[o]*dwn[p] - sumGrad[p]*dwn[o])/sumWeights;
                    }
                }
                m_ddweight[i*nbRef+j]=ddwn;
                m_dweight[i*nbRef+j]=dwn;
                m_weights[i*nbRef+j] =wn;
//                            cerr<<"  normalized weight = "<< wn << endl;
            }
        }
    }
}






template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::draw()
{
    const typename Out::VecCoord& xto = *this->toModel->getX();
    const typename In::VecCoord& xfrom = *this->fromModel->getX();
    const unsigned int nbRef = this->f_nbRefs.getValue();
    ReadAccessor<Data<vector<unsigned> > > m_reps = this->f_index;
    ReadAccessor<Data<vector<OutReal> > > m_weights = weight ;
    ReadAccessor<Data<vector<MaterialCoord> > >  m_dweights = weightDeriv ;
    const int valueScale = showValuesNbDecimals.getValue();
    int scale = 1;
    for (int i = 0; i < valueScale; ++i) scale *= 10;
    const double textScale = showTextScaleFactor.getValue();

    glDisable ( GL_LIGHTING );

    if ( this->getShow() )
    {
        // Display mapping links between in and out elements
        glDisable ( GL_LIGHTING );
        glPointSize ( 1 );
        glColor4f ( 1,1,0,1 );
        glBegin ( GL_LINES );

        cerr<<"FrameBlendingMapping<TIn, TOut>::draw(), xto.size() = "<<xto.size()<<endl;
        cerr<<"FrameBlendingMapping<TIn, TOut>::draw(), nbRef = "<< nbRef <<endl;
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            for ( unsigned int m=0 ; m<nbRef; m++ )
            {
                const int idxReps=m_reps[i*nbRef+m];
                double coef = m_weights[i*nbRef+m];
                if ( coef > 0.0 )
                {
                    glColor4d ( coef,coef,0,1 );
                    glColor4d ( 1,1,1,1 );
                    helper::gl::glVertexT ( xfrom[idxReps].getCenter() );
                    helper::gl::glVertexT ( xto[i] );
                    cerr<<"FrameBlendingMapping<TIn, TOut>::draw() from "<< xfrom[idxReps].getCenter() << " to " << xto[i] << endl;
                }
            }
        }
        glEnd();
        cerr<<"FrameBlendingMapping<TIn, TOut>::draw()"<<endl;
    }

    // Display  m_reps for each points
    if ( showReps.getValue())
    {
        for ( unsigned int i=0; i<xto.size(); i++ )
            sofa::helper::gl::GlText::draw ( m_reps[i*nbRef+0]*scale, center(xto[i]), textScale );
    }

    // Display distances for each points
    //if ( showDistancesValues.getValue())
    //{
    //    glColor3f( 1.0, 1.0, 1.0);
    //    for ( unsigned int i=0;i<xto.size();i++ )
    //    {
    //        bool influenced;
    //        unsigned int refIndex;
    //        findIndexInRepartition(influenced, refIndex, i, showFromIndex.getValue()%nbRefs.getValue());
    //        if ( influenced)
    //        {
    //            sofa::helper::gl::GlText::draw ( (int)(distances[refIndex][i]*scale), xto[i], textScale );
    //        }
    //    }
    //}

    // Display distance gradients values for each points
    if ( showGradientsValues.getValue())
    {
        char txt[100];
        glColor3f( 0.5, 0.5, 0.5);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            bool influenced;
            unsigned int refIndex;
            findIndexInRepartition(influenced, refIndex, i, showFromIndex.getValue()%nbRef);
            if ( influenced)
            {
                const MaterialCoord& grad = m_dweights[i*nbRef+refIndex];
                sprintf( txt, "( %i, %i, %i)", (int)(grad[0]*scale), (int)(grad[1]*scale), (int)(grad[2]*scale));
                sofa::helper::gl::GlText::draw ( txt, center(xto[i]), textScale );
            }
        }
    }

    // Display weights for each points
    if ( showWeightsValues.getValue())
    {
        glColor3f( 1.0, 1.0, 1.0);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            bool influenced;
            unsigned int indexRep;
            findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRef);
            if ( influenced)
            {
                sofa::helper::gl::GlText::draw ( (int)(m_weights[i*nbRef+indexRep]*scale), center(xto[i]), textScale );
            }
        }
    }

    // Display weights gradients for each points
    if ( showGradients.getValue())
    {
        glColor3f ( 0.0, 1.0, 0.3 );
        glBegin ( GL_LINES );
        for ( unsigned int i = 0; i < xto.size(); i++ )
        {
            bool influenced;
            unsigned int indexRep;
            findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRef);
            if (influenced)
            {
                const SpatialCoord& gradMap = m_dweights[i*nbRef+indexRep];
                const SpatialCoord& point = center(xto[i]);
                glVertex3f ( point[0], point[1], point[2] );
                glVertex3f ( point[0] + gradMap[0] * showGradientsScaleFactor.getValue(), point[1] + gradMap[1] * showGradientsScaleFactor.getValue(), point[2] + gradMap[2] * showGradientsScaleFactor.getValue() );
            }
        }
        glEnd();
    }
    //

    // Show weights
    if ( showWeights.getValue())
    {
        // Compute min and max values.
        OutReal minValue = std::numeric_limits<OutReal>::max();
        OutReal maxValue = -std::numeric_limits<OutReal>::min();
        for ( unsigned int i = 0; i < xto.size(); i++)
        {
            bool influenced;
            unsigned int indexRep;
            findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRef);
            if (influenced)
            {
                const OutReal& weight = m_weights[i*nbRef+indexRep];
                if ( weight < minValue && weight != 0xFFF) minValue = weight;
                if ( weight > maxValue && weight != 0xFFF) maxValue = weight;
            }
        }

        TriangleSetTopologyContainer *mesh;
        this->getContext()->get( mesh);
        if ( mesh)
        {
            glPushAttrib( GL_LIGHTING_BIT || GL_COLOR_BUFFER_BIT || GL_ENABLE_BIT);
            std::vector< defaulttype::Vector3 > points;
            std::vector< defaulttype::Vector3 > normals;
            std::vector< defaulttype::Vec<4,float> > colors;
            const TriangleSetTopologyContainer::SeqTriangles& tri = mesh->getTriangles();
            for ( unsigned int i = 0; i < mesh->getNumberOfTriangles(); i++)
            {
                for ( unsigned int j = 0; j < 3; j++)
                {
                    bool influenced;
                    unsigned int indexRep;
                    //                                findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRef);  FF
                    findIndexInRepartition(influenced, indexRep, tri[i][j], showFromIndex.getValue()%nbRef);
                    if (influenced)
                    {
                        const unsigned int& indexPoint = tri[i][j];
                        float color = (float)(m_weights[indexPoint*nbRef+indexRep] - minValue) / (maxValue - minValue);
                        color = (float)pow((float)color, (float)showGammaCorrection.getValue());
                        points.push_back(defaulttype::Vector3(xto[indexPoint][0],xto[indexPoint][1],xto[indexPoint][2]));
                        colors.push_back(defaulttype::Vec<4,float>(color, 0.0, 0.0,1.0));
                    }
                }
            }
            simulation::getSimulation()->DrawUtility.drawTriangles(points, normals, colors);
            glPopAttrib();
        }
        else // Show by points
        {
            glPointSize( 10);
            glBegin( GL_POINTS);
            for ( unsigned int i = 0; i < xto.size(); i++)
            {
                bool influenced;
                unsigned int indexRep;
                findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRef);
                if (influenced)
                {
                    float color = (float)(m_weights[i*nbRef+indexRep] - minValue) / (maxValue - minValue);
                    color = (float)pow((float)color, (float)showGammaCorrection.getValue());
                    glColor3f( color, 0.0, 0.0);
                    glVertex3f( xto[i][0], xto[i][1], xto[i][2]);
                }
            }
            glEnd();
        }
    }

    //                // Display def tensor values for each points
    //                if ( this->showDefTensorsValues.getValue())
    //                {
    //                    char txt[100];
    //                    glColor3f( 0.5, 0.5, 0.5);
    //                    for ( unsigned int i=0;i<xto.size();i++ )
    //                    {
    //                        const Vec6& e = this->deformationTensors[i];
    //                        sprintf( txt, "( %i, %i, %i)", (int)(e[0]*scale), (int)(e[1]*scale), (int)(e[2]*scale));
    //                        sofa::helper::gl::GlText::draw ( txt, xto[i], textScale );
    //                    }
    //                }

    //                // Deformation tensor show
    //                if ( this->showDefTensors.getValue() && this->computeAllMatrices.getValue() )
    //                {
    //                    TriangleSetTopologyContainer *mesh;
    //                    this->getContext()->get( mesh);
    //                    if ( mesh)
    //                    {
    //                        glPushAttrib( GL_LIGHTING_BIT || GL_COLOR_BUFFER_BIT || GL_ENABLE_BIT);
    //                        glDisable( GL_LIGHTING);
    //                        glBegin( GL_TRIANGLES);
    //                        const TriangleSetTopologyContainer::SeqTriangles& tri = mesh->getTriangles();
    //                        for ( unsigned int i = 0; i < mesh->getNumberOfTriangles(); i++)
    //                        {
    //                            for ( unsigned int j = 0; j < 3; j++)
    //                            {
    //                                const Vec6& e = this->deformationTensors[tri[i][j]];
    //                                float color = 0.5 + ( e[0] + e[1] + e[2])/this->showDefTensorScale.getValue();
    //                                glColor3f( 0.0, color, 1.0-color);// /*e[0]*/, e[1], e[2]);
    //                                glVertex3f( xto[tri[i][j]][0], xto[tri[i][j]][1], xto[tri[i][j]][2]);
    //                            }
    //                        }
    //                        glEnd();
    //                        glPopAttrib();
    //                    }
    //                    else // Show by points
    //                    {
    //                        glPointSize( 10);
    //                        glBegin( GL_POINTS);
    //                        for ( unsigned int i = 0; i < xto.size(); i++)
    //                        {
    //                            const Vec6& e = this->deformationTensors[i];
    //                            float mult=500;
    //                            float color = (e[0]+e[1]+e[2])/3.;
    //                            if (color<0) color=2*color/(color+1.);
    //                            color*=mult;
    //                            color+=120;
    //                            if (color<0) color=0;
    //                            if (color>240) color=240;
    //                            sofa::helper::gl::Color::setHSVA(color,1.,.8,1.);
    //                            glVertex3f( xto[i][0], xto[i][1], xto[i][2]);
    //                        }
    //                        glEnd();
    //                    }
    //                }
}




template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::findIndexInRepartition( bool& influenced, unsigned int& realIndex, const unsigned int& pointIndex, const unsigned int& frameIndex)
{
    const unsigned int& nbRef = this->f_nbRefs.getValue();
    ReadAccessor<Data<vector<unsigned> > >  m_reps( f_index );
    influenced = false;
    for ( unsigned int j = 0; j < nbRef; ++j)
    {
        if ( m_reps[pointIndex*nbRef+j] == frameIndex)
        {
            influenced = true;
            realIndex = j;
            return;
        }
    }
}


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
