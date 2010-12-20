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
    : Inherit ( from, to ), FrameData<TIn>(), SampleData<TOut>()
    , f_initPos ( initData ( &f_initPos,"initPos","initial child coordinates in the world reference frame" ) )
    , f_index ( initData ( &f_index,"indices","parent indices for each child" ) )
    , weight ( initData ( &weight,"weights","influence weights of the Dofs" ) )
    , weightDeriv ( initData ( &weightDeriv,"weightGradients","weight gradients" ) )
    , weightDeriv2 ( initData ( &weightDeriv2,"weightHessians","weight Hessians" ) )
    //, f_initialInverseMatrices ( initData ( &f_initialInverseMatrices,"initialInverseMatrices","inverses of the initial parent matrices in the world reference frame" ) )
    , showBlendedFrame ( initData ( &showBlendedFrame, false, "showBlendedFrame","weights list for the influences of the references Dofs" ) )
    , showDefTensors ( initData ( &showDefTensors, false, "showDefTensors","show computed deformation tensors." ) )
    , showDefTensorsValues ( initData ( &showDefTensorsValues, false, "showDefTensorsValues","Show Deformation Tensors Values." ) )
    , showDefTensorScale ( initData ( &showDefTensorScale, 1.0, "showDefTensorScale","deformation tensor scale." ) )
    , showFromIndex ( initData ( &showFromIndex, ( unsigned ) 0, "showFromIndex","Displayed From Index." ) )
    , showDistancesValues ( initData ( &showDistancesValues, false, "showDistancesValues","Show dstances values." ) )
    , showWeights ( initData ( &showWeights, false, "showWeights","Show coeficients." ) )
    , showGammaCorrection ( initData ( &showGammaCorrection, 1.0, "showGammaCorrection","Correction of the Gamma by a power" ) )
    , showWeightsValues ( initData ( &showWeightsValues, false, "showWeightsValues","Show coeficients values." ) )
    , showReps ( initData ( &showReps, true, "showReps","Show repartition." ) )
    , showValuesNbDecimals ( initData ( &showValuesNbDecimals, 0, "showValuesNbDecimals","Multiply floating point by 10^n." ) )
    , showTextScaleFactor ( initData ( &showTextScaleFactor, 0.00005, "showTextScaleFactor","Text Scale Factor." ) )
    , showGradients ( initData ( &showGradients, false, "showGradients","Show gradients." ) )
    , showGradientsValues ( initData ( &showGradientsValues, false, "showGradientsValues","Show Gradients Values." ) )
    , showGradientsScaleFactor ( initData ( &showGradientsScaleFactor, 0.0001, "showGradientsScaleFactor","Gradients Scale Factor." ) )
    , useElastons ( initData ( &useElastons, false, "useElastons","Use Elastons to improve numerical integration" ) )
    , targetFrameNumber ( initData ( &targetFrameNumber, "targetFrameNumber","Target frames number" ) )
    , targetSampleNumber ( initData ( &targetSampleNumber, "targetSampleNumber","Target samples number" ) )
{
//                maskFrom = NULL;
//                if ( core::behavior::BaseMechanicalState *stateFrom = dynamic_cast< core::behavior::BaseMechanicalState *> ( from ) )
//                    maskFrom = &stateFrom->forceMask;
//                maskTo = NULL;
//                if ( core::behavior::BaseMechanicalState *stateTo = dynamic_cast< core::behavior::BaseMechanicalState *> ( to ) )
//                    maskTo = &stateTo->forceMask;
}

template <class TIn, class TOut>
FrameBlendingMapping<TIn, TOut>::~FrameBlendingMapping ()
{
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::init()
{
    // Set isPhysical internal flag.
    this->isPhysical = (defaulttype::OutDataTypesInfo<Out,OutReal,num_spatial_dimensions>::primitive_order > 0);

    // init samples and frames according to target numbers
    gridMaterial=NULL;
    this->getContext()->get( gridMaterial, core::objectmodel::BaseContext::SearchRoot);
    if ( !gridMaterial )
    {
        serr << "GridMaterial component not found -> use model vertices as Gauss point and 1/d^2 as weights." << sendl;
    }
    else
    {
        initFrames();
        initSamples();
    }

    // update init pos (necessary ??)
    //   unsigned numParents = this->fromModel->getSize();
    unsigned numChildren = this->toModel->getSize();
    ReadAccessor<Data<VecOutCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
    WriteAccessor<Data<VecOutCoord> > initPos(this->f_initPos);

    if( this->f_initPos.getValue().size() != numChildren )
    {
        initPos.resize(out.size());
        for(unsigned i=0; i<out.size(); i++ )
            initPos[i] = out[i];
    }


    // init weights and sample info (mass, moments) todo: ask the Material
    updateWeights();
    LumpMassesToFrames ();

    // init jacobians for mapping
    inout.resize( out.size() );
    for(unsigned i=0; i<out.size(); i++ )
    {
        inout[i].init(
            this->f_initPos.getValue()[i],
            f_index.getValue()[i],
            this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),
            weight.getValue()[i],
            weightDeriv.getValue()[i],
            weightDeriv2.getValue()[i]
        );
    }

    Inherit::init();
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    if( this->f_printLog.getValue() )
    {
        cerr<<"FrameBlendingMapping<TIn, TOut>::apply, in = "<< in << endl;
    }
    for ( unsigned int i = 0 ; i < out.size(); i++ )
    {
        out[i] = inout[i].apply( in );
        if( this->f_printLog.getValue() )
        {
            cerr<<"FrameBlendingMapping<TIn, TOut>::apply, out = "<< out[i] << endl;
        }
    }

}

template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{

//                if ( ! ( this->maskTo->isInUse() ) )
//                {
    if( this->f_printLog.getValue() )
    {
        cerr<<"FrameBlendingMapping<TIn, TOut>::applyJ, in = "<< in << endl;
    }
    for ( unsigned int i=0; i<out.size(); i++ )
    {
        out[i] = inout[i].mult( in );
        if( this->f_printLog.getValue() )
        {
            cerr<<"FrameBlendingMapping<TIn, TOut>::applyJ, out = "<< out[i] << endl;
        }
    }
//                }
//                else
//                {
//                    typedef helper::ParticleMask ParticleMask;
//                    const ParticleMask::InternalStorage &indices=this->maskTo->getEntries();
//
//                    ParticleMask::InternalStorage::const_iterator it;
//                    for ( it=indices.begin();it!=indices.end();it++ )
//                    {
//                        unsigned i= ( unsigned ) ( *it );
//                        out[i] = inout[i].mult( in );
//                    }
//                }
}

template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
//                if ( ! ( this->maskTo->isInUse() ) )
//                {
//                    this->maskFrom->setInUse ( false );
    if( this->f_printLog.getValue() )
    {
        cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, parent values before = "<< out << endl;
    }
    for ( unsigned int i=0; i<in.size(); i++ ) // VecType
    {
        inout[i].addMultTranspose( out, in[i] );
        if( this->f_printLog.getValue() )
        {
            cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, child value = "<< in[i] << endl;
        }
    }
    if( this->f_printLog.getValue() )
    {
        cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, parent values after = "<< out << endl;
    }
//                }
//                else
//                {
//                    typedef helper::ParticleMask ParticleMask;
//                    const ParticleMask::InternalStorage &indices=this->maskTo->getEntries();
//
//                    ParticleMask::InternalStorage::const_iterator it;
//                    if( this->f_printLog.getValue() ){
//                        cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, parent values before = "<< out << endl;
//                    }
//                    for ( it=indices.begin();it!=indices.end();it++ ) // VecType
//                    {
//                        const int i= ( int ) ( *it );
//                        inout[i].addMultTranspose( out, in[i] );
//                        if( this->f_printLog.getValue() ){
//                            cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, child value = "<< in[i] << endl;
//                        }
//                    }
//                    if( this->f_printLog.getValue() ){
//                        cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, parent values after = "<< out << endl;
//                    }
//                }
}


template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::applyJT ( typename In::MatrixDeriv& /*out*/, const typename Out::MatrixDeriv& /*in*/ )
{
//                if( this->f_printLog.getValue() ){
//                    cerr<<"WARNING ! FrameBlendingMapping<TIn, TOut>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) not implemented"<< endl;
//                }
}




template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::initFrames()
{
    if( targetFrameNumber.getValue() == 0) return;

    // Get references
    WriteAccessor<Data<VecInCoord> > xfrom0 = *this->fromModel->write(core::VecCoordId::restPosition());
    WriteAccessor<Data<VecInCoord> >  xfrom = *this->fromModel->write(core::VecCoordId::position());
    WriteAccessor<Data<VecInCoord> >  xfromReset = *this->fromModel->write(core::VecCoordId::resetPosition());
    unsigned int num_points=xfrom0.size();

    core::behavior::MechanicalState< In >* mstateFrom = dynamic_cast<core::behavior::MechanicalState< In >* >( this->fromModel);
    if ( !mstateFrom)
    {
        serr << "Error: try to insert new frames, which are not mechanical states !" << sendl;
        return;
    }

    // retrieve initial frames
    vector<SpatialCoord> points(num_points);
    for ( unsigned int i=0; i<num_points; i++ )
        for ( unsigned int j=0; j<num_spatial_dimensions; j++ )
            points[i][j]= xfrom0[i][j];

    // Insert new frames and compute associated voxel weights
    std::cout<<"Inserting "<<targetFrameNumber.getValue()-num_points<<" frames..."<<std::endl;
    gridMaterial->computeUniformSampling(points,targetFrameNumber.getValue());
    std::cout<<"Computing weights in grid..."<<std::endl;
    gridMaterial->computeWeights(points);

    //// copy
    this->fromModel->resize(points.size());
    for ( unsigned int i=num_points; i<points.size(); i++ )
        for ( unsigned int j=0; j<num_spatial_dimensions; j++ )
            xfrom[i][j] = xfrom0[i][j] = xfromReset[i][j]=  points[i][j];
}


template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::initSamples()
{
    if(!this->isPhysical)  return; // no gauss point -> use visual/collision or used-define points

    // Get references
    WriteAccessor<Data<VecOutCoord> >  xto0 = *this->toModel->write(core::VecCoordId::restPosition());
    WriteAccessor<Data<VecOutCoord> >  xto = *this->toModel->write(core::VecCoordId::position());
    WriteAccessor<Data<VecOutCoord> >  xtoReset = *this->toModel->write(core::VecCoordId::resetPosition());
    unsigned int num_points=xto0.size();

    core::behavior::MechanicalState< Out >* mstateto = dynamic_cast<core::behavior::MechanicalState< Out >* >( this->toModel);
    if ( !mstateto)
    {
        serr << "Error: try to insert new samples, which are not mechanical states !" << sendl;
        return;
    }

    // retrieve initial samples -> is problematic when there is no user sample : the mechanical object is always initialized with one sample centered on 0
    /*vector<SpatialCoord> points(num_points);
    for ( unsigned int i=0;i<num_points;i++ )
    	for ( unsigned int j=0;j<num_spatial_dimensions;j++ )
    		points[i][j]= xto0[i][j];*/

    num_points=0;
    vector<MaterialCoord> p(num_points);

    // Insert new samples
    //gridMaterial->computeRegularSampling(p,3);
    // gridMaterial->computeUniformSampling(p,targetSampleNumber.getValue(),100);
    gridMaterial->computeLinearRegionsSampling(p,0.1);

    WriteAccessor<Data<typename MaterialTraits<typename Out::Coord>::VecMaterialCoord> >  points(this->f_materialPoints);
    points.resize(p.size());
    for(unsigned i=0; i<p.size(); i++ )
        points[i] = p[i];

    std::cout<<"Inserting "<<points.size()<<" gauss points..."<<std::endl;

    // copy
    this->toModel->resize(points.size());
    for ( unsigned int i=num_points; i<points.size(); i++ )
        for ( unsigned int j=0; j<num_spatial_dimensions; j++ )
            xto[i][j] = xto0[i][j] = xtoReset[i][j]= points[i][j];
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::updateWeights ()
{
    ReadAccessor<Data<VecOutCoord> > xto (this->f_initPos);
    ReadAccessor<Data<VecInCoord> > xfrom = *this->fromModel->read(core::ConstVecCoordId::restPosition());
    WriteAccessor<Data<vector<Vec<nbRef,InReal> > > >       m_weights  ( weight );
    WriteAccessor<Data<vector<Vec<nbRef,MaterialDeriv> > > > m_dweight  ( weightDeriv );
    WriteAccessor<Data<vector<Vec<nbRef,MaterialMat> > > >   m_ddweight ( weightDeriv2 );
    WriteAccessor<Data<vector<Vec<nbRef,unsigned> > > > index ( f_index );

    index.resize( xto.size() );
    m_weights.resize ( xto.size() );
    //if(primitiveorder > 0)
    m_dweight.resize ( xto.size() );
    //if(primitiveorder > 1)
    m_ddweight.resize( xto.size() );

    if(gridMaterial)
    {
        if(this->isPhysical) std::cout<<"Lumping weights to gauss points..."<<std::endl;
        SpatialCoord point;

        for (unsigned i=0; i<xto.size(); i++ )
        {
            Out::get(point[0],point[1],point[2], xto[i]);

            if(!this->isPhysical)  // no gauss point here -> interpolate weights in the grid
                gridMaterial->interpolateWeightsRepartition(point,index[i],m_weights[i]);
            else // gauss points generated -> approximate weights over a set of voxels by least squares fitting
                gridMaterial->lumpWeightsRepartition(i,point,index[i],m_weights[i],&m_dweight[i],&m_ddweight[i]);
        }
    }
    else
    {
        if(xfrom.size()==2)  // linear weights based on 2 closest primitives
        {
            for (unsigned int i=0; i<xto.size(); i++ )
            {
                Vec<3,InReal> cto; Out::get( cto[0],cto[1],cto[2], xto[i] ); // OutReal??

                // get the 2 closest primitives
                for (unsigned int j=0; j<nbRef; j++ )
                {
                    m_weights[i][j]=0; index[i][j]=0;
                    m_dweight[i][j].fill(0);
                    m_ddweight[i][j].fill(0);
                }
                m_weights[i][0]=std::numeric_limits<InReal>::max();
                m_weights[i][1]=std::numeric_limits<InReal>::max();
                for (unsigned int j=0; j<xfrom.size(); j++ )
                {
                    Vec<3,InReal> cfrom; In::get( cfrom[0],cfrom[1],cfrom[2], xfrom[j] );
                    InReal d=(cto-cfrom).norm();
                    if(m_weights[i][0]>d) {m_weights[i][0]=d; index[i][0]=j;}
                    else if(m_weights[i][1]>d) {m_weights[i][1]=d; index[i][1]=j;}
                }
                Vec<3,InReal> cfrom1; In::get( cfrom1[0],cfrom1[1],cfrom1[2], xfrom[index[i][0]] );
                Vec<3,InReal> cfrom2; In::get( cfrom2[0],cfrom2[1],cfrom2[2], xfrom[index[i][1]] );
                Vec<3,InReal> u=cfrom2-cfrom1;
                InReal d=u.norm2(); u=u/d;
                InReal d1=dot(cto-cfrom1,u),d2=dot(cto-cfrom2,u);
                if(d1<0) d1=-d1; if(d2<0) d2=-d2;
                if(d1>d) m_weights[i][1]=1;
                else if(d2>d) m_weights[i][0]=1;
                else
                {
                    m_weights[i][0]=d2; m_weights[i][1]=d1;
                    m_dweight[i][0]=-u; m_dweight[i][1]=u;
                }
            }
        }
        else	// 1/d^2 weights with Euclidean distance
        {
            for (unsigned int i=0; i<xto.size(); i++ )
            {
                Vec<3,InReal> cto; Out::get( cto[0],cto[1],cto[2], xto[i] ); // OutReal??
                // get the nbRef closest primitives
                for (unsigned int j=0; j<nbRef; j++ )
                {
                    m_weights[i][j]=0;
                    index[i][j]=0;
                }
                //  cerr<<"FrameBlendingMapping<TIn, TOut>::updateWeights, xto = "<< xto[i] << endl;
                for (unsigned int j=0; j<xfrom.size(); j++ )
                {
                    Vec<3,InReal> cfrom; In::get( cfrom[0],cfrom[1],cfrom[2], xfrom[j] );
                    InReal w=(cto-cfrom)*(cto-cfrom);
                    // cerr<<"  distance = "<< sqrt(w) << endl;
                    if(w!=0) w=1./w;
                    else w=std::numeric_limits<InReal>::max();
                    unsigned m=0; while (m!=nbRef && m_weights[i][m]>w) m++;
                    if(m!=nbRef)
                    {
                        for (unsigned k=nbRef-1; k>m; k--)
                        {
                            m_weights[i][k]=m_weights[i][k-1];
                            index[i][k]=index[i][k-1];
                        }
                        m_weights[i][m]=w;
                        index[i][m]=j;
                    }
                }
                // compute weight gradients
                for (unsigned j=0; j<nbRef; j++ )
                {
                    InReal w=m_weights[i][j];
                    //    cerr<<"  weight = "<< w << endl;
                    m_dweight[i][j].fill(0);
                    m_ddweight[i][j].fill(0);
                    if (w)
                    {
                        InReal w2=w*w,w3=w2*w;
                        Vec<3,InReal> u;
                        for(unsigned k=0; k<3; k++)
                            u[k]=(xto[i][k]-xfrom[j][k]);
                        m_dweight[i][j] = - u * w2* 2.0;
                        // m_dweight[i][j] = u * w2; // hack FF for a special case. Todo: compute this right.
                        //  cerr<<" xfrom[j]  = "<< xfrom[j] << endl;
                        //  cerr<<"  m_dweight = "<< m_dweight[i][j] << endl;
                        for(unsigned k=0; k<num_spatial_dimensions; k++)
                            m_ddweight[i][j][k][k]= - w2* 2.0;
                        for(unsigned k=0; k<num_spatial_dimensions; k++)
                            for(unsigned m=0; m<num_spatial_dimensions; m++)
                                m_ddweight[i][j][k][m]+=u[k]*u[m]*w3* 8.0;
                    }
                }
            }
        }

    }
    normalizeWeights();
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::LumpMassesToFrames ( )
{

    ReadAccessor<Data<VecOutCoord> > out (*this->toModel->read(core::ConstVecCoordId::restPosition()));
    ReadAccessor<Data<VecInCoord> > in (*this->fromModel->read(core::ConstVecCoordId::restPosition()));

    if(!this->isPhysical) return; // no gauss point here -> no need for lumping

    MassVector& massVector = *this->f_mass0.beginEdit();
    massVector.resize(in.size());
    for(unsigned int i=0; i<in.size(); i++) massVector[i].clear();

    vector<Vec<nbRef,unsigned> > reps;
    vector<Vec<nbRef,InReal> > w;
    vector<SpatialCoord> pts;
    vector<InReal> masses;

    VecInDeriv d(in.size()),m(in.size());

    defaulttype::LinearBlendTypes<In,Vec3dTypes,GridMat,nbRef, 0 > map;

    for(unsigned int i=0; i<out.size(); i++) // treat each sample
    {
        if(gridMaterial)  gridMaterial->getWeightedMasses(i,reps,w,pts,masses);
        else
        {
            SpatialCoord point;
            Out::get(point[0],point[1],point[2], out[i]) ;
            pts.clear(); pts.push_back(point);
            w.clear(); w.push_back(weight.getValue()[i]);
            reps.clear(); reps.push_back(f_index.getValue()[i]);
            masses.clear();  masses.push_back(1);	// default value for the mass when model vertices are used as gauss points
        }

        for(unsigned int j=0; j<pts.size(); j++) // treat each voxel j of the sample
        {
            map.init(pts[j],reps[j],this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),w[j],Vec<nbRef,MaterialDeriv>(),Vec<nbRef,MaterialMat>());

            for(unsigned int k=0; k<nbRef && w[j][k]>0 ; k++) // treat each primitive influencing the voxel
            {
                unsigned int findex=reps[j][k];
                for(unsigned int l=0; l<InVSize; l++)	// treat each dof of the primitive
                {
                    d[findex][l]=1;
                    m[findex].clear();
                    map.addMultTranspose( m , map.mult(d) );  // get the contribution of j to each column l of the mass = mass(j).J^T.J.[0...1...0]^T
                    m[findex]*=masses[j];
                    for(unsigned int col=0; col<InVSize; col++)  massVector[findex].inertiaMassMatrix[col][l]+= m[findex][col];
                    d[reps[j][k]][l]=0;
                }
            }
        }
    }

    this->f_mass0.endEdit();
    //for(unsigned int i=0;i<in.size();i++) std::cout<<"mass["<<i<<"]="<<frameMass[i].inertiaMassMatrix<<std::endl;
}


template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::normalizeWeights()
{
    const unsigned int xtoSize = this->toModel->getX()->size();
    WriteAccessor<Data<vector<Vec<nbRef,InReal> > > >       m_weights  ( weight );
    WriteAccessor<Data<vector<Vec<nbRef,MaterialDeriv> > > > m_dweight  ( weightDeriv );
    WriteAccessor<Data<vector<Vec<nbRef,MaterialMat> > > >   m_ddweight ( weightDeriv2 );

    for (unsigned int i = 0; i < xtoSize; ++i)
    {
        InReal sumWeights = 0,wn;
        MaterialDeriv sumGrad,dwn;			sumGrad.fill(0);
        MaterialMat sumGrad2,ddwn;				sumGrad2.fill(0);

        // Compute norm
        for (unsigned int j = 0; j < nbRef; ++j)
        {
            sumWeights += m_weights[i][j];
            sumGrad += m_dweight[i][j];
            sumGrad2 += m_ddweight[i][j];
        }

        // Normalise
        if(sumWeights!=0)
        {
            for (unsigned int j = 0; j < nbRef; ++j)
            {
                wn=m_weights[i][j]/sumWeights;
                dwn=(m_dweight[i][j] - sumGrad*wn)/sumWeights;
                for(unsigned int o=0; o<num_material_dimensions; o++)
                {
                    for(unsigned int p=0; p<num_material_dimensions; p++)
                    {
                        ddwn[o][p]=(m_ddweight[i][j][o][p] - wn*sumGrad2[o][p] - sumGrad[o]*dwn[p] - sumGrad[p]*dwn[o])/sumWeights;
                    }
                }
                m_ddweight[i][j]=ddwn;
                m_dweight[i][j]=dwn;
                m_weights[i][j] =wn;
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
    //                const unsigned int nbRef = this->f_nbRefs.getValue();
    ReadAccessor<Data<vector<Vec<nbRef,unsigned> > > > m_reps = this->f_index;
    ReadAccessor<Data<vector<Vec<nbRef,InReal> > > > m_weights = weight ;
    ReadAccessor<Data<vector<Vec<nbRef,MaterialDeriv> > > >  m_dweights = weightDeriv ;
    const int valueScale = showValuesNbDecimals.getValue();
    int scale = 1;
    for (int i = 0; i < valueScale; ++i) scale *= 10;
    const double textScale = showTextScaleFactor.getValue();

    glDisable ( GL_LIGHTING );

    if ( this->getShow() )
    {
        //                    this->toModel->draw();

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
                const int idxReps=m_reps[i][m];
                double coef = m_weights[i][m];
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
        {
            SpatialCoord p;
            Out::get(p[0],p[1],p[2],xto[i]);
            sofa::helper::gl::GlText::draw ( m_reps[i][0]*scale, p, textScale );
        }
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
                const MaterialDeriv& grad = m_dweights[i][refIndex];
                sprintf( txt, "( %i, %i, %i)", (int)(grad[0]*scale), (int)(grad[1]*scale), (int)(grad[2]*scale));
                SpatialCoord p;
                Out::get(p[0],p[1],p[2],xto[i]);
                sofa::helper::gl::GlText::draw ( txt, p, textScale );
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
                SpatialCoord p;
                Out::get(p[0],p[1],p[2],xto[i]);
                sofa::helper::gl::GlText::draw ( (int)(m_weights[i][indexRep]*scale), p, textScale );
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
                const MaterialDeriv& gradMap = m_dweights[i][indexRep];
                SpatialCoord point;
                Out::get(point[0],point[1],point[2],xto[i]);
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
        InReal minValue = std::numeric_limits<InReal>::max();
        InReal maxValue = -std::numeric_limits<InReal>::min();
        for ( unsigned int i = 0; i < xto.size(); i++)
        {
            bool influenced;
            unsigned int indexRep;
            findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRef);
            if (influenced)
            {
                const InReal& weight = m_weights[i][indexRep];
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
                        float color = (float)(m_weights[indexPoint][indexRep] - minValue) / (maxValue - minValue);
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
                    float color = (float)(m_weights[i][indexRep] - minValue) / (maxValue - minValue);
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
    //                const unsigned int& nbRef = this->f_nbRefs.getValue();
    ReadAccessor<Data<vector<Vec<nbRef,unsigned> > > >  m_reps( f_index );
    influenced = false;
    for ( unsigned int j = 0; j < nbRef; ++j)
    {
        if ( m_reps[pointIndex][j] == frameIndex)
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
