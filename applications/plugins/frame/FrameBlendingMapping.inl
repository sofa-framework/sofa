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


template <class TIn, class TOut>
FrameBlendingMapping<TIn, TOut>::FrameBlendingMapping (core::State<In>* from, core::State<Out>* to )
    : Inherit ( from, to )
    , weight ( initData ( &weight,"weights","influence weights of the Dofs" ) )
    , weightDeriv ( initData ( &weightDeriv,"weight gradients","weight gradients" ) )
    , weightDeriv2 ( initData ( &weightDeriv2,"weight Hessians","weight Hessians" ) )
    , showBlendedFrame ( initData ( &showBlendedFrame, false, "showBlendedFrame","weights list for the influences of the references Dofs" ) )
    , showDefTensors ( initData ( &showDefTensors, false, "showDefTensors","show computed deformation tensors." ) )
    , showDefTensorsValues ( initData ( &showDefTensorsValues, false, "showDefTensorsValues","Show Deformation Tensors Values." ) )
    , showDefTensorScale ( initData ( &showDefTensorScale, 1.0, "showDefTensorScale","deformation tensor scale." ) )
    , showFromIndex ( initData ( &showFromIndex, ( unsigned ) 0, "showFromIndex","Displayed From Index." ) )
    , showDistancesValues ( initData ( &showDistancesValues, false, "showDistancesValues","Show dstances values." ) )
    , showWeights ( initData ( &showWeights, false, "showWeights","Show coeficients." ) )
    , showGammaCorrection ( initData ( &showGammaCorrection, 1.0, "showGammaCorrection","Correction of the Gamma by a power" ) )
    , showWeightsValues ( initData ( &showWeightsValues, false, "showWeightsValues","Show coeficients values." ) )
    , showReps ( initData ( &showReps, false, "showReps","Show repartition." ) )
    , showValuesNbDecimals ( initData ( &showValuesNbDecimals, 0, "showValuesNbDecimals","Multiply floating point by 10^n." ) )
    , showTextScaleFactor ( initData ( &showTextScaleFactor, 0.00005, "showTextScaleFactor","Text Scale Factor." ) )
    , showGradients ( initData ( &showGradients, false, "showGradients","Show gradients." ) )
    , showGradientsValues ( initData ( &showGradientsValues, false, "showGradientsValues","Show Gradients Values." ) )
    , showGradientsScaleFactor ( initData ( &showGradientsScaleFactor, 0.0001, "showGradientsScaleFactor","Gradients Scale Factor." ) )
    //, enableSkinning ( initData ( &enableSkinning, true, "enableSkinning","enable skinning." ) )
    //      , voxelVolume ( initData ( &voxelVolume, 1.0, "voxelVolume","default volume voxel. Use if no hexa topo is found." ) )
    , useElastons ( initData ( &useElastons, false, "useElastons","Use Elastons to improve numerical integration" ) )
    , targetFrameNumber ( initData ( &targetFrameNumber, false, "targetFrameNumber","Target frames number" ) )
    , targetSampleNumber ( initData ( &targetSampleNumber, false, "targetSampleNumber","Target samples number" ) )
    //, wheightingType ( initData ( &wheightingType, "wheightingType","Weighting computation method.\n0 - none (distance is used).\n1 - inverse distance square.\n2 - linear.\n3 - hermite (on groups of four dofs).\n4 - spline (on groups of four dofs)." ) )
    //, distanceType ( initData ( &distanceType, "distanceType","Distance computation method.\n0 - euclidian distance.\n1 - geodesic distance.\n2 - harmonic diffusion." ) )
    //, computeWeights ( true )
{
    //maskFrom = NULL;
    //if ( core::behavior::BaseMechanicalState *stateFrom = dynamic_cast< core::behavior::BaseMechanicalState *> ( from ) )
    //    maskFrom = &stateFrom->forceMask;
    //maskTo = NULL;
    //if ( core::behavior::BaseMechanicalState *stateTo = dynamic_cast< core::behavior::BaseMechanicalState *> ( to ) )
    //    maskTo = &stateTo->forceMask;
}

template <class TIn, class TOut>
FrameBlendingMapping<TIn, TOut>::~FrameBlendingMapping ()
{
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::init()
{
    gridMaterial=NULL;
    vector<GridMaterial<materialType> *> vgmat;
    this->getContext()->get<GridMaterial<materialType> >( &vgmat, core::objectmodel::BaseContext::SearchUp);
    if(vgmat.size()!=0)  gridMaterial=vgmat[0];
    if ( !gridMaterial )
    {
        serr << "GridMaterial component not found -> use model vertices as Gauss point and 1/d^2 as weights." << sendl;
    }
    else
    {
        initFrames();
        initSamples();
    }
    updateWeights();
    computeInitPos ();

    Inherit::init();


    //if ( distanceType.getValue().getSelectedId() != SM_DISTANCE_EUCLIDIAN)
    //{
    //    this->getContext()->get ( distOnGrid, core::objectmodel::BaseContext::SearchRoot );
    //    if ( !distOnGrid )
    //    {
    //        serr << "Can not find the DistanceOnGrid component: distances used are euclidian." << sendl;
    //        distanceType.setValue( SM_DISTANCE_EUCLIDIAN);
    //    }
    //}
    //distanceType.beginEdit()->setSelectedItem(SM_DISTANCE_EUCLIDIAN);
    //distanceType.endEdit();
    //const VecInCoord& xfrom = *this->fromModel->getX0();
    //if ( this->initPos.empty() && this->toModel!=NULL && computeWeights==true && weights.getValue().size() ==0 )
    //{
    //    //*
    //    if ( wheightingType.getValue().getSelectedId() == WEIGHT_LINEAR || wheightingType.getValue().getSelectedId() == WEIGHT_HERMITE )
    //        this->nbRefs.setValue ( 2 );

    //    if ( wheightingType.getValue().getSelectedId() == WEIGHT_SPLINE)
    //        this->nbRefs.setValue ( 4 );

    //    if ( xfrom.size() < this->nbRefs.getValue())
    //        this->nbRefs.setValue ( xfrom.size() );
    //    /*/
    //    this->nbRefs.setValue ( xfrom.size() );
    //    //*/

    //    computeDistances();
    //    vector<unsigned int>& m_reps = * ( this->repartition.beginEdit() );
    //    sortReferences ( m_reps);
    //    this->repartition.endEdit();
    //    updateWeights ();
    //    computeInitPos ();
    //}
    //else if ( computeWeights == false || weights.getValue().size() !=0 )
    //{
    //    computeInitPos();
    //}

    Inherit::init();
}





template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::computeInitPos ( )
{
    const VecCoord& xto = ( this->toModel->getX0()->size() == 0)?*this->toModel->getX():*this->toModel->getX0();
    const VecInCoord& xfrom = *this->fromModel->getX0();

    const unsigned int& nbRef = this->nbRefs.getValue();
    const VVUI& m_reps = this->repartition.getValue();

    initPos.resize ( xto.size() * nbRef );
    for ( unsigned int i = 0; i < xto.size(); i++ )
        for ( unsigned int m = 0; m < nbRef; m++ )
        {
            const int& idx=nbRef *i+m;
            const int& idxReps=m_reps[i][m];
            getLocalCoord( initPos[idx], xfrom[idxReps], xto[i]);
        }
}




template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::initFrames()
{
    // Get references
    Data<VecInCoord> &xfrom0_d = *this->fromModel->write(core::VecCoordId::restPosition());
    VecInCoord &xfrom0 = *xfrom0_d.beginEdit();
    Data<VecInCoord> &xfrom_d = *this->fromModel->write(core::VecCoordId::position());
    VecInCoord &xfrom = *xfrom_d.beginEdit();
    Data<VecInCoord> &xfromReset_d = *this->fromModel->write(core::VecCoordId::resetPosition());
    VecInCoord &xfromReset = *xfromReset_d.beginEdit();

    core::behavior::MechanicalState< In >* mstateFrom = dynamic_cast<core::behavior::MechanicalState< In >* >( this->fromModel);
    if ( !mstateFrom)
    {
        serr << "Error: try to insert new frames, which are not mechanical states !" << sendl;
        return;
    }

    // retrieve initial frames
    materialVecCoordType points(xfrom0.size());
    for ( unsigned int i=0; i<xfrom.size(); i++ )
        for ( unsigned int j=0; j<N; j++ )
            points[i][j]= xfrom0[i][j];

    // Insert new frames and compute associated voxel weights
    std::cout<<"Inserting "<<targetFrameNumber.getValue()-xfrom.size()<<" frames..."<<std::endl;
    gridMaterial->computeUniformSampling(points,targetFrameNumber.getValue());
    std::cout<<"Computing weights in grid..."<<std::endl;
    gridMaterial->computeWeights(this->nbRefs.getValue(),points);

    //// copy
    this->fromModel->resize(points.size());
    for ( unsigned int i=xfrom.size(); i<points.size(); i++ )
        for ( unsigned int j=0; j<N; j++ )
            xfrom[i][j] = xfrom0[i][j] = xfromReset[i][j]=  points[i][j];
    xfrom0_d.endEdit();
    xfrom_d.endEdit();
    xfromReset_d.endEdit();
}


template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::initSamples()
{
    // Get references
    Data<VecCoord> &xto0_d = *this->toModel->write(core::VecCoordId::restPosition());
    VecCoord &xto0 = *xto0_d.beginEdit();
    Data<VecCoord> &xto_d = *this->toModel->write(core::VecCoordId::position());
    VecCoord &xto = *xto_d.beginEdit();
    Data<VecCoord> &xtoReset_d = *this->toModel->write(core::VecCoordId::resetPosition());
    VecCoord &xtoReset = *xtoReset_d.beginEdit();

    core::behavior::MechanicalState< Out >* mstateto = dynamic_cast<core::behavior::MechanicalState< Out >* >( this->toModel);
    if ( !mstateto)
    {
        serr << "Error: try to insert new samples, which are not mechanical states !" << sendl;
        return;
    }

    // Insert new samples
    std::cout<<"Inserting "<<targetSampleNumber.getValue()<<" gauss points..."<<std::endl;
    materialVecCoordType points;
    gridMaterial->computeUniformSampling(points,targetSampleNumber.getValue());

    // copy
    this->toModel->resize(points.size());
    for ( unsigned int i=0; i<targetSampleNumber.getValue(); i++ )
        for ( unsigned int j=0; j<N; j++ )
            xto[i][j] = xto0[i][j] = xtoReset[i][j]= points[i][j];
    xto0_d.endEdit();
    xto_d.endEdit();
    xtoReset_d.endEdit();
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::updateWeights ()
{
    std::cout<<"Lumping weights to gauss points..."<<std::endl;

    const VecCoord& xto = ( this->toModel->getX0()->size() == 0)?*this->toModel->getX():*this->toModel->getX0();
    const VecInCoord& xfrom = *this->fromModel->getX0();

    VVD& m_weights = * ( weight.beginEdit() );
    VVSpatialCoord& m_dweight = * ( weightDeriv.beginEdit());
    VVMatNN& m_ddweight = * ( weightDeriv2.beginEdit());

    const unsigned int& nbRef = this->nbRefs.getValue();
    VVUI& m_reps = * ( repartition.beginEdit());

    m_weights.resize ( xto.size() );    for ( unsigned int i=0; i<xto.size(); i++ )        m_weights[i].resize ( nbRef );
    m_dweight.resize ( xto.size() );    for ( unsigned int i=0; i<xto.size(); i++ )        m_dweight[i].resize ( nbRef );
    m_ddweight.resize( xto.size() );    for ( unsigned int i=0; i<xto.size(); i++ )        m_ddweight[i].resize( nbRef );
    m_reps.resize( xto.size() );		for ( unsigned int i=0; i<xto.size(); i++ )        m_reps[i].resize( nbRef );

    unsigned int i,j,k,m;
    if(gridMaterial)
    {
        materialVecCoordType points(xto.size());
        for (i=0; i<xto.size(); i++ ) for (j=0; j<N; j++ )  points[i][j]= xto[i][j];

        VD w;
        VUI reps;
        materialVecCoordType dw;
        materialVecMat33Type ddw;

        for (i=0; i<xto.size(); i++ )
        {
            if(gridMaterial->lumpWeightsRepartition(points[i],reps,w,&dw,&ddw))
            {
                for (j=0; j<nbRef; j++) {	m_weights[i][j]=w[j]; 	m_reps[i][j]=reps[j]; 	for(k=0; k<N; k++) m_dweight[i][j][k]=dw[j][k]; for(k=0; k<N; k++)  for(m=0; m<N; m++) 	m_ddweight[i][j][k][m]=ddw[j][k][m]; }
            }
        }
    }
    else	// 1/d^2 weights with Euclidean distance
    {
        Real w,w2,w3,w4;
        SpatialCoord u;
        for (i=0; i<xto.size(); i++ )
        {
            // get the nbRef closest primitives
            for (j=0; j<nbRef; j++ ) {m_weights[i][j]=0; m_reps[i][j]=0;}
            for (j=0; j<xfrom.size(); j++ )
            {
                w=0; for(k=0; k<N; k++) w+=(xto[i][k]-xfrom[j][k])*(xto[i][k]-xfrom[j][k]); if(w!=0) w=1./w; else w=std::numeric_limits<Real>::max();
                m=0; while (m!=nbRef && m_weights[i][m]>w) m++;
                if(m!=nbRef)
                {
                    for (k=nbRef-1; k>m; k--) {m_weights[i][k]=m_weights[i][k-1]; m_reps[i][k]=m_reps[i][k-1];}
                    m_weights[i][m]=w;
                    m_reps[i][m]=j;
                }
            }
            // compute weight gradients
            for (j=0; j<nbRef; j++ )
            {
                w=m_weights[i][m];
                m_dweight[i][j].fill(0);
                m_ddweight[i][j].fill(0);
                if (w)
                {
                    w2=w*w; w3=w2*w; w4=w3*w;
                    for(k=0; k<N; k++) u[k]=(xto[i][k]-xfrom[j][k]);
                    m_dweight[i][j] = - u * w2* 2.0;
                    for(k=0; k<N; k++) m_ddweight[i][j][k][k]= - w2* 2.0;
                    for(k=0; k<N; k++) for(m=0; m<N; m++) m_ddweight[i][j][k][m]+=u[k]*u[m]*w3* 8.0;
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
    const unsigned int& nbRef = this->nbRefs.getValue();
    VVD& m_weights = * ( weight.beginEdit() );
    VVSpatialCoord& m_dweight = * ( weightDeriv.beginEdit());
    VVMatNN& m_ddweight = * ( weightDeriv2.beginEdit());

    for (unsigned int i = 0; i < xtoSize; ++i)
    {
        Real sumWeights = 0,wn;
        SpatialCoord sumGrad,dwn;			sumGrad.fill(0);
        MatNN sumGrad2,ddwn;				sumGrad2.fill(0);

        // Compute norm
        for (unsigned int j = 0; j < nbRef; ++j)
        {
            sumWeights += m_weights[i][j];
            sumGrad += m_dweight[i][j];
            sumGrad2 += m_ddweight[i][j];
        }

        // Normalise
        if(sumWeights!=0)
            for (unsigned int j = 0; j < nbRef; ++j)
            {
                wn=m_weights[i][j]/sumWeights;
                dwn=(m_dweight[i][j] - sumGrad*wn)/sumWeights;
                for(unsigned int o=0; o<N; o++) for(unsigned int p=0; p<N; p++) ddwn[o][p]=(m_ddweight[i][j][o][p] - wn*sumGrad2[o][p] - sumGrad[o]*dwn[p] - sumGrad[p]*dwn[o])/sumWeights;
                m_ddweight[i][j]=ddwn;
                m_dweight[i][j]=dwn;
                m_weights[i][j] =wn;
            }
    }
}




//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::computeDistances ()
//{
//#ifdef SOFA_DEV
//    if ( this->computeAllMatrices.getValue() && distanceType.getValue().getSelectedId() != SM_DISTANCE_EUCLIDIAN)
//    {
//        const VecInCoord& xfrom0 = *this->fromModel->getX0();
//
//        GeoVecCoord tmpFrom;
//        tmpFrom.resize ( xfrom0.size() );
//        for ( unsigned int i = 0; i < xfrom0.size(); i++ )
//            tmpFrom[i] = xfrom0[i].getCenter();
//
//        if (this->computeAllMatrices.getValue())
//        {
//            sofa::helper::OptionsGroup* distOnGridanceTypeOption = distOnGrid->distanceType.beginEdit();
//            if ( distanceType.getValue().getSelectedId() == SM_DISTANCE_GEODESIC) distOnGridanceTypeOption->setSelectedItem(TYPE_GEODESIC);
//            if ( distanceType.getValue().getSelectedId() == SM_DISTANCE_HARMONIC) distOnGridanceTypeOption->setSelectedItem(TYPE_HARMONIC);
//            if ( distanceType.getValue().getSelectedId() == SM_DISTANCE_STIFFNESS_DIFFUSION) distOnGridanceTypeOption->setSelectedItem(TYPE_STIFFNESS_DIFFUSION);
//            if ( distanceType.getValue().getSelectedId() == SM_DISTANCE_HARMONIC_STIFFNESS) distOnGridanceTypeOption->setSelectedItem(TYPE_HARMONIC_STIFFNESS);
//            distOnGrid->distanceType.endEdit();
//        }
//        distOnGrid->computeDistanceMap ( tmpFrom );
//    }
//#endif
//
//    this->getDistances( 0);
//}
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::getDistances( int xfromBegin)
//{
//    const VecCoord& xto0 = ( this->toModel->getX0()->size() == 0)?*this->toModel->getX():*this->toModel->getX0();
//    const VecInCoord& xfrom0 = *this->fromModel->getX0();
//
//    switch ( distanceType.getValue().getSelectedId() )
//    {
//    case SM_DISTANCE_EUCLIDIAN:
//    {
//        const unsigned int& toSize = xto0.size();
//        const unsigned int& fromSize = xfrom0.size();
//        distances.resize( fromSize);
//        distGradients.resize( fromSize);
//        for ( unsigned int i = xfromBegin; i < fromSize; ++i ) // for each new frame
//        {
//            distances[i].resize (toSize);
//            distGradients[i].resize (toSize);
//            for ( unsigned int j=0;j<toSize;++j )
//            {
//                distGradients[i][j] = xto0[j] - xfrom0[i].getCenter();
//                distances[i][j] = distGradients[i][j].norm();
//                distGradients[i][j].normalize();
//            }
//        }
//        break;
//    }
//#ifdef SOFA_DEV
//    case SM_DISTANCE_GEODESIC:
//    case SM_DISTANCE_HARMONIC:
//    case SM_DISTANCE_STIFFNESS_DIFFUSION:
//    case SM_DISTANCE_HARMONIC_STIFFNESS:
//    {
//        GeoVecCoord goals;
//        goals.resize ( xto0.size() );
//        for ( unsigned int j = 0; j < xto0.size(); ++j )
//            goals[j] = xto0[j];
//        distOnGrid->getDistances ( distances, distGradients, goals );
//        break;
//    }
//#endif
//    default: {}
//    }
//}
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::sortReferences( vector<unsigned int>& references)
//{
//    const unsigned int& toSize = this->toModel->getX()->size();
//    const unsigned int& fromSize = this->fromModel->getX0()->size();
//    const unsigned int& nbRef = this->nbRefs.getValue();
//
//    references.clear();
//    references.resize (nbRef*toSize);
//    for ( unsigned int i=0;i< nbRef *toSize;i++ )
//        references[i] = -1;
//
//    for ( unsigned int i=0;i<fromSize;i++ )
//        for ( unsigned int j=0;j<toSize;j++ )
//            for ( unsigned int k=0; k<nbRef; k++ )
//            {
//                const int idxReps=references[nbRef*j+k];
//                if ( ( idxReps == -1 ) || ( distances[i][j] < distances[idxReps][j] ) )
//                {
//                    for ( unsigned int m=nbRef-1 ; m>k ; m-- )
//                        references[nbRef *j+m] = references[nbRef *j+m-1];
//                    references[nbRef *j+k] = i;
//                    break;
//                }
//            }
//}
//
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::normalizeWeights()
//{
//    const unsigned int& xtoSize = this->toModel->getX()->size();
//    const unsigned int& nbRef = this->nbRefs.getValue();
//    VVD& m_weights = * ( weights.beginEdit() );
//    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());
//    VVMat33& m_ddweight = this->weightGradients2;
//
//    // Normalise weights & dweights
//    for (unsigned int i = 0; i < xtoSize; ++i)
//    {
//        double sumWeights = 0,wn;
//        Vec3 sumGrad,dwn;
//        sumGrad.fill(0);
//        Mat33 sumGrad2,ddwn;
//        sumGrad2.fill(0);
//
//        // Compute norm
//        for (unsigned int j = 0; j < nbRef; ++j)
//        {
//            sumWeights += m_weights[i][j];
//            sumGrad += m_dweight[i][j];
//            sumGrad2 += m_ddweight[i][j];
//        }
//
//        // Normalise
//        if (sumWeights!=0)
//            for (unsigned int j = 0; j < nbRef; ++j)
//            {
//                wn=m_weights[i][j]/sumWeights;
//                dwn=(m_dweight[i][j] - sumGrad*wn)/sumWeights;
//                for (unsigned int o=0;o<3;o++) for (unsigned int p=0;p<3;p++) ddwn[o][p]=(m_ddweight[i][j][o][p] - wn*sumGrad2[o][p] - sumGrad[o]*dwn[p] - sumGrad[p]*dwn[o])/sumWeights;
//                m_ddweight[i][j]=ddwn;
//                m_dweight[i][j]=dwn;
//                m_weights[i][j] =wn;
//            }
//    }
//}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::draw()
{
    const typename Out::VecCoord& xto = *this->toModel->getX();
    const typename In::VecCoord& xfrom = *this->fromModel->getX();
    const unsigned int nbRef = this->nbRefs.getValue();
    const VVUI& m_reps = this->repartition.getValue();
    const VVD& m_weights = weight.getValue();
    const VVSpatialCoord& m_dweights = weightDeriv.getValue();
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

        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            for ( unsigned int m=0 ; m<nbRef; m++ )
            {
                const int idxReps=m_reps[i][m];
                double coef = m_weights[i][m];
                if ( coef > 0.0 )
                {
                    glColor4d ( coef,coef,0,1 );
                    helper::gl::glVertexT ( xfrom[idxReps].getCenter() );
                    helper::gl::glVertexT ( xto[i] );
                }
            }
        }
        glEnd();
    }

    // Display  m_reps for each points
    if ( showReps.getValue())
    {
        for ( unsigned int i=0; i<xto.size(); i++ )
            sofa::helper::gl::GlText::draw ( m_reps[i][0]*scale, xto[i], textScale );
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
            findIndexInRepartition(influenced, refIndex, i, showFromIndex.getValue()%nbRefs.getValue());
            if ( influenced)
            {
                const SpatialCoord& grad = m_dweights[i][refIndex];
                sprintf( txt, "( %i, %i, %i)", (int)(grad[0]*scale), (int)(grad[1]*scale), (int)(grad[2]*scale));
                sofa::helper::gl::GlText::draw ( txt, xto[i], textScale );
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
            findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRefs.getValue());
            if ( influenced)
            {
                sofa::helper::gl::GlText::draw ( (int)(m_weights[i][indexRep]*scale), xto[i], textScale );
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
            findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRefs.getValue());
            if (influenced)
            {
                const SpatialCoord& gradMap = m_dweights[i][indexRep];
                const SpatialCoord& point = xto[i];
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
        Real minValue = std::numeric_limits<Real>::max();
        Real maxValue = -std::numeric_limits<Real>::min();
        for ( unsigned int i = 0; i < xto.size(); i++)
        {
            bool influenced;
            unsigned int indexRep;
            findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRefs.getValue());
            if (influenced)
            {
                const Real& weight = m_weights[i][indexRep];
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
                    findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRefs.getValue());
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
                findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRefs.getValue());
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

    // Display def tensor values for each points
    if ( this->showDefTensorsValues.getValue())
    {
        char txt[100];
        glColor3f( 0.5, 0.5, 0.5);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            const Vec6& e = this->deformationTensors[i];
            sprintf( txt, "( %i, %i, %i)", (int)(e[0]*scale), (int)(e[1]*scale), (int)(e[2]*scale));
            sofa::helper::gl::GlText::draw ( txt, xto[i], textScale );
        }
    }

    // Deformation tensor show
    if ( this->showDefTensors.getValue() && this->computeAllMatrices.getValue() )
    {
        TriangleSetTopologyContainer *mesh;
        this->getContext()->get( mesh);
        if ( mesh)
        {
            glPushAttrib( GL_LIGHTING_BIT || GL_COLOR_BUFFER_BIT || GL_ENABLE_BIT);
            glDisable( GL_LIGHTING);
            glBegin( GL_TRIANGLES);
            const TriangleSetTopologyContainer::SeqTriangles& tri = mesh->getTriangles();
            for ( unsigned int i = 0; i < mesh->getNumberOfTriangles(); i++)
            {
                for ( unsigned int j = 0; j < 3; j++)
                {
                    const Vec6& e = this->deformationTensors[tri[i][j]];
                    float color = 0.5 + ( e[0] + e[1] + e[2])/this->showDefTensorScale.getValue();
                    glColor3f( 0.0, color, 1.0-color);// /*e[0]*/, e[1], e[2]);
                    glVertex3f( xto[tri[i][j]][0], xto[tri[i][j]][1], xto[tri[i][j]][2]);
                }
            }
            glEnd();
            glPopAttrib();
        }
        else // Show by points
        {
            glPointSize( 10);
            glBegin( GL_POINTS);
            for ( unsigned int i = 0; i < xto.size(); i++)
            {
                const Vec6& e = this->deformationTensors[i];
                float mult=500;
                float color = (e[0]+e[1]+e[2])/3.;
                if (color<0) color=2*color/(color+1.);
                color*=mult;
                color+=120;
                if (color<0) color=0;
                if (color>240) color=240;
                sofa::helper::gl::Color::setHSVA(color,1.,.8,1.);
                glVertex3f( xto[i][0], xto[i][1], xto[i][2]);
            }
            glEnd();
        }
    }
}


template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::clear()
{
    this->initPos.clear();
}
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::setWeightsToHermite()
//{
//    wheightingType.beginEdit()->setSelectedItem(WEIGHT_HERMITE);
//    wheightingType.endEdit();
//}
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::setWeightsToLinear()
//{
//    wheightingType.beginEdit()->setSelectedItem(WEIGHT_LINEAR);
//    wheightingType.endEdit();
//}
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::setWeightsToInvDist()
//{
//    wheightingType.beginEdit()->setSelectedItem(WEIGHT_INVDIST_SQUARE);
//    wheightingType.endEdit();
//}
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::updateWeights ()
//{
//    const VecCoord& xto = ( this->toModel->getX0()->size() == 0)?*this->toModel->getX():*this->toModel->getX0();
//    const VecInCoord& xfrom = *this->fromModel->getX0();
//
//    VVD& m_weights = * ( weights.beginEdit() );
//    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());
//    VVMat33& m_ddweight = this->weightGradients2;
//
//    const unsigned int& nbRef = this->nbRefs.getValue();
//    const vector<unsigned int>& m_reps = this->repartition.getValue();
//
//    m_weights.resize ( xto.size() );
//    for ( unsigned int i=0;i<xto.size();i++ )        m_weights[i].resize ( nbRef );
//    m_dweight.resize ( xto.size() );
//    for ( unsigned int i=0;i<xto.size();i++ )        m_dweight[i].resize ( nbRef );
//    m_ddweight.resize( xto.size() );
//    for ( unsigned int i=0;i<xto.size();i++ )        m_ddweight[i].resize( nbRef );
//
//    switch ( wheightingType.getValue().getSelectedId() )
//    {
//    case WEIGHT_NONE:
//    {
//        for ( unsigned int i=0;i<xto.size();i++ )
//            for ( unsigned int j=0;j<nbRef;j++ )
//            {
//                int indexFrom = m_reps[nbRef*i+j];
//                if ( distanceType.getValue().getSelectedId()  == SM_DISTANCE_HARMONIC)
//                {
//                    m_weights[indexFrom][i] = distOnGrid->harmonicMaxValue.getValue() - distances[indexFrom][i];
//                    if ( distances[indexFrom][i] < 0.0) distances[indexFrom][i] = 0.0;
//                    if ( distances[indexFrom][i] > distOnGrid->harmonicMaxValue.getValue()) distances[indexFrom][i] = distOnGrid->harmonicMaxValue.getValue();
//                    if (distances[indexFrom][i]==0 || distances[indexFrom][i]==distOnGrid->harmonicMaxValue.getValue()) m_dweight[indexFrom][i]=Coord();
//                    else m_dweight[indexFrom][i] = - distGradients[indexFrom][i];
//                }
//                else
//                {
//                    m_weights[i][j] = distances[indexFrom][i];
//                    m_dweight[i][j] = distGradients[indexFrom][i];
//                }
//            }
//        break;
//    }
//    case WEIGHT_LINEAR:
//    {
//        vector<unsigned int> tmpReps;
//        sortReferences( tmpReps);
//        for ( unsigned int i=0;i<xto.size();i++ )
//        {
//            for ( unsigned int j=0;j<nbRef;j++ )
//            {
//                m_weights[i][j] = 0.0;
//                m_dweight[i][j] = GeoCoord();
//                m_ddweight[i][j].fill(0);
//            }
//            Vec3d r1r2, r1p;
//            r1r2 = xfrom[tmpReps[nbRef *i+1]].getCenter() - xfrom[tmpReps[nbRef *i+0]].getCenter();
//            r1p  = xto[i] - xfrom[tmpReps[nbRef *i+0]].getCenter();
//            double r1r2NormSquare = r1r2.norm()*r1r2.norm();
//            double wi = ( r1r2*r1p ) / ( r1r2NormSquare);
//
//            // Abscisse curviligne
//            m_weights[i][0] = ( 1 - wi );
//            m_weights[i][1] = wi;
//            m_dweight[i][0] = -r1r2 / r1r2NormSquare;
//            m_dweight[i][1] = r1r2 / r1r2NormSquare;
//            //m_ddweight[i][0] = ;
//            //m_ddweight[i][1] = ;
//        }
//        break;
//    }
//    case WEIGHT_INVDIST_SQUARE:
//    {
//        for ( unsigned int i=0;i<xto.size();i++ )
//        {
//            for ( unsigned int j=0;j<nbRef;j++ )
//            {
//                int indexFrom = m_reps[nbRef*i+j];
//                double d2=distances[indexFrom][i]*distances[indexFrom][i];
//                double d3=d2*distances[indexFrom][i];
//                double d4=d3*distances[indexFrom][i];
//
//                m_ddweight[i][j].fill(0);
//                if (d2)
//                {
//                    m_weights[i][j] = 1 / d2;
//                    m_dweight[i][j] = - distGradients[indexFrom][i] / d3* 2.0;
//                    m_ddweight[i][j][0][0]-=2.0/d4;
//                    m_ddweight[i][j][1][1]-=2.0/d4;
//                    m_ddweight[i][j][2][2]-=2.0/d4;
//                    for (unsigned int k=0;k<3;k++)
//                        for (unsigned int m=0;m<3;m++)
//                            m_ddweight[i][j][k][m]+=distGradients[indexFrom][i][k]*distGradients[indexFrom][i][m]*8.0/d2;
//                }
//                else
//                {
//                    m_weights[i][j] = 0xFFF;
//                    m_dweight[i][j] = GeoCoord();
//                }
//
//            }
//        }
//
//        break;
//    }
//    case WEIGHT_HERMITE:
//    {
//        vector<unsigned int> tmpReps;
//        sortReferences( tmpReps);
//        for ( unsigned int i=0;i<xto.size();i++ )
//        {
//            for ( unsigned int j=0;j<nbRef;j++ )
//            {
//                m_weights[i][j] = 0.0;
//                m_dweight[i][j] = GeoCoord();
//            }
//            Vec3d r1r2, r1p;
//            double wi;
//            r1r2 = xfrom[tmpReps[nbRef *i+1]].getCenter() - xfrom[tmpReps[nbRef *i+0]].getCenter();
//            r1p  = xto[i] - xfrom[tmpReps[nbRef *i+0]].getCenter();
//            double r1r2NormSquare = r1r2.norm()*r1r2.norm();
//            wi = ( r1r2*r1p ) / r1r2NormSquare;
//
//            // Fonctions d'Hermite
//            m_weights[i][0] = 1-3*wi*wi+2*wi*wi*wi;
//            m_weights[i][1] = 3*wi*wi-2*wi*wi*wi;
//
//            r1r2.normalize();
//            m_dweight[i][0] = -r1r2 * (6*wi-6*wi*wi) / (r1r2NormSquare);
//            m_dweight[i][1] = r1r2 * (6*wi-6*wi*wi) / (r1r2NormSquare);
//        }
//        break;
//    }
//    case WEIGHT_SPLINE:
//    {
//        if ( xfrom.size() < 4 || nbRef < 4)
//        {
//            serr << "Error ! To use WEIGHT_SPLINE, you must use at least 4 DOFs and set nbRefs to 4.\n WEIGHT_SPLINE requires also the DOFs are ordered along z-axis." << sendl;
//            return;
//        }
//        vector<unsigned int> tmpReps;
//        sortReferences( tmpReps);
//        for ( unsigned int i=0;i<xto.size();i++ )
//        {
//            // Clear all weights and dweights.
//            for ( unsigned int j=0;j<nbRef;j++ )
//            {
//                m_weights[i][j] = 0.0;
//                m_dweight[i][j] = GeoCoord();
//            }
//            // Get the 4 nearest DOFs.
//            vector<unsigned int> sortedFrames;
//            for ( unsigned int j = 0; j < 4; ++j)
//                sortedFrames.push_back( tmpReps[nbRef *i+j]);
//            std::sort( sortedFrames.begin(), sortedFrames.end());
//
//            if ( xto[i][2] < xfrom[sortedFrames[1]].getCenter()[2])
//            {
//                vector<unsigned int> sortedFramesCpy = sortedFrames;
//                sortedFrames.clear();
//                sortedFrames.push_back( sortedFramesCpy[0]);
//                sortedFrames.push_back( sortedFramesCpy[0]);
//                sortedFrames.push_back( sortedFramesCpy[1]);
//                sortedFrames.push_back( sortedFramesCpy[2]);
//            }
//            else if ( xto[i][2] > xfrom[sortedFrames[2]].getCenter()[2])
//            {
//                vector<unsigned int> sortedFramesCpy = sortedFrames;
//                sortedFrames.clear();
//                sortedFrames.push_back( sortedFramesCpy[1]);
//                sortedFrames.push_back( sortedFramesCpy[2]);
//                sortedFrames.push_back( sortedFramesCpy[3]);
//                sortedFrames.push_back( sortedFramesCpy[3]);
//            }
//
//            // Compute u
//            Vec3d r1r2 = xfrom[sortedFrames[2]].getCenter() - xfrom[sortedFrames[1]].getCenter();
//            Vec3d r1p  = xto[i] - xfrom[sortedFrames[1]].getCenter();
//            double r1r2NormSquare = r1r2.norm()*r1r2.norm();
//            double u = ( r1r2*r1p ) / r1r2NormSquare;
//
//            // Set weights and dweights.
//            m_weights[i][0] = 1-3*u*u+2*u*u*u;
//            m_weights[i][1] = u*u*u - 2*u*u + u;
//            m_weights[i][2] = 3*u*u-2*u*u*u;
//            m_weights[i][3] = u*u*u - u*u;
//
//            r1r2.normalize();
//            m_dweight[i][0] = -r1r2 * (6*u - 6*u*u) / (r1r2NormSquare);
//            m_dweight[i][1] = r1r2 * (3*u*u - 4*u + 1) / (r1r2NormSquare);
//            m_dweight[i][2] = r1r2 * (6*u - 6*u*u) / (r1r2NormSquare);
//            m_dweight[i][3] = r1r2 * (3*u*u - 2*u) / (r1r2NormSquare);
//        }
//        break;
//    }
//    default:
//    {}
//    }
//
//    normalizeWeights();
//}
//
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::setWeightCoefs ( VVD &weights )
//{
//    VVD * m_weights = this->weights.beginEdit();
//    m_weights->clear();
//    m_weights->insert ( m_weights->begin(), weights.begin(), weights.end() );
//    this->weights.endEdit();
//}
//
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::setRepartition ( vector<int> &rep )
//{
//    vector<unsigned int> * m_reps = this->repartition.beginEdit();
//    m_reps->clear();
//    m_reps->insert ( m_reps->begin(), rep.begin(), rep.end() );;
//    this->repartition.endEdit();
//}


template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::findIndexInRepartition( bool& influenced, unsigned int& realIndex, const unsigned int& pointIndex, const unsigned int& frameIndex)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const VVUI& m_reps = this->repartition.getValue();
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
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::removeFrame( const unsigned int /*index*/)
//{
//    //VecCoord& xto0 = *this->toModel->getX0();
//
//    Data<VecCoord> &xto_d = *this->toModel->write(core::VecCoordId::position());
//    VecCoord& xto = *xto_d.beginEdit();
//
//    //VecInCoord& xfrom0 = *this->fromModel->getX0();
//    const VecInCoord& xfrom = *this->fromModel->getX();
//
//    // this->T.erase( T.begin()+index);
//    // coeffs
//
//    // Recompute matrices
//    apply( xto, xfrom);
//
//    xto_d.endEdit();
//}
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::insertFrame( const Coord& pos, const Quat& rot, GeoVecCoord beginPointSet, double distMax)
//{/*
//                changeSettingsDueToInsertion();
//
//                if (!this->toModel->getX0()) return;
//                // Get references
//                Data<VecCoord> &xto_d = *this->toModel->write(core::VecCoordId::position());
//                VecCoord& xto = *xto_d.beginEdit();
//                Data<VecInCoord> &xfrom0_d = *this->fromModel->write(core::VecCoordId::restPosition());
//                VecInCoord &xfrom0 = *xfrom0_d.beginEdit();
//                Data<VecInCoord> &xfrom_d = *this->fromModel->write(core::VecCoordId::position());
//                VecInCoord &xfrom = *xfrom_d.beginEdit();
//                Data<VecInCoord> &xfromReset_d = *this->fromModel->write(core::VecCoordId::resetPosition());
//                VecInCoord &xfromReset = *xfromReset_d.beginEdit();
//
//                unsigned int indexFrom = xfrom.size();
//                core::behavior::MechanicalState< In >* mstateFrom = dynamic_cast<core::behavior::MechanicalState< In >* >( this->fromModel);
//                if ( !mstateFrom) {
//                    serr << "Error: try to insert a new frame on fromModel, which is not a mechanical state !" << sendl;
//                    return;
//                }
//
//                // Compute the rest position of the frame.
//                InCoord newX, newX0;
//                InCoord targetDOF;
//                setInCoord( targetDOF, pos, rot);
//                inverseSkinning( newX0, newX, targetDOF);
//
//                // Insert a new DOF
//                this->fromModel->resize( indexFrom + 1);
//                xfrom[indexFrom] = newX;
//                xfrom0[indexFrom] = newX0;
//                xfromReset[indexFrom] = newX0;
//
//                if ( distMax == 0.0)
//                    distMax = this->newFrameDefaultCutOffDistance.getValue();
//
//                // Compute geodesical/euclidian distance for this frame.
//                if ( this->distanceType.getValue().getSelectedId() == SM_DISTANCE_GEODESIC || this->distanceType.getValue().getSelectedId() == SM_DISTANCE_HARMONIC || this->distanceType.getValue().getSelectedId() == SM_DISTANCE_STIFFNESS_DIFFUSION || this->distanceType.getValue().getSelectedId() == TYPE_HARMONIC_STIFFNESS)
//                    this->distOnGrid->addElt( newX0.getCenter(), beginPointSet, distMax);
//                vector<double>& vRadius = (*this->newFrameWeightingRadius.beginEdit());
//                vRadius.resize( indexFrom + 1);
//                vRadius[indexFrom] = distMax;
//                this->newFrameWeightingRadius.endEdit();
//
//                // Recompute matrices
//                apply( xto, xfrom);
//
//                xto_d.endEdit();
//                xfrom0_d.endEdit();
//                xfrom_d.endEdit();
//                xfromReset_d.endEdit();*/
//}
//
//template <class TIn, class TOut>
//bool FrameBlendingMapping<TIn, TOut>::inverseSkinning( InCoord& /*X0*/, InCoord& /*X*/, const InCoord& /*Xtarget*/)
//// compute position in reference configuration from deformed position
//{
//    /*  TODO !!
//    const VecInCoord& xi = *this->fromModel->getX();
//    const VMat88& T = this->T;
//    //const VecInCoord& xi0 = *this->fromModel->getX0();
//    const VecCoord& P0 = *this->toModel->getX0();
//    const VecCoord& P = *this->toModel->getX();
//
//    int i,j,k,l,nbP=P0.size(),nbDOF=xi.size();
//    VDUALQUAT qrel(nbDOF);
//    DUALQUAT q,b,bn;
//    Vec3 t;
//    Vec4 qinv;
//    Mat33 R,U,Uinv;
//    Mat88 N;
//    Mat38 Q;
//    Mat83 W,NW;
//    double QEQ0,Q0Q0,Q0,d,dmin=1E5;
//    Mat44 q0q0T,q0qeT,qeq0T;
//    VVD w;
//    w.resize(nbDOF);
//    for ( int i = 0; i < nbDOF; i++) w[i].resize(P.size());
//    VecVecCoord dw(nbDOF);
//    for ( int i = 0; i < nbDOF; i++) dw[i].resize(P.size());
//    X.getOrientation() = Xtarget.getOrientation();
//
//    // init skinning
//    for (i=0;i<nbDOF;i++)
//    {
//    XItoQ( q, xi[i]); //update DOF quats
//    computeQrel( qrel[i], T[i], q); //update qrel=Tq
//    }
//
//    // get closest material point
//    for (i=0;i<nbP;i++)
//    {
//    t = Xtarget.getCenter() - P[i];
//    d = t * t;
//    if (d<dmin) {
//    dmin = d;
//    X0.getCenter() = P0[i];
//    }
//    }
//    if (dmin==1E5) return false;
//
//    // iterate: pos0(t+1)=pos0(t) + (dPos/dPos0)^-1 (Pos - Pos(t))
//    double eps=1E-5;
//    bool stop=false;
//    int count=0;
//
//    while (!stop)
//    {
//    // update weigths
//    computeWeight( w, dw, X0.getCenter());
//
//    // update skinning
//    BlendDualQuat( b, bn, QEQ0, Q0Q0, Q0, 0, qrel, w); //skinning: sum(wTq)
//    computeDqRigid( R, t, bn); //update Rigid
//
//    qinv[0]=-bn.q0[0];
//    qinv[1]=-bn.q0[1];
//    qinv[2]=-bn.q0[2];
//    qinv[3]=bn.q0[3];
//    Multi_Q( X0.getOrientation(), qinv, Xtarget.getOrientation());
//    for (k=0;k<3;k++) {
//    X.getCenter()[k] = t[k];
//    for (j=0;j<3;j++) X.getCenter()[k] += R[k][j] * X0.getCenter()[j];
//    }
//    //update skinned points
//
//    t = Xtarget.getCenter()- X.getCenter();
//
//    //std::cout<<" "<<t*t;
//
//    if ( t*t < eps || count >= 10) stop = true;
//    count++;
//
//    if (!stop)
//    {
//    computeDqN_constants( q0q0T, q0qeT, qeq0T, bn);
//    computeDqN( N, q0q0T, q0qeT, qeq0T, QEQ0, Q0Q0, Q0); //update N=d(bn)/d(b)
//    computeDqQ( Q, bn, X0.getCenter()); //update Q=d(P)/d(bn)
//    W.fill(0);
//    for (j=0;j<nbDOF;j++) for (k=0;k<4;k++) for (l=0;l<3;l++)
//        {
//            W[k][l]+=dw[j][0][l]*qrel[j].q0[k];
//            W[k+4][l]+=dw[j][0][l]*qrel[j].qe[k];
//        }
//    //update W=sum(wrel.dw)=d(b)/d(p)
//    NW=N*W; // update NW
//    // grad def = d(P)/d(p0)
//    U=R+Q*NW; // update A=R+QNW
//    invertMatrix(Uinv,U);
//
//    // Update pos0
//    X0.getCenter() += Uinv * t;
//    }
//    }
//    //std::cout<<"err:"<<t*t;
//    */
//    return true;
//}
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::computeWeight( VVD& w, VecVecCoord& dw, const Coord& x0)
//{
//    const VecCoord& xto0 = *this->toModel->getX0();
//    const VecInCoord& xfrom0 = *this->fromModel->getX0();
//    // Get Distances
//    VVD dist;
//    GeoVecVecCoord ddist;
//
//    switch ( this->distanceType.getValue().getSelectedId()  )
//    {
//    case SM_DISTANCE_EUCLIDIAN:
//    {
//        dist.resize( xfrom0.size());
//        ddist.resize( xfrom0.size());
//        for ( unsigned int i = 0; i < xfrom0.size(); i++)
//        {
//            dist[i].resize( 1);
//            ddist[i].resize( 1);
//
//            ddist[i][0] = x0 - xfrom0[i].getCenter();
//            dist[i][0] = ddist[i][0].norm();
//            ddist[i][0].normalize();
//        }
//        break;
//    }
//    case SM_DISTANCE_GEODESIC:
//    case SM_DISTANCE_HARMONIC:
//    case SM_DISTANCE_STIFFNESS_DIFFUSION:
//    case SM_DISTANCE_HARMONIC_STIFFNESS:
//    {
//        GeoVecCoord goals;
//        goals.push_back( x0);
//        this->distOnGrid->getDistances ( dist, ddist, goals );
//        break;
//    }
//    default: {}
//    }
//
//    // Compute Weights
//    switch ( this->wheightingType.getValue().getSelectedId()  )
//    {
//    case WEIGHT_NONE:
//    {
//        for ( unsigned int i=0;i<xfrom0.size();i++ )
//        {
//            w[i][0] = dist[i][0];
//            dw[i][0] = ddist[i][0];
//        }
//        break;
//    }
//    case WEIGHT_LINEAR:
//    {
//        for ( unsigned int i=0;i<xfrom0.size();i++ )
//        {
//            Vec3d r1r2, r1p;
//            r1r2 = xfrom0[(i+1)%(xfrom0.size())].getCenter() - xfrom0[i].getCenter();
//            r1p  = xto0[0] - xfrom0[i].getCenter();
//            double r1r2NormSquare = r1r2.norm()*r1r2.norm();
//            double wi = ( r1r2*r1p ) / ( r1r2NormSquare);
//
//            // Abscisse curviligne
//            w[i][0]                   = ( 1 - wi );
//            w[(i+1)%(xfrom0.size())][0] = wi;
//            dw[i][0]                   = -r1r2 / r1r2NormSquare;
//            dw[(i+1)%(xfrom0.size())][0] = r1r2 / r1r2NormSquare;
//        }
//        break;
//    }
//    case WEIGHT_INVDIST_SQUARE:
//    {
//        for ( unsigned int i=0;i<xfrom0.size();i++ )
//        {
//            if ( dist[i][0])
//            {
//                w[i][0] = 1 / (dist[i][0]*dist[i][0]);
//                dw[i][0] = - ddist[i][0] / (dist[i][0]*dist[i][0]*dist[i][0]) * 2.0;
//            }
//            else
//            {
//                w[i][0] = 0xFFF;
//                dw[i][0] = GeoCoord();
//            }
//        }
//        break;
//    }
//    case WEIGHT_HERMITE:
//    {
//        for ( unsigned int i=0;i<xfrom0.size();i++ )
//        {
//            Vec3d r1r2, r1p;
//            double wi;
//            r1r2 = xfrom0[(i+1)%xfrom0.size()].getCenter() - xfrom0[i].getCenter();
//            r1p  = xto0[0] - xfrom0[i].getCenter();
//            wi = ( r1r2*r1p ) / ( r1r2.norm() *r1r2.norm() );
//
//            // Fonctions d'Hermite
//            w[i][0]                   = 1-3*wi*wi+2*wi*wi*wi;
//            w[(i+1)%(xfrom0.size())][0] = 3*wi*wi-2*wi*wi*wi;
//
//            r1r2.normalize();
//            dw[i][0]                   = -r1r2;
//            dw[(i+1)%(xfrom0.size())][0] = r1r2;
//        }
//        break;
//    }
//    default:
//    {}
//    }
//}
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::updateDataAfterInsertion()
//{
//    /* TODO !!!
//    const VecCoord& xto = *this->toModel->getX();
//    const VecInCoord& xfrom0 = *this->fromModel->getX0();
//    const VecInCoord& xfrom = *this->fromModel->getX();
//    VVD& m_weights = * ( this->weights.beginEdit() );
//    SVector<SVector<GeoCoord> >& dw = *(this->weightGradients.beginEdit());
//    vector<double>& radius = (*this->newFrameWeightingRadius.beginEdit());
//
//    changeSettingsDueToInsertion();
//
//    //TODO fix it ! Synchro between many SMapping
//    if ( radius.size() != xfrom.size())
//    {
//    int oldSize = radius.size();
//    radius.resize( xfrom.size());
//    for ( unsigned int i = oldSize; i < xfrom.size(); ++i) radius[i] = this->newFrameDefaultCutOffDistance.getValue();
//    }
//
//    // Resize T
//    unsigned int size = this->T.size();
//    DUALQUAT qi0;
//    this->T.resize ( xfrom.size() );
//
//    // Get distances
//    this->getDistances( size);
//
//    // for each new frame
//    const double& maximizeWeightDist = this->newFrameDistanceToMaximizeWeight.getValue();
//    for ( unsigned int i = size; i < xfrom.size(); ++i )
//    {
//    // Get T
//    XItoQ( qi0, xfrom0[i]);
//    computeDqT ( this->T[i], qi0 );
//
//    // Update weights
//    m_weights.resize ( xfrom.size() );
//    m_weights[i].resize ( xto.size() );
//    dw.resize ( xfrom.size() );
//    dw[i].resize ( xto.size() );
//    for ( unsigned int j = 0; j < xto.size(); ++j )
//    {
//    if ( this->distances[i][j])
//    {
//    if ( this->distances[i][j] == -1.0)
//    {
//        m_weights[i][j] = 0.0;
//        dw[i][j] = GeoCoord();
//    }
//    else
//    {
//        if( maximizeWeightDist != 0.0)
//        {
//          if( this->distances[i][j] < maximizeWeightDist)
//          {
//            m_weights[i][j] = 0xFFF;
//            dw[i][j] = GeoCoord();
//          }
//          else if( this->distances[i][j] > radius[i])
//          {
//            m_weights[i][j] = 0.0;
//            dw[i][j] = GeoCoord();
//          }
//          else
//          {
//            // linear interpolation from 0 to 0xFFF
//            //m_weights[i][j] = 0xFFF / (maximizeWeightDist - radius[i]) * (this->distances[i][j] - radius[i]);
//            //dw[i][j] = this->distGradients[i][j] * 0xFFF / (maximizeWeightDist - radius[i]);
//
//            // Hermite between 0 and 0xFFF
//            double dPrime = (this->distances[i][j] - maximizeWeightDist) / (radius[i]-maximizeWeightDist);
//            m_weights[i][j] = (1-3*dPrime*dPrime+2*dPrime*dPrime*dPrime) * 0xFFF;
//            dw[i][j] = - this->distGradients[i][j] * (6*dPrime-6*dPrime*dPrime) / (radius[i]-maximizeWeightDist ) * 0xFFF;
//          }
//        }
//        else
//        {
//          m_weights[i][j] = 1.0 / (this->distances[i][j]*this->distances[i][j]) - 1.0 / (radius[i]*radius[i]);
//          if (m_weights[i][j] < 0)
//          {
//            m_weights[i][j] = 0.0;
//            dw[i][j] = GeoCoord();
//          }
//          else
//            dw[i][j] = - this->distGradients[i][j] / (this->distances[i][j]*this->distances[i][j]*this->distances[i][j]) * 2.0;
//        }
//    }
//    }
//    else
//    {
//    m_weights[i][j] = 0xFFF;
//    dw[i][j] = GeoCoord();
//    }
//    }
//    }
//    this->weights.endEdit();
//    this->weightGradients.endEdit();
//    this->newFrameWeightingRadius.endEdit();
//    */
//}
//
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::changeSettingsDueToInsertion()
//{
//    this->setWeightsToInvDist();
//    if ( this->distOnGrid)
//    {
//        this->distanceType.beginEdit()->setSelectedItem(SM_DISTANCE_GEODESIC);
//    }
//    else
//    {
//        this->distanceType.beginEdit()->setSelectedItem(SM_DISTANCE_EUCLIDIAN);
//    }
//}
//
//
//
//

template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::M33toV6(Vec6 &v,const Mat33& M) const
// referred as operator V in the paper
{
    v[0]=M[0][0];
    v[1]=M[1][1];
    v[2]=M[2][2];
    v[3]=(M[0][1]+M[1][0])/2.;
    v[4]=(M[2][1]+M[1][2])/2.;
    v[5]=(M[2][0]+M[0][2])/2.;
}

//
template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::resizeMatrices()
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const unsigned int& toSize = this->toModel->getX()->size();

    this->det.resize(toSize);
    this->deformationTensors.resize(toSize);
    this->Atilde.resize(toSize);
    for (unsigned int i = 0; i < toSize; ++i)      this->Atilde[i].resize(nbRef);
    this->ddet.resize(toSize);
    for (unsigned int i = 0; i < toSize; ++i)      this->ddet[i].resize(nbRef);
    this->B.resize(toSize);
    for (unsigned int i = 0; i < toSize; ++i)      this->B[i].resize(nbRef);
    this->J0.resize (toSize);
    for (unsigned int i = 0; i < toSize; ++i)      this->J0[i].resize(nbRef);
    this->J.resize(toSize);
    for (unsigned int i = 0; i < toSize; ++i)      this->J[i].resize(nbRef);
}

//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::initSamples() // Temporary (will be done in FrameSampler class)
//{
//    const unsigned int& nbRef = this->nbRefs.getValue();
//    // vol and massDensity
//    sofa::component::topology::DynamicSparseGridTopologyContainer* hexaContainer;
//    this->getContext()->get( hexaContainer);
//    //TODO get the volume from the FrameSampler !
//    double volume = this->voxelVolume.getValue();
//    //if ( hexaContainer && this->distOnGrid) volume = this->distOnGrid->initTargetStep.getValue()*this->distOnGrid->initTargetStep.getValue()*this->distOnGrid->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[0]*hexaContainer->voxelSize.getValue()[1]*hexaContainer->voxelSize.getValue()[2];
//    const VecCoord& xto = *this->toModel->getX();
//    const unsigned int& toSize = xto.size();
//    this->vol.resize( toSize);
//    for ( unsigned int i = 0; i < toSize; i++) this->vol[i] = volume;
//    this->massDensity.resize( toSize);
//    for ( unsigned int i = 0; i < toSize; i++) this->massDensity[i] = 1.0;
//
//    if ( useElastons.getValue())
//    {
//        this->integ_Elaston.resize( toSize);
//        for ( unsigned int i = 0; i < toSize; i++) // to update
//        {
//            double lx,ly,lz;
//            if ( hexaContainer && this->distOnGrid)
//            {
//                lx=volume;
//                ly=volume; //TODO get the volume from the FrameSampler !
//                lz=volume;
//            }
//            else lx=ly=lz=pow(this->voxelVolume.getValue(),1./3.);
//            this->integ_Elaston[i][0] = 1;
//            this->integ_Elaston[i][4] = lx*lx/12.;
//            this->integ_Elaston[i][7] = ly*ly/12.;
//            this->integ_Elaston[i][9] = lz*lz/12.;
//            this->integ_Elaston[i][20] = lx*lx*lx*lx/80.;
//            this->integ_Elaston[i][21] = lx*lx*ly*ly/144.;
//            this->integ_Elaston[i][22] = lx*lx*lz*lz/144.;
//            this->integ_Elaston[i][23] = ly*ly*ly*ly/80.;
//            this->integ_Elaston[i][24] = ly*ly*lz*lz/144.;
//            this->integ_Elaston[i][25] = lz*lz*lz*lz/80.;
//            this->integ_Elaston[i]=this->integ_Elaston[i]*lx*ly*lz;
//        }
//
//        this->deformationTensorsElaston.resize(toSize);
//        this->Stilde_x.resize(toSize);
//        this->Stilde_y.resize(toSize);
//        this->Stilde_z.resize(toSize);
//        for (unsigned int i = 0; i < toSize; ++i)
//        {
//            this->Stilde_x[i].resize(nbRef);
//            this->Stilde_y[i].resize(nbRef);
//            this->Stilde_z[i].resize(nbRef);
//        }
//        this->B_Elaston.resize(toSize);
//        for (unsigned int i = 0; i < toSize; ++i)      this->B_Elaston[i].resize(nbRef);
//    }
//}

template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::getLocalCoord( Coord& result, const typename In::Coord& inCoord, const Coord& coord) const
{
    result = inCoord.pointToChild ( coord );
}
//
//
//template <class TIn, class TOut>
//void FrameBlendingMapping<TIn, TOut>::setInCoord( InCoord& coord, const Coord& position, const Quat& rotation) const
//{
//    coord.getCenter() = position;
//    coord.getOrientation() = rotation;
//}


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
