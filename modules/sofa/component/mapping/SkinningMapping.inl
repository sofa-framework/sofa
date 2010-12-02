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
#ifndef SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_INL
#define SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_INL

#include <sofa/component/mapping/SkinningMapping.h>

#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <sofa/core/Mapping.inl>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/gl/glText.inl>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/io/Mesh.h>

#ifdef SOFA_DEV
#include <sofa/helper/DualQuat.h>
#include <sofa/component/topology/DistanceOnGrid.inl>
#include <sofa/component/topology/DynamicSparseGridTopologyContainer.h>
#include <sofa/simulation/common/Simulation.h>
#endif

#include <string>
#include <iostream>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using sofa::component::topology::TriangleSetTopologyContainer;


template <class TIn, class TOut>
SkinningMapping<TIn, TOut>::SkinningMapping (core::State<In>* from, core::State<Out>* to )
    : Inherit ( from, to )
#ifndef SOFA_DEV
    , nbRefs ( initData ( &nbRefs, ( unsigned ) 3,"nbRefs","Number of primitives influencing each point." ) )
    , repartition ( initData ( &repartition,"repartition","Repartition between input DOFs and skinned vertices." ) )
#endif
    , weights ( initData ( &weights,"weights","weights list for the influences of the references Dofs" ) )
    , weightGradients ( initData ( &weightGradients,"weightGradients","weight gradients list for the influences of the references Dofs" ) )
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
#ifdef SOFA_DEV
    , newFrameMinDist ( initData ( &newFrameMinDist, 0.1, "newFrameMinDist","Minimal distance to insert a new frame." ) )
    , newFrameWeightingRadius ( initData ( &newFrameWeightingRadius, "newFrameWeightingRadius","new frame weightin radius." ) )
    , newFrameDefaultCutOffDistance ( initData ( &newFrameDefaultCutOffDistance, (double)0xFFF, "newFrameDefaultCutOffDistance","new frame defaultCut off distance." ) )
    , newFrameDistanceToMaximizeWeight ( initData ( &newFrameDistanceToMaximizeWeight, 0.0, "newFrameDistanceToMaximizeWeight","new frame distance used to maximize weights." ) )
    , enableSkinning ( initData ( &enableSkinning, true, "enableSkinning","enable skinning." ) )
    , voxelVolume ( initData ( &voxelVolume, 1.0, "voxelVolume","default volume voxel. Use if no hexa topo is found." ) )
    , useElastons ( initData ( &useElastons, false, "useElastons","Use Elastons to improve numerical integration" ) )
#endif
    , wheightingType ( initData ( &wheightingType, "wheightingType","Weighting computation method.\n0 - none (distance is used).\n1 - inverse distance square.\n2 - linear.\n3 - hermite (on groups of four dofs).\n4 - spline (on groups of four dofs)." ) )
    , distanceType ( initData ( &distanceType, "distanceType","Distance computation method.\n0 - euclidian distance.\n1 - geodesic distance.\n2 - harmonic diffusion." ) )
    , computeWeights ( true )
{


    maskFrom = NULL;
    if ( core::behavior::BaseMechanicalState *stateFrom = dynamic_cast< core::behavior::BaseMechanicalState *> ( from ) )
        maskFrom = &stateFrom->forceMask;
    maskTo = NULL;
    if ( core::behavior::BaseMechanicalState *stateTo = dynamic_cast< core::behavior::BaseMechanicalState *> ( to ) )
        maskTo = &stateTo->forceMask;

#ifdef SOFA_DEV
    distOnGrid = NULL;
#endif

    sofa::helper::OptionsGroup wheightingTypeOptions(5,"None","InvDistSquare","Linear", "Hermite", "Spline");
    wheightingTypeOptions.setSelectedItem(WEIGHT_INVDIST_SQUARE);
    wheightingType.setValue(wheightingTypeOptions);

    sofa::helper::OptionsGroup* distanceTypeOptions = distanceType.beginEdit();
    distanceTypeOptions->setNames(5,"Euclidian","Geodesic", "Harmonic", "StiffnessDiffusion", "HarmonicWithStiffness");
    distanceTypeOptions->setSelectedItem(0);
    distanceType.endEdit();
}

template <class TIn, class TOut>
SkinningMapping<TIn, TOut>::~SkinningMapping ()
{
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::computeInitPos ( )
{
    const VecCoord& xto = ( this->toModel->getX0()->size() == 0)?*this->toModel->getX():*this->toModel->getX0();
    const VecInCoord& xfrom = *this->fromModel->getX0();

    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();

    initPos.resize ( xto.size() * nbRef );
    for ( unsigned int i = 0; i < xto.size(); i++ )
        for ( unsigned int m = 0; m < nbRef; m++ )
        {
            const int& idx=nbRef *i+m;
            const int& idxReps=m_reps[idx];
            getLocalCoord( initPos[idx], xfrom[idxReps], xto[i]);
        }
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::computeDistances ()
{
#ifdef SOFA_DEV
    if( this->computeAllMatrices.getValue() && distanceType.getValue().getSelectedId() != DISTANCE_EUCLIDIAN)
    {
        const VecInCoord& xfrom0 = *this->fromModel->getX0();

        GeoVecCoord tmpFrom;
        tmpFrom.resize ( xfrom0.size() );
        for ( unsigned int i = 0; i < xfrom0.size(); i++ )
            tmpFrom[i] = xfrom0[i].getCenter();

        if (this->computeAllMatrices.getValue())
        {
            sofa::helper::OptionsGroup* distOnGridanceTypeOption = distOnGrid->distanceType.beginEdit();
            if ( distanceType.getValue().getSelectedId() == DISTANCE_GEODESIC) distOnGridanceTypeOption->setSelectedItem(TYPE_GEODESIC);
            if ( distanceType.getValue().getSelectedId() == DISTANCE_HARMONIC) distOnGridanceTypeOption->setSelectedItem(TYPE_HARMONIC);
            if ( distanceType.getValue().getSelectedId() == DISTANCE_STIFFNESS_DIFFUSION) distOnGridanceTypeOption->setSelectedItem(TYPE_STIFFNESS_DIFFUSION);
            if ( distanceType.getValue().getSelectedId() == DISTANCE_HARMONIC_STIFFNESS) distOnGridanceTypeOption->setSelectedItem(TYPE_HARMONIC_STIFFNESS);
            distOnGrid->distanceType.endEdit();
        }
        distOnGrid->computeDistanceMap ( tmpFrom );
    }
#endif

    this->getDistances( 0);
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::getDistances( int xfromBegin)
{
    const VecCoord& xto0 = ( this->toModel->getX0()->size() == 0)?*this->toModel->getX():*this->toModel->getX0();
    const VecInCoord& xfrom0 = *this->fromModel->getX0();

    switch ( distanceType.getValue().getSelectedId() )
    {
    case DISTANCE_EUCLIDIAN:
    {
        distances.resize( xfrom0.size());
        distGradients.resize( xfrom0.size());
        for ( unsigned int i = xfromBegin; i < xfrom0.size(); ++i ) // for each new frame
        {
            distances[i].resize ( xto0.size() );
            distGradients[i].resize ( xto0.size() );
            for ( unsigned int j=0; j<xto0.size(); ++j )
            {
                distGradients[i][j] = xto0[j] - xfrom0[i].getCenter();
                distances[i][j] = distGradients[i][j].norm();
                distGradients[i][j].normalize();
            }
        }
        break;
    }
#ifdef SOFA_DEV
    case DISTANCE_GEODESIC:
    case DISTANCE_HARMONIC:
    case DISTANCE_STIFFNESS_DIFFUSION:
    case DISTANCE_HARMONIC_STIFFNESS:
    {
        GeoVecCoord goals;
        goals.resize ( xto0.size() );
        for ( unsigned int j = 0; j < xto0.size(); ++j )
            goals[j] = xto0[j];
        distOnGrid->getDistances ( distances, distGradients, goals );
        break;
    }
#endif
    default: {}
    }
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::sortReferences( vector<unsigned int>& references)
{
    const VecCoord& xto = ( this->toModel->getX0()->size() == 0)?*this->toModel->getX():*this->toModel->getX0();
    const VecInCoord& xfrom = *this->fromModel->getX0();
    const unsigned int& nbRef = this->nbRefs.getValue();

    references.clear();
    references.resize ( nbRef *xto.size() );
    for ( unsigned int i=0; i< nbRef *xto.size(); i++ )
        references[i] = -1;

    for ( unsigned int i=0; i<xfrom.size(); i++ )
        for ( unsigned int j=0; j<xto.size(); j++ )
            for ( unsigned int k=0; k<nbRef; k++ )
            {
                const int idxReps=references[nbRef*j+k];
                if ( ( idxReps == -1 ) || ( distances[i][j] < distances[idxReps][j] ) )
                {
                    for ( unsigned int m=nbRef-1 ; m>k ; m-- )
                        references[nbRef *j+m] = references[nbRef *j+m-1];
                    references[nbRef *j+k] = i;
                    break;
                }
            }
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::normalizeWeights()
{
    const unsigned int xtoSize = this->toModel->getX()->size();
    const unsigned int& nbRef = this->nbRefs.getValue();
    VVD& m_weights = * ( weights.beginEdit() );
    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());
    VVMat33& m_ddweight = this->weightGradients2;

    // Normalise weights & dweights
    for (unsigned int i = 0; i < xtoSize; ++i)
    {
        double sumWeights = 0,wn;
        Vec3 sumGrad,dwn;			sumGrad.fill(0);
        Mat33 sumGrad2,ddwn;		sumGrad2.fill(0);

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
                for(unsigned int o=0; o<3; o++) for(unsigned int p=0; p<3; p++) ddwn[o][p]=(m_ddweight[i][j][o][p] - wn*sumGrad2[o][p] - sumGrad[o]*dwn[p] - sumGrad[p]*dwn[o])/sumWeights;
                m_ddweight[i][j]=ddwn;
                m_dweight[i][j]=dwn;
                m_weights[i][j] =wn;
            }
    }
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::init()
{
#ifdef SOFA_DEV
    if ( distanceType.getValue().getSelectedId() != DISTANCE_EUCLIDIAN)
    {
        this->getContext()->get ( distOnGrid, core::objectmodel::BaseContext::SearchRoot );
        if ( !distOnGrid )
        {
            serr << "Can not find the DistanceOnGrid component: distances used are euclidian." << sendl;
            distanceType.setValue( DISTANCE_EUCLIDIAN);
        }
    }
#else
    distanceType.beginEdit()->setSelectedItem(DISTANCE_EUCLIDIAN);
    distanceType.endEdit();
#endif
    const VecInCoord& xfrom = *this->fromModel->getX0();
    if ( this->initPos.empty() && this->toModel!=NULL && computeWeights==true && weights.getValue().size() ==0 )
    {
        //*
        if ( wheightingType.getValue().getSelectedId() == WEIGHT_LINEAR || wheightingType.getValue().getSelectedId() == WEIGHT_HERMITE )
            this->nbRefs.setValue ( 2 );

        if ( wheightingType.getValue().getSelectedId() == WEIGHT_SPLINE)
            this->nbRefs.setValue ( 4 );

        if ( xfrom.size() < this->nbRefs.getValue())
            this->nbRefs.setValue ( xfrom.size() );
        /*/
        this->nbRefs.setValue ( xfrom.size() );
        //*/

        computeDistances();
        vector<unsigned int>& m_reps = * ( this->repartition.beginEdit() );
        sortReferences ( m_reps);
        this->repartition.endEdit();
        updateWeights ();
        computeInitPos ();
    }
    else if ( computeWeights == false || weights.getValue().size() !=0 )
    {
        computeInitPos();
    }

#ifdef SOFA_DEV
    precomputeMatrices<In>(In());
#endif

    Inherit::init();
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::clear()
{
    this->initPos.clear();
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::setWeightsToHermite()
{
    wheightingType.beginEdit()->setSelectedItem(WEIGHT_HERMITE);
    wheightingType.endEdit();
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::setWeightsToLinear()
{
    wheightingType.beginEdit()->setSelectedItem(WEIGHT_LINEAR);
    wheightingType.endEdit();
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::setWeightsToInvDist()
{
    wheightingType.beginEdit()->setSelectedItem(WEIGHT_INVDIST_SQUARE);
    wheightingType.endEdit();
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::updateWeights ()
{
    const VecCoord& xto = ( this->toModel->getX0()->size() == 0)?*this->toModel->getX():*this->toModel->getX0();
    const VecInCoord& xfrom = *this->fromModel->getX0();

    VVD& m_weights = * ( weights.beginEdit() );
    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());
    VVMat33& m_ddweight = this->weightGradients2;

    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();

    m_weights.resize ( xto.size() );    for ( unsigned int i=0; i<xto.size(); i++ )        m_weights[i].resize ( nbRef );
    m_dweight.resize ( xto.size() );    for ( unsigned int i=0; i<xto.size(); i++ )        m_dweight[i].resize ( nbRef );
    m_ddweight.resize( xto.size() );    for ( unsigned int i=0; i<xto.size(); i++ )        m_ddweight[i].resize( nbRef );

    switch ( wheightingType.getValue().getSelectedId() )
    {
    case WEIGHT_NONE:
    {
        for ( unsigned int i=0; i<xto.size(); i++ )
            for ( unsigned int j=0; j<nbRef; j++ )
            {
                int indexFrom = m_reps[nbRef*i+j];
#ifdef SOFA_DEV
                if ( distanceType.getValue().getSelectedId()  == DISTANCE_HARMONIC)
                {
                    m_weights[indexFrom][i] = distOnGrid->harmonicMaxValue.getValue() - distances[indexFrom][i];
                    if ( distances[indexFrom][i] < 0.0) distances[indexFrom][i] = 0.0;
                    if ( distances[indexFrom][i] > distOnGrid->harmonicMaxValue.getValue()) distances[indexFrom][i] = distOnGrid->harmonicMaxValue.getValue();
                    if(distances[indexFrom][i]==0 || distances[indexFrom][i]==distOnGrid->harmonicMaxValue.getValue()) m_dweight[indexFrom][i]=Coord();
                    else m_dweight[indexFrom][i] = - distGradients[indexFrom][i];
                }
                else
                {
#endif
                    m_weights[i][j] = distances[indexFrom][i];
                    m_dweight[i][j] = distGradients[indexFrom][i];
#ifdef SOFA_DEV
                }
#endif
            }
        break;
    }
    case WEIGHT_LINEAR:
    {
        vector<unsigned int> tmpReps;
        sortReferences( tmpReps);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            for ( unsigned int j=0; j<nbRef; j++ )
            {
                m_weights[i][j] = 0.0;
                m_dweight[i][j] = GeoCoord();
                m_ddweight[i][j].fill(0);
            }
            Vec3d r1r2, r1p;
            r1r2 = xfrom[tmpReps[nbRef *i+1]].getCenter() - xfrom[tmpReps[nbRef *i+0]].getCenter();
            r1p  = xto[i] - xfrom[tmpReps[nbRef *i+0]].getCenter();
            double r1r2NormSquare = r1r2.norm()*r1r2.norm();
            double wi = ( r1r2*r1p ) / ( r1r2NormSquare);

            // Abscisse curviligne
            m_weights[i][0] = ( 1 - wi );
            m_weights[i][1] = wi;
            m_dweight[i][0] = -r1r2 / r1r2NormSquare;
            m_dweight[i][1] = r1r2 / r1r2NormSquare;
            //m_ddweight[i][0] = ;
            //m_ddweight[i][1] = ;
        }
        break;
    }
    case WEIGHT_INVDIST_SQUARE:
    {
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            for ( unsigned int j=0; j<nbRef; j++ )
            {
                int indexFrom = m_reps[nbRef*i+j];
                double d2=distances[indexFrom][i]*distances[indexFrom][i];
                double d3=d2*distances[indexFrom][i];
                double d4=d3*distances[indexFrom][i];

                m_ddweight[i][j].fill(0);
                if (d2)
                {
                    m_weights[i][j] = 1 / d2;
                    m_dweight[i][j] = - distGradients[indexFrom][i] / d3* 2.0;
                    m_ddweight[i][j][0][0]-=2.0/d4;
                    m_ddweight[i][j][1][1]-=2.0/d4;
                    m_ddweight[i][j][2][2]-=2.0/d4;
                    for(unsigned int k=0; k<3; k++)
                        for(unsigned int m=0; m<3; m++)
                            m_ddweight[i][j][k][m]+=distGradients[indexFrom][i][k]*distGradients[indexFrom][i][m]*8.0/d2;
                }
                else
                {
                    m_weights[i][j] = 0xFFF;
                    m_dweight[i][j] = GeoCoord();
                }

            }
        }

        break;
    }
    case WEIGHT_HERMITE:
    {
        vector<unsigned int> tmpReps;
        sortReferences( tmpReps);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            for ( unsigned int j=0; j<nbRef; j++ )
            {
                m_weights[i][j] = 0.0;
                m_dweight[i][j] = GeoCoord();
            }
            Vec3d r1r2, r1p;
            double wi;
            r1r2 = xfrom[tmpReps[nbRef *i+1]].getCenter() - xfrom[tmpReps[nbRef *i+0]].getCenter();
            r1p  = xto[i] - xfrom[tmpReps[nbRef *i+0]].getCenter();
            double r1r2NormSquare = r1r2.norm()*r1r2.norm();
            wi = ( r1r2*r1p ) / r1r2NormSquare;

            // Fonctions d'Hermite
            m_weights[i][0] = 1-3*wi*wi+2*wi*wi*wi;
            m_weights[i][1] = 3*wi*wi-2*wi*wi*wi;

            r1r2.normalize();
            m_dweight[i][0] = -r1r2 * (6*wi-6*wi*wi) / (r1r2NormSquare);
            m_dweight[i][1] = r1r2 * (6*wi-6*wi*wi) / (r1r2NormSquare);
        }
        break;
    }
    case WEIGHT_SPLINE:
    {
        if( xfrom.size() < 4 || nbRef < 4)
        {
            serr << "Error ! To use WEIGHT_SPLINE, you must use at least 4 DOFs and set nbRefs to 4.\n WEIGHT_SPLINE requires also the DOFs are ordered along z-axis." << sendl;
            return;
        }
        vector<unsigned int> tmpReps;
        sortReferences( tmpReps);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            // Clear all weights and dweights.
            for ( unsigned int j=0; j<nbRef; j++ )
            {
                m_weights[i][j] = 0.0;
                m_dweight[i][j] = GeoCoord();
            }
            // Get the 4 nearest DOFs.
            vector<unsigned int> sortedFrames;
            for( unsigned int j = 0; j < 4; ++j)
                sortedFrames.push_back( tmpReps[nbRef *i+j]);
            std::sort( sortedFrames.begin(), sortedFrames.end());

            if( xto[i][2] < xfrom[sortedFrames[1]].getCenter()[2])
            {
                vector<unsigned int> sortedFramesCpy = sortedFrames;
                sortedFrames.clear();
                sortedFrames.push_back( sortedFramesCpy[0]);
                sortedFrames.push_back( sortedFramesCpy[0]);
                sortedFrames.push_back( sortedFramesCpy[1]);
                sortedFrames.push_back( sortedFramesCpy[2]);
            }
            else if( xto[i][2] > xfrom[sortedFrames[2]].getCenter()[2])
            {
                vector<unsigned int> sortedFramesCpy = sortedFrames;
                sortedFrames.clear();
                sortedFrames.push_back( sortedFramesCpy[1]);
                sortedFrames.push_back( sortedFramesCpy[2]);
                sortedFrames.push_back( sortedFramesCpy[3]);
                sortedFrames.push_back( sortedFramesCpy[3]);
            }

            // Compute u
            Vec3d r1r2 = xfrom[sortedFrames[2]].getCenter() - xfrom[sortedFrames[1]].getCenter();
            Vec3d r1p  = xto[i] - xfrom[sortedFrames[1]].getCenter();
            double r1r2NormSquare = r1r2.norm()*r1r2.norm();
            double u = ( r1r2*r1p ) / r1r2NormSquare;

            // Set weights and dweights.
            m_weights[i][0] = 1-3*u*u+2*u*u*u;
            m_weights[i][1] = u*u*u - 2*u*u + u;
            m_weights[i][2] = 3*u*u-2*u*u*u;
            m_weights[i][3] = u*u*u - u*u;

            r1r2.normalize();
            m_dweight[i][0] = -r1r2 * (6*u - 6*u*u) / (r1r2NormSquare);
            m_dweight[i][1] = r1r2 * (3*u*u - 4*u + 1) / (r1r2NormSquare);
            m_dweight[i][2] = r1r2 * (6*u - 6*u*u) / (r1r2NormSquare);
            m_dweight[i][3] = r1r2 * (3*u*u - 2*u) / (r1r2NormSquare);
        }
        break;
    }
    default:
    {}
    }

    normalizeWeights();
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::setWeightCoefs ( VVD &weights )
{
    VVD * m_weights = this->weights.beginEdit();
    m_weights->clear();
    m_weights->insert ( m_weights->begin(), weights.begin(), weights.end() );
    this->weights.endEdit();
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::setRepartition ( vector<int> &rep )
{
    vector<unsigned int> * m_reps = this->repartition.beginEdit();
    m_reps->clear();
    m_reps->insert ( m_reps->begin(), rep.begin(), rep.end() );;
    this->repartition.endEdit();
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    _apply< InCoord >(out, in);
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    _applyJ< InDeriv >(out, in);
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    _applyJT< InDeriv >(out, in);
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    _applyJT_Matrix< InMatrixDeriv >( out, in);
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::draw()
{
    const typename Out::VecCoord& xto = *this->toModel->getX();
    const typename In::VecCoord& xfrom = *this->fromModel->getX();
    const unsigned int nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    const VVD& m_weights = weights.getValue();
    const SVector<SVector<GeoCoord> >& dw = weightGradients.getValue();
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
                const int idxReps=m_reps[nbRef *i+m];
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
            sofa::helper::gl::GlText::draw ( m_reps[nbRef*i+0]*scale, xto[i], textScale );
    }

    // Display distances for each points
    if ( showDistancesValues.getValue())
    {
        glColor3f( 1.0, 1.0, 1.0);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            bool influenced;
            unsigned int refIndex;
            reverseRepartition(influenced, refIndex, i, showFromIndex.getValue()%distances.size());
            if( influenced)
            {
                sofa::helper::gl::GlText::draw ( (int)(distances[refIndex][i]*scale), xto[i], textScale );
            }
        }
    }

    // Display distance gradients values for each points
    if ( showGradientsValues.getValue())
    {
        char txt[100];
        glColor3f( 0.5, 0.5, 0.5);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            bool influenced;
            unsigned int refIndex;
            reverseRepartition(influenced, refIndex, i, showFromIndex.getValue()%distances.size());
            if( influenced)
            {
                const Vec3& grad = dw[i][refIndex];
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
            reverseRepartition(influenced, indexRep, i, showFromIndex.getValue()%distances.size());
            if( influenced)
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
            reverseRepartition(influenced, indexRep, i, showFromIndex.getValue()%distances.size());
            if (influenced)
            {
                const GeoCoord& gradMap = dw[i][indexRep];
                const Coord& point = xto[i];
                glVertex3f ( point[0], point[1], point[2] );
                glVertex3f ( point[0] + gradMap[0] * showGradientsScaleFactor.getValue(), point[1] + gradMap[1] * showGradientsScaleFactor.getValue(), point[2] + gradMap[2] * showGradientsScaleFactor.getValue() );
            }
        }
        glEnd();
    }
    //*/

#ifdef SOFA_DEV
    // Show weights
    if ( showWeights.getValue())
    {
        // Compute min and max values.
        double minValue = 0xFFFF;
        double maxValue = -0xFFF;
        for ( unsigned int i = 0; i < xto.size(); i++)
        {
            bool influenced;
            unsigned int indexRep;
            reverseRepartition(influenced, indexRep, i, showFromIndex.getValue()%distances.size());
            if (influenced)
            {
                const double& weight = m_weights[i][indexRep];
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
                    reverseRepartition(influenced, indexRep, i, showFromIndex.getValue()%distances.size());
                    if (influenced)
                    {
                        const unsigned int& indexPoint = tri[i][j];
                        double color = (m_weights[indexPoint][indexRep] - minValue) / (maxValue - minValue);
                        color = pow(color, showGammaCorrection.getValue());
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
                reverseRepartition(influenced, indexRep, i, showFromIndex.getValue()%distances.size());
                if (influenced)
                {
                    double color = (m_weights[i][indexRep] - minValue) / (maxValue - minValue);
                    color = pow(color, showGammaCorrection.getValue());
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
                    double color = 0.5 + ( e[0] + e[1] + e[2])/this->showDefTensorScale.getValue();
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
                double mult=500;
                double color = (e[0]+e[1]+e[2])/3.;
                if(color<0) color=2*color/(color+1.);
                color*=mult; color+=120; if(color<0) color=0; if(color>240) color=240;
                sofa::helper::gl::Color::setHSVA(color,1.,.8,1.);
                glVertex3f( xto[i][0], xto[i][1], xto[i][2]);
            }
            glEnd();
        }
    }


#endif
}

#ifdef SOFA_DEV
template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::removeFrame( const unsigned int /*index*/)
{
    //VecCoord& xto0 = *this->toModel->getX0();

    Data<VecCoord> &xto_d = *this->toModel->write(core::VecCoordId::position());
    VecCoord& xto = *xto_d.beginEdit();

    //VecInCoord& xfrom0 = *this->fromModel->getX0();
    const VecInCoord& xfrom = *this->fromModel->getX();

    // this->T.erase( T.begin()+index);
    // coeffs

    // Recompute matrices
    apply( xto, xfrom);

    xto_d.endEdit();
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::insertFrame( const Coord& pos, const Quat& rot, GeoVecCoord beginPointSet, double distMax)
{
    changeSettingsDueToInsertion();

    if (!this->toModel->getX0()) return;
    // Get references
    Data<VecCoord> &xto_d = *this->toModel->write(core::VecCoordId::position());
    VecCoord& xto = *xto_d.beginEdit();
    Data<VecInCoord> &xfrom0_d = *this->fromModel->write(core::VecCoordId::restPosition());
    VecInCoord &xfrom0 = *xfrom0_d.beginEdit();
    Data<VecInCoord> &xfrom_d = *this->fromModel->write(core::VecCoordId::position());
    VecInCoord &xfrom = *xfrom_d.beginEdit();
    Data<VecInCoord> &xfromReset_d = *this->fromModel->write(core::VecCoordId::resetPosition());
    VecInCoord &xfromReset = *xfromReset_d.beginEdit();

    unsigned int indexFrom = xfrom.size();
    core::behavior::MechanicalState< In >* mstateFrom = dynamic_cast<core::behavior::MechanicalState< In >* >( this->fromModel);
    if ( !mstateFrom)
    {
        serr << "Error: try to insert a new frame on fromModel, which is not a mechanical state !" << sendl;
        return;
    }

    // Compute the rest position of the frame.
    InCoord newX, newX0;
    InCoord targetDOF;
    setInCoord( targetDOF, pos, rot);
    inverseSkinning( newX0, newX, targetDOF);

    // Insert a new DOF
    this->fromModel->resize( indexFrom + 1);
    xfrom[indexFrom] = newX;
    xfrom0[indexFrom] = newX0;
    xfromReset[indexFrom] = newX0;

    if ( distMax == 0.0)
        distMax = this->newFrameDefaultCutOffDistance.getValue();

    // Compute geodesical/euclidian distance for this frame.
    if ( this->distanceType.getValue().getSelectedId() == DISTANCE_GEODESIC || this->distanceType.getValue().getSelectedId() == DISTANCE_HARMONIC || this->distanceType.getValue().getSelectedId() == DISTANCE_STIFFNESS_DIFFUSION || this->distanceType.getValue().getSelectedId() == TYPE_HARMONIC_STIFFNESS)
        this->distOnGrid->addElt( newX0.getCenter(), beginPointSet, distMax);
    vector<double>& vRadius = (*this->newFrameWeightingRadius.beginEdit());
    vRadius.resize( indexFrom + 1);
    vRadius[indexFrom] = distMax;
    this->newFrameWeightingRadius.endEdit();

    // Recompute matrices
    apply( xto, xfrom);

    xto_d.endEdit();
    xfrom0_d.endEdit();
    xfrom_d.endEdit();
    xfromReset_d.endEdit();
}

template <class TIn, class TOut>
bool SkinningMapping<TIn, TOut>::inverseSkinning( InCoord& /*X0*/, InCoord& /*X*/, const InCoord& /*Xtarget*/)
// compute position in reference configuration from deformed position
{
    /*  TODO !!
    const VecInCoord& xi = *this->fromModel->getX();
    const VMat88& T = this->T;
    //const VecInCoord& xi0 = *this->fromModel->getX0();
    const VecCoord& P0 = *this->toModel->getX0();
    const VecCoord& P = *this->toModel->getX();

    int i,j,k,l,nbP=P0.size(),nbDOF=xi.size();
    VDUALQUAT qrel(nbDOF);
    DUALQUAT q,b,bn;
    Vec3 t;
    Vec4 qinv;
    Mat33 R,U,Uinv;
    Mat88 N;
    Mat38 Q;
    Mat83 W,NW;
    double QEQ0,Q0Q0,Q0,d,dmin=1E5;
    Mat44 q0q0T,q0qeT,qeq0T;
    VVD w;
    w.resize(nbDOF);
    for ( int i = 0; i < nbDOF; i++) w[i].resize(P.size());
    VecVecCoord dw(nbDOF);
    for ( int i = 0; i < nbDOF; i++) dw[i].resize(P.size());
    X.getOrientation() = Xtarget.getOrientation();

    // init skinning
    for (i=0;i<nbDOF;i++)
    {
    XItoQ( q, xi[i]); //update DOF quats
    computeQrel( qrel[i], T[i], q); //update qrel=Tq
    }

    // get closest material point
    for (i=0;i<nbP;i++)
    {
    t = Xtarget.getCenter() - P[i];
    d = t * t;
    if (d<dmin) {
    dmin = d;
    X0.getCenter() = P0[i];
    }
    }
    if (dmin==1E5) return false;

    // iterate: pos0(t+1)=pos0(t) + (dPos/dPos0)^-1 (Pos - Pos(t))
    double eps=1E-5;
    bool stop=false;
    int count=0;

    while (!stop)
    {
    // update weigths
    computeWeight( w, dw, X0.getCenter());

    // update skinning
    BlendDualQuat( b, bn, QEQ0, Q0Q0, Q0, 0, qrel, w); //skinning: sum(wTq)
    computeDqRigid( R, t, bn); //update Rigid

    qinv[0]=-bn.q0[0];
    qinv[1]=-bn.q0[1];
    qinv[2]=-bn.q0[2];
    qinv[3]=bn.q0[3];
    Multi_Q( X0.getOrientation(), qinv, Xtarget.getOrientation());
    for (k=0;k<3;k++) {
    X.getCenter()[k] = t[k];
    for (j=0;j<3;j++) X.getCenter()[k] += R[k][j] * X0.getCenter()[j];
    }
    //update skinned points

    t = Xtarget.getCenter()- X.getCenter();

    //std::cout<<" "<<t*t;

    if ( t*t < eps || count >= 10) stop = true;
    count++;

    if (!stop)
    {
    computeDqN_constants( q0q0T, q0qeT, qeq0T, bn);
    computeDqN( N, q0q0T, q0qeT, qeq0T, QEQ0, Q0Q0, Q0); //update N=d(bn)/d(b)
    computeDqQ( Q, bn, X0.getCenter()); //update Q=d(P)/d(bn)
    W.fill(0);
    for (j=0;j<nbDOF;j++) for (k=0;k<4;k++) for (l=0;l<3;l++)
        {
            W[k][l]+=dw[j][0][l]*qrel[j].q0[k];
            W[k+4][l]+=dw[j][0][l]*qrel[j].qe[k];
        }
    //update W=sum(wrel.dw)=d(b)/d(p)
    NW=N*W; // update NW
    // grad def = d(P)/d(p0)
    U=R+Q*NW; // update A=R+QNW
    invertMatrix(Uinv,U);

    // Update pos0
    X0.getCenter() += Uinv * t;
    }
    }
    //std::cout<<"err:"<<t*t;
    */
    return true;
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::computeWeight( VVD& w, VecVecCoord& dw, const Coord& x0)
{
    const VecCoord& xto0 = *this->toModel->getX0();
    const VecInCoord& xfrom0 = *this->fromModel->getX0();
    // Get Distances
    VVD dist;
    GeoVecVecCoord ddist;

    switch ( this->distanceType.getValue().getSelectedId()  )
    {
    case DISTANCE_EUCLIDIAN:
    {
        dist.resize( xfrom0.size());
        ddist.resize( xfrom0.size());
        for ( unsigned int i = 0; i < xfrom0.size(); i++)
        {
            dist[i].resize( 1);
            ddist[i].resize( 1);

            ddist[i][0] = x0 - xfrom0[i].getCenter();
            dist[i][0] = ddist[i][0].norm();
            ddist[i][0].normalize();
        }
        break;
    }
    case DISTANCE_GEODESIC:
    case DISTANCE_HARMONIC:
    case DISTANCE_STIFFNESS_DIFFUSION:
    case DISTANCE_HARMONIC_STIFFNESS:
    {
        GeoVecCoord goals;
        goals.push_back( x0);
        this->distOnGrid->getDistances ( dist, ddist, goals );
        break;
    }
    default: {}
    }

    // Compute Weights
    switch ( this->wheightingType.getValue().getSelectedId()  )
    {
    case WEIGHT_NONE:
    {
        for ( unsigned int i=0; i<xfrom0.size(); i++ )
        {
            w[i][0] = dist[i][0];
            dw[i][0] = ddist[i][0];
        }
        break;
    }
    case WEIGHT_LINEAR:
    {
        for ( unsigned int i=0; i<xfrom0.size(); i++ )
        {
            Vec3d r1r2, r1p;
            r1r2 = xfrom0[(i+1)%(xfrom0.size())].getCenter() - xfrom0[i].getCenter();
            r1p  = xto0[0] - xfrom0[i].getCenter();
            double r1r2NormSquare = r1r2.norm()*r1r2.norm();
            double wi = ( r1r2*r1p ) / ( r1r2NormSquare);

            // Abscisse curviligne
            w[i][0]                   = ( 1 - wi );
            w[(i+1)%(xfrom0.size())][0] = wi;
            dw[i][0]                   = -r1r2 / r1r2NormSquare;
            dw[(i+1)%(xfrom0.size())][0] = r1r2 / r1r2NormSquare;
        }
        break;
    }
    case WEIGHT_INVDIST_SQUARE:
    {
        for ( unsigned int i=0; i<xfrom0.size(); i++ )
        {
            if ( dist[i][0])
            {
                w[i][0] = 1 / (dist[i][0]*dist[i][0]);
                dw[i][0] = - ddist[i][0] / (dist[i][0]*dist[i][0]*dist[i][0]) * 2.0;
            }
            else
            {
                w[i][0] = 0xFFF;
                dw[i][0] = GeoCoord();
            }
        }
        break;
    }
    case WEIGHT_HERMITE:
    {
        for ( unsigned int i=0; i<xfrom0.size(); i++ )
        {
            Vec3d r1r2, r1p;
            double wi;
            r1r2 = xfrom0[(i+1)%xfrom0.size()].getCenter() - xfrom0[i].getCenter();
            r1p  = xto0[0] - xfrom0[i].getCenter();
            wi = ( r1r2*r1p ) / ( r1r2.norm() *r1r2.norm() );

            // Fonctions d'Hermite
            w[i][0]                   = 1-3*wi*wi+2*wi*wi*wi;
            w[(i+1)%(xfrom0.size())][0] = 3*wi*wi-2*wi*wi*wi;

            r1r2.normalize();
            dw[i][0]                   = -r1r2;
            dw[(i+1)%(xfrom0.size())][0] = r1r2;
        }
        break;
    }
    default:
    {}
    }
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::updateDataAfterInsertion()
{
    /* TODO !!!
    const VecCoord& xto = *this->toModel->getX();
    const VecInCoord& xfrom0 = *this->fromModel->getX0();
    const VecInCoord& xfrom = *this->fromModel->getX();
    VVD& m_weights = * ( this->weights.beginEdit() );
    SVector<SVector<GeoCoord> >& dw = *(this->weightGradients.beginEdit());
    vector<double>& radius = (*this->newFrameWeightingRadius.beginEdit());

    changeSettingsDueToInsertion();

    //TODO fix it ! Synchro between many SMapping
    if ( radius.size() != xfrom.size())
    {
    int oldSize = radius.size();
    radius.resize( xfrom.size());
    for ( unsigned int i = oldSize; i < xfrom.size(); ++i) radius[i] = this->newFrameDefaultCutOffDistance.getValue();
    }

    // Resize T
    unsigned int size = this->T.size();
    DUALQUAT qi0;
    this->T.resize ( xfrom.size() );

    // Get distances
    this->getDistances( size);

    // for each new frame
    const double& maximizeWeightDist = this->newFrameDistanceToMaximizeWeight.getValue();
    for ( unsigned int i = size; i < xfrom.size(); ++i )
    {
    // Get T
    XItoQ( qi0, xfrom0[i]);
    computeDqT ( this->T[i], qi0 );

    // Update weights
    m_weights.resize ( xfrom.size() );
    m_weights[i].resize ( xto.size() );
    dw.resize ( xfrom.size() );
    dw[i].resize ( xto.size() );
    for ( unsigned int j = 0; j < xto.size(); ++j )
    {
    if ( this->distances[i][j])
    {
    if ( this->distances[i][j] == -1.0)
    {
        m_weights[i][j] = 0.0;
        dw[i][j] = GeoCoord();
    }
    else
    {
        if( maximizeWeightDist != 0.0)
        {
          if( this->distances[i][j] < maximizeWeightDist)
          {
            m_weights[i][j] = 0xFFF;
            dw[i][j] = GeoCoord();
          }
          else if( this->distances[i][j] > radius[i])
          {
            m_weights[i][j] = 0.0;
            dw[i][j] = GeoCoord();
          }
          else
          {
            // linear interpolation from 0 to 0xFFF
            //m_weights[i][j] = 0xFFF / (maximizeWeightDist - radius[i]) * (this->distances[i][j] - radius[i]);
            //dw[i][j] = this->distGradients[i][j] * 0xFFF / (maximizeWeightDist - radius[i]);

            // Hermite between 0 and 0xFFF
            double dPrime = (this->distances[i][j] - maximizeWeightDist) / (radius[i]-maximizeWeightDist);
            m_weights[i][j] = (1-3*dPrime*dPrime+2*dPrime*dPrime*dPrime) * 0xFFF;
            dw[i][j] = - this->distGradients[i][j] * (6*dPrime-6*dPrime*dPrime) / (radius[i]-maximizeWeightDist ) * 0xFFF;
          }
        }
        else
        {
          m_weights[i][j] = 1.0 / (this->distances[i][j]*this->distances[i][j]) - 1.0 / (radius[i]*radius[i]);
          if (m_weights[i][j] < 0)
          {
            m_weights[i][j] = 0.0;
            dw[i][j] = GeoCoord();
          }
          else
            dw[i][j] = - this->distGradients[i][j] / (this->distances[i][j]*this->distances[i][j]*this->distances[i][j]) * 2.0;
        }
    }
    }
    else
    {
    m_weights[i][j] = 0xFFF;
    dw[i][j] = GeoCoord();
    }
    }
    }
    this->weights.endEdit();
    this->weightGradients.endEdit();
    this->newFrameWeightingRadius.endEdit();
    */
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::changeSettingsDueToInsertion()
{
    this->setWeightsToInvDist();
    if( this->distOnGrid)
    {
        this->distanceType.beginEdit()->setSelectedItem(DISTANCE_GEODESIC);
    }
    else
    {
        this->distanceType.beginEdit()->setSelectedItem(DISTANCE_EUCLIDIAN);
    }
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::setInCoord( typename defaulttype::StdRigidTypes<N, InReal>::Coord& coord, const Coord& position, const Quat& rotation) const
{
    coord.getCenter() = position;
    coord.getOrientation() = rotation;
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::setInCoord( typename defaulttype::StdAffineTypes<N, InReal>::Coord& coord, const Coord& position, const Quat& rotation) const
{
    coord.getCenter() = position;
    rotation.toMatrix( coord.getAffine());
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::setInCoord( typename defaulttype::StdQuadraticTypes<N, InReal>::Coord& coord, const Coord& position, const Quat& rotation) const
{
    coord.getCenter() = position;
    rotation.toMatrix( coord.getQuadratic());
}
#endif


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::getLocalCoord( Coord& result, const typename In::Coord& inCoord, const Coord& coord) const
{
    result = inCoord.pointToChild ( coord );
}


#ifdef SOFA_DEV
//template <class TIn, class TOut>
//void SkinningMapping<TIn, TOut>::getLocalCoord( Coord& result, const typename defaulttype::StdAffineTypes<N, InReal>::Coord& inCoord, const Coord& coord) const
//{
//  Mat33 affineInv;
//  affineInv.invert( inCoord.getAffine() );
//  result = affineInv * ( coord - inCoord.getCenter() );
//}
//
//
//template <class TIn, class TOut>
//void SkinningMapping<TIn, TOut>::getLocalCoord( Coord& result, const typename defaulttype::StdQuadraticTypes<N, InReal>::Coord& inCoord, const Coord& coord) const
//{
//  result = coord - inCoord.getCenter();
//}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::reverseRepartition( bool& influenced, unsigned int& realIndex, const unsigned int& pointIndex, const unsigned int& frameIndex)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
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


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::M33toV6(Vec6 &v,const Mat33& M) const
// referred as operator V in the paper
{
    v[0]=M[0][0];
    v[1]=M[1][1];
    v[2]=M[2][2];
    v[3]=(M[0][1]+M[1][0])/2.;
    v[4]=(M[2][1]+M[1][2])/2.;
    v[5]=(M[2][0]+M[0][2])/2.;
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::QtoR(Mat33& M, const Quat& q) const
{
    // q to M
    double xs = q[0]*2., ys = q[1]*2., zs = q[2]*2.;
    double wx = q[3]*xs, wy = q[3]*ys, wz = q[3]*zs;
    double xx = q[0]*xs, xy = q[0]*ys, xz = q[0]*zs;
    double yy = q[1]*ys, yz = q[1]*zs, zz = q[2]*zs;
    M[0][0] = 1.0 - (yy + zz); M[0][1]= xy - wz; M[0][2] = xz + wy;
    M[1][0] = xy + wz; M[1][1] = 1.0 - (xx + zz); M[1][2] = yz - wx;
    M[2][0] = xz - wy; M[2][1] = yz + wx; M[2][2] = 1.0 - (xx + yy);
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::ComputeL(Mat76& L, const Quat& q) const
{
    L[0][0]=1;	L[0][1]=0; L[0][2]=0;	L[0][3]=0;			L[0][4]=0;			L[0][5]=0;
    L[1][0]=0;	L[1][1]=1; L[1][2]=0;	L[1][3]=0;			L[1][4]=0;			L[1][5]=0;
    L[2][0]=0;	L[2][1]=0; L[2][2]=1;	L[2][3]=0;			L[2][4]=0;			L[2][5]=0;
    L[3][0]=0;	L[3][1]=0; L[3][2]=0;	L[3][3]=q[3]/2.;	L[3][4]=q[2]/2.;	L[3][5]=-q[1]/2.;
    L[4][0]=0;	L[4][1]=0; L[4][2]=0;	L[4][3]=-q[2]/2.;	L[4][4]=q[3]/2.;	L[4][5]=q[0]/2.;
    L[5][0]=0;	L[5][1]=0; L[5][2]=0;	L[5][3]=q[1]/2.;	L[5][4]=-q[0]/2.;	L[5][5]=q[3]/2.;
    L[6][0]=0;	L[6][1]=0; L[6][2]=0;	L[6][3]=-q[0]/2.;	L[6][4]=-q[1]/2.;	L[6][5]=-q[2]/2.;
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::ComputeQ(Mat37& Q, const Quat& q, const Vec3& p) const
{
    Q[0][0]=1; Q[0][1]=0; Q[0][2]=0;	Q[0][3]=2*(q[1]*p[1]+q[2]*p[2]);				Q[0][4]=2*(-2*q[1]*p[0]+q[0]*p[1]+q[3]*p[2]);	Q[0][5]=2*(-2*q[2]*p[0]-q[3]*p[1]+q[0]*p[2]);	Q[0][6]=2*(-q[2]*p[1]+q[1]*p[2]);
    Q[1][0]=0; Q[1][1]=1; Q[1][2]=0;	Q[1][3]=2*(q[1]*p[0]-2*q[0]*p[1]-q[3]*p[2]);	Q[1][4]=2*(q[0]*p[0]+q[2]*p[2]);				Q[1][5]=2*(q[3]*p[0]-2*q[2]*p[1]+q[1]*p[2]);	Q[1][6]=2*(q[2]*p[0]-q[0]*p[2]);
    Q[2][0]=0; Q[2][1]=0; Q[2][2]=1;	Q[2][3]=2*(q[2]*p[0]+q[3]*p[1]-2*q[0]*p[2]);	Q[2][4]=2*(-q[3]*p[0]+q[2]*p[1]-2*q[1]*p[2]);	Q[2][5]=2*(q[0]*p[0]+q[1]*p[1]);				Q[2][6]=2*(-q[1]*p[0]+q[0]*p[1]);
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::ComputeMa(Mat33& M, const Quat& q) const
{
    M[0][0]=0; M[0][1]=q[1]; M[0][2]=q[2];
    M[1][0]=q[1]; M[1][1]=-2*q[0]; M[1][2]=-q[3];
    M[2][0]=q[2]; M[2][1]=q[3]; M[2][2]=-2*q[0];
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::ComputeMb(Mat33& M, const Quat& q) const
{
    M[0][0]=-2*q[1]; M[0][1]=q[0]; M[0][2]=q[3];
    M[1][0]=q[0]; M[1][1]=0; M[1][2]=q[2];
    M[2][0]=-q[3]; M[2][1]=q[2]; M[2][2]=-2*q[1];
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::ComputeMc(Mat33& M, const Quat& q) const
{
    M[0][0]=-2*q[2]; M[0][1]=-q[3]; M[0][2]=q[0];
    M[1][0]=q[3]; M[1][1]=-2*q[2]; M[1][2]=q[1];
    M[2][0]=q[0]; M[2][1]=q[1]; M[2][2]=0;
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::ComputeMw(Mat33& M, const Quat& q) const
{
    M[0][0]=0; M[0][1]=-q[2]; M[0][2]=q[1];
    M[1][0]=q[2]; M[1][1]=0; M[1][2]=-q[0];
    M[2][0]=-q[1]; M[2][1]=q[0]; M[2][2]=0;
}

//rigid
template <class TIn, class TOut>
template<class T>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::RigidType, T> >::type SkinningMapping<TIn, TOut>::strainDeriv(Mat33 Ma,Mat33 Mb,Mat33 Mc,Mat33 Mw,Vec3 dw,Mat33 At,Mat33 F,Mat67 &B) const
{
    unsigned int k,m;
    Mat33 D; Mat33 FT=F.transposed();
    for(k=0; k<3; k++)  for(m=0; m<3; m++) B[m][k]+=dw[m]*F[k][m];
    for(k=0; k<3; k++)  B[3][k]+=(dw[0]*F[k][1]+dw[1]*F[k][0])/2.;
    for(k=0; k<3; k++)  B[4][k]+=(dw[1]*F[k][2]+dw[2]*F[k][1])/2.;
    for(k=0; k<3; k++)  B[5][k]+=(dw[0]*F[k][2]+dw[2]*F[k][0])/2.;
    D=FT*Ma*At; B[0][3]+=2*D[0][0]; B[1][3]+=2*D[1][1]; B[2][3]+=2*D[2][2]; B[3][3]+=D[0][1]+D[1][0]; B[4][3]+=D[1][2]+D[2][1]; B[5][3]+=D[0][2]+D[2][0];
    D=FT*Mb*At; B[0][4]+=2*D[0][0]; B[1][4]+=2*D[1][1]; B[2][4]+=2*D[2][2]; B[3][4]+=D[0][1]+D[1][0]; B[4][4]+=D[1][2]+D[2][1]; B[5][4]+=D[0][2]+D[2][0];
    D=FT*Mc*At; B[0][5]+=2*D[0][0]; B[1][5]+=2*D[1][1]; B[2][5]+=2*D[2][2]; B[3][5]+=D[0][1]+D[1][0]; B[4][5]+=D[1][2]+D[2][1]; B[5][5]+=D[0][2]+D[2][0];
    D=FT*Mw*At; B[0][6]+=2*D[0][0]; B[1][6]+=2*D[1][1]; B[2][6]+=2*D[2][2]; B[3][6]+=D[0][1]+D[1][0]; B[4][6]+=D[1][2]+D[2][1]; B[5][6]+=D[0][2]+D[2][0];
}

//affine
template <class TIn, class TOut>
template<class T>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::AffineType, T> >::type SkinningMapping<TIn, TOut>::strainDeriv(Vec3 dw,MatInAtx3 At,Mat33 F,Mat6xIn &B) const
{
    unsigned int k,l,m;
    // stretch
    for(k=0; k<3; k++)  for(m=0; m<3; m++) B[m][k]+=dw[m]*F[k][m];
    for(k=0; k<3; k++)  for(m=0; m<3; m++) for(l=0; l<3; l++) B[m][3*l+k+3]+=F[l][m]*At[k][m];
    // shear
    for(k=0; k<3; k++)  for(l=0; l<3; l++) B[3][3*l+k+3]+=0.5*(F[l][0]*At[k][1] + F[l][1]*At[k][0]);
    for(k=0; k<3; k++)  B[3][k]+=0.5*(dw[0]*F[k][1] + dw[1]*F[k][0]);
    for(k=0; k<3; k++)  for(l=0; l<3; l++) B[4][3*l+k+3]+=0.5*(F[l][1]*At[k][2]+F[l][2]*At[k][1]);
    for(k=0; k<3; k++)  B[4][k]+=0.5*(dw[1]*F[k][2] + dw[2]*F[k][1]);
    for(k=0; k<3; k++)  for(l=0; l<3; l++) B[5][3*l+k+3]+=0.5*(F[l][2]*At[k][0]+F[l][0]*At[k][2]);
    for(k=0; k<3; k++)  B[5][k]+=0.5*(dw[2]*F[k][0] + dw[0]*F[k][2]);
}

// quadratic
template <class TIn, class TOut>
template<class T>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::QuadraticType, T> >::type SkinningMapping<TIn, TOut>::strainDeriv(Vec3 dw,MatInAtx3 At,Mat33 F,Mat6xIn &B) const
{
    unsigned int k,l,m;
    // stretch
    for(k=0; k<3; k++)  for(m=0; m<3; m++) B[m][k]+=dw[m]*F[k][m];
    for(k=0; k<9; k++)  for(m=0; m<3; m++) for(l=0; l<3; l++) B[m][9*l+k+3]+=F[l][m]*At[k][m];
    // shear
    for(k=0; k<9; k++)  for(l=0; l<3; l++) B[3][9*l+k+3]+=0.5*(F[l][0]*At[k][1] + F[l][1]*At[k][0]);
    for(k=0; k<3; k++)  B[3][k]+=0.5*(dw[0]*F[k][1] + dw[1]*F[k][0]);
    for(k=0; k<9; k++)  for(l=0; l<3; l++) B[4][9*l+k+3]+=0.5*(F[l][1]*At[k][2]+F[l][2]*At[k][1]);
    for(k=0; k<3; k++)  B[4][k]+=0.5*(dw[1]*F[k][2] + dw[2]*F[k][1]);
    for(k=0; k<9; k++)  for(l=0; l<3; l++) B[5][9*l+k+3]+=0.5*(F[l][2]*At[k][0]+F[l][0]*At[k][2]);
    for(k=0; k<3; k++)  B[5][k]+=0.5*(dw[2]*F[k][0] + dw[0]*F[k][2]);
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::resizeMatrices()
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


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::initSamples() // Temporary (will be done in FrameSampler class)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    // vol and massDensity
    sofa::component::topology::DynamicSparseGridTopologyContainer* hexaContainer;
    this->getContext()->get( hexaContainer);
    //TODO get the volume from the FrameSampler !
    double volume = this->voxelVolume.getValue();
    //if ( hexaContainer && this->distOnGrid) volume = this->distOnGrid->initTargetStep.getValue()*this->distOnGrid->initTargetStep.getValue()*this->distOnGrid->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[0]*hexaContainer->voxelSize.getValue()[1]*hexaContainer->voxelSize.getValue()[2];
    const VecCoord& xto = *this->toModel->getX();
    const unsigned int& toSize = xto.size();
    this->vol.resize( toSize);
    for ( unsigned int i = 0; i < toSize; i++) this->vol[i] = volume;
    this->massDensity.resize( toSize);
    for ( unsigned int i = 0; i < toSize; i++) this->massDensity[i] = 1.0;

    if ( useElastons.getValue())
    {
        this->integ_Elaston.resize( toSize);
        for ( unsigned int i = 0; i < toSize; i++) // to update
        {
            double lx,ly,lz;
            if ( hexaContainer && this->distOnGrid)
            {
                lx=volume;
                ly=volume; //TODO get the volume from the FrameSampler !
                lz=volume;
            }
            else lx=ly=lz=pow(this->voxelVolume.getValue(),1./3.);
            this->integ_Elaston[i][0] = 1;
            this->integ_Elaston[i][4] = lx*lx/12.;
            this->integ_Elaston[i][7] = ly*ly/12.;
            this->integ_Elaston[i][9] = lz*lz/12.;
            this->integ_Elaston[i][20] = lx*lx*lx*lx/80.;
            this->integ_Elaston[i][21] = lx*lx*ly*ly/144.;
            this->integ_Elaston[i][22] = lx*lx*lz*lz/144.;
            this->integ_Elaston[i][23] = ly*ly*ly*ly/80.;
            this->integ_Elaston[i][24] = ly*ly*lz*lz/144.;
            this->integ_Elaston[i][25] = lz*lz*lz*lz/80.;
            this->integ_Elaston[i]=this->integ_Elaston[i]*lx*ly*lz;
        }

        this->deformationTensorsElaston.resize(toSize);
        this->Stilde_x.resize(toSize);
        this->Stilde_y.resize(toSize);
        this->Stilde_z.resize(toSize);
        for (unsigned int i = 0; i < toSize; ++i)
        {
            this->Stilde_x[i].resize(nbRef);
            this->Stilde_y[i].resize(nbRef);
            this->Stilde_z[i].resize(nbRef);
        }
        this->B_Elaston.resize(toSize);
        for (unsigned int i = 0; i < toSize; ++i)      this->B_Elaston[i].resize(nbRef);
    }
}


template <class TIn, class TOut>
template<class T>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::RigidType, T> >::type SkinningMapping<TIn, TOut>::precomputeMatrices(const RigidType&)
{
    const VecInCoord& xfrom0 = *this->fromModel->getX0();
    const VecCoord& xto0 = *this->toModel->getX0();
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    const VVD& m_weights = weights.getValue();
    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());
    VVMat33& m_ddweight = this->weightGradients2;

    resizeMatrices();
    initSamples();

    // Precompute matrices
    for ( unsigned int i=0 ; i<xto0.size(); i++ )
    {
        for ( unsigned int m=0 ; m<nbRef; m++ )
        {
            const int& idx=nbRef *i+m;
            const int& idxReps=m_reps[idx];

            const InCoord& xi0 = xfrom0[idxReps];
            Mat33 R0;	 QtoR( R0, xi0.getOrientation());
            Mat33 R0Inv;  R0Inv.invert (R0);

            for(int k=0; k<3; k++) for(int l=0; l<3; l++) (this->Atilde[i][m])[k][l] = (initPos[idx])[k] * (m_dweight[i][m])[l]  +  m_weights[i][m] * (R0Inv)[k][l];

            if ( useElastons.getValue())
            {
                for (int k=0; k<3; k++) for (int l=0; l<3; l++) (this->Stilde_x[i][m])[k][l] = (initPos[idx])[k] * (m_ddweight[i][m])[l][0] + (R0Inv)[k][0] * (m_dweight[i][m])[l]  +  m_dweight[i][m][0] * (R0Inv)[k][l];
                for (int k=0; k<3; k++) for (int l=0; l<3; l++) (this->Stilde_y[i][m])[k][l] = (initPos[idx])[k] * (m_ddweight[i][m])[l][1] + (R0Inv)[k][1] * (m_dweight[i][m])[l]  +  m_dweight[i][m][1] * (R0Inv)[k][l];
                for (int k=0; k<3; k++) for (int l=0; l<3; l++) (this->Stilde_z[i][m])[k][l] = (initPos[idx])[k] * (m_ddweight[i][m])[l][2] + (R0Inv)[k][2] * (m_dweight[i][m])[l]  +  m_dweight[i][m][2] * (R0Inv)[k][l];
            }

            Mat76 L; ComputeL(L, xi0.getOrientation());
            Mat37 Q; ComputeQ(Q, xi0.getOrientation(), initPos[idx]);

            this->J[i][m] = (InReal)m_weights[i][m] * Q * L;
            this->J0[i][m] = this->J[i][m];
        }
    }
}


template <class TIn, class TOut>
template<class T>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::AffineType, T> >::type SkinningMapping<TIn, TOut>::precomputeMatrices(const AffineType&)
{
    const VecInCoord& xfrom0 = *this->fromModel->getX0();
    const VecCoord& xto0 = *this->toModel->getX0();
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    const VVD& m_weights = weights.getValue();
    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());
    VVMat33& m_ddweight = this->weightGradients2;

    resizeMatrices();
    initSamples();

    // Precompute matrices
    for ( unsigned int i=0 ; i<xto0.size(); i++ )
    {
        for ( unsigned int m=0 ; m<nbRef; m++ )
        {
            const int& idx = nbRef *i+m;
            const int& idxReps = m_reps[idx];

            const InCoord& xi0 = xfrom0[idxReps];
            const Mat33& A0 = xi0.getAffine();
            Mat33 A0Inv; A0Inv.invert (A0);

            for(int k=0; k<3; k++) for(int l=0; l<3; l++) (this->Atilde[i][m])[k][l] = (initPos[idx])[k] * (m_dweight[i][m])[l]  +  m_weights[i][m] * (A0Inv)[k][l];

            if( useElastons.getValue())
            {
                for(int k=0; k<3; k++) for(int l=0; l<3; l++) (this->Stilde_x[i][m])[k][l] = (initPos[idx])[k] * (m_ddweight[i][m])[l][0] + (A0Inv)[k][0] * (m_dweight[i][m])[l]  +  m_dweight[i][m][0] * (A0Inv)[k][l];
                for(int k=0; k<3; k++) for(int l=0; l<3; l++) (this->Stilde_y[i][m])[k][l] = (initPos[idx])[k] * (m_ddweight[i][m])[l][1] + (A0Inv)[k][1] * (m_dweight[i][m])[l]  +  m_dweight[i][m][1] * (A0Inv)[k][l];
                for(int k=0; k<3; k++) for(int l=0; l<3; l++) (this->Stilde_z[i][m])[k][l] = (initPos[idx])[k] * (m_ddweight[i][m])[l][2] + (A0Inv)[k][2] * (m_dweight[i][m])[l]  +  m_dweight[i][m][2] * (A0Inv)[k][l];
            }

            Mat3xIn& Ji = this->J[i][m];
            Ji.fill(0);
            double val;
            for(int k=0; k<3; k++)
            {
                val = m_weights[i][m] * initPos[idx][k];
                Ji[0][k+3]=val;
                Ji[1][k+6]=val;
                Ji[2][k+9]=val;
                Ji[k][k]=m_weights[i][m];
            }
        }
    }
}


template <class TIn, class TOut>
template<class T>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::QuadraticType, T> >::type SkinningMapping<TIn, TOut>::precomputeMatrices(const QuadraticType&)
{
    const VecCoord& xto0 = *this->toModel->getX0();
    const unsigned int& nbRef = this->nbRefs.getValue();
    const VVD& m_weights = weights.getValue();
    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());
    VVMat33& m_ddweight = this->weightGradients2;

    resizeMatrices();
    initSamples();

    // Precompute matrices ( suppose that A0=I )
    for ( unsigned int i=0 ; i<xto0.size(); i++ )
    {
        for ( unsigned int m=0 ; m<nbRef; m++ )
        {
            const int& idx = nbRef *i+m;
            const Vec3& p0 = initPos[idx];
            Vec9 p2 = Vec9( p0[0], p0[1], p0[2], p0[0]*p0[0], p0[1]*p0[1], p0[2]*p0[2], p0[0]*p0[1], p0[1]*p0[2], p0[0]*p0[2]);

            MatInAtx3 A0Inv; // supposing that A0=I
            A0Inv[0][0] = 1;       A0Inv[0][1]=0;       A0Inv[0][2]=0;
            A0Inv[1][0] = 0;       A0Inv[1][1]=1;       A0Inv[1][2]=0;
            A0Inv[2][0] = 0;       A0Inv[2][1]=0;       A0Inv[2][2]=1;
            A0Inv[3][0] = 2*p2[0]; A0Inv[3][1]=0;       A0Inv[3][2]=0;
            A0Inv[4][0] = 0;       A0Inv[4][1]=2*p2[1]; A0Inv[4][2]=0;
            A0Inv[5][0] = 0;       A0Inv[5][1]=0;       A0Inv[5][2]=2*p2[2];
            A0Inv[6][0] = p2[1];   A0Inv[6][1]=p2[0];   A0Inv[6][2]=0;
            A0Inv[7][0] = 0;       A0Inv[7][1]=p2[2];   A0Inv[7][2]=p2[1];
            A0Inv[8][0] = p2[2];   A0Inv[8][1]=0;       A0Inv[8][2]=p2[0];
            for (int k=0; k<9; k++) for (int l=0; l<3; l++) (this->Atilde[i][m])[k][l] = p2[k] * m_dweight[i][m][l] + m_weights[i][m] * (A0Inv)[k][l];

            if( useElastons.getValue())
            {
                for(int k=0; k<9; k++) for(int l=0; l<3; l++) (this->Stilde_x[i][m])[k][l] = p2[k] * (m_ddweight[i][m])[l][0] + (A0Inv)[k][0] * (m_dweight[i][m])[l]  +  m_dweight[i][m][0] * (A0Inv)[k][l];
                for(int k=0; k<9; k++) for(int l=0; l<3; l++) (this->Stilde_y[i][m])[k][l] = p2[k] * (m_ddweight[i][m])[l][1] + (A0Inv)[k][1] * (m_dweight[i][m])[l]  +  m_dweight[i][m][1] * (A0Inv)[k][l];
                for(int k=0; k<9; k++) for(int l=0; l<3; l++) (this->Stilde_z[i][m])[k][l] = p2[k] * (m_ddweight[i][m])[l][2] + (A0Inv)[k][2] * (m_dweight[i][m])[l]  +  m_dweight[i][m][2] * (A0Inv)[k][l];
            }


            Mat3xIn& Ji = this->J[i][m];
            Ji.fill(0);
            double val;
            for(int k=0; k<9; k++)
            {
                val = m_weights[i][m] * p2[k];
                Ji[0][k+3] = val;
                Ji[1][k+12] = val;
                Ji[2][k+21] = val;
            }
            for(int k=0; k<3; k++)
                Ji[k][k] = m_weights[i][m];
        }
    }
}

#endif

// Generic Apply (old one in .inl)
template <class TIn, class TOut>
template<class TCoord>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::RigidType::Coord, TCoord > >::type
SkinningMapping<TIn, TOut>::_apply( typename Out::VecCoord& out, const sofa::helper::vector<typename RigidType::Coord>& in)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    const VVD& m_weights = weights.getValue();

    rotatedPoints.resize ( initPos.size() );
    out.resize ( initPos.size() / nbRef );
    for ( unsigned int i=0 ; i<out.size(); i++ )
    {
        out[i] = Coord();
        for ( unsigned int j = 0; j < nbRef; ++j)
        {
            const int& idx=nbRef *i+j;
            const int& idxReps=m_reps[idx];

            // Save rotated points for applyJ/JT
            rotatedPoints[idx] = in[idxReps].getOrientation().rotate ( initPos[idx] );

            // And add each reference frames contributions to the new position out[i]
            out[i] += ( in[idxReps ].getCenter() + rotatedPoints[idx] ) * m_weights[i][j];
        }

#ifdef SOFA_DEV
        if ( !(this->isMechanical() || this->computeAllMatrices.getValue())) continue;

        // update J
        Mat37 Q;
        VMat76 L;
        L.resize(nbRef);
        for(unsigned int j = 0; j < nbRef; j++)
        {
            const int& idx = nbRef * i + j;
            const int& idxReps = m_reps[idx];

            ComputeQ( Q, in[idxReps].getOrientation(), initPos[idx]);
            ComputeL( L[j], in[idxReps].getOrientation());
            this->J[i][j] = (InReal)m_weights[i][j] * Q * L[j];
        }

        // Physical computations
        if ( !this->computeAllMatrices.getValue()) continue;

        const SVector<SVector<GeoCoord> >& dw = this->weightGradients.getValue();
        VVMat33& ddw = this->weightGradients2;

        Mat33 F, Finv, E,A,S_x,S_y,S_z;
        F.fill ( 0 ); S_x.fill(0); S_y.fill(0); S_z.fill(0);
        E.fill ( 0 );
        for ( unsigned int j = 0 ; j < nbRef; ++j )
        {
            const int& idx = nbRef * i + j;
            const int& idxReps = m_reps[idx];

            in[idxReps ].getOrientation().toMatrix(A);
            for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) F[k][l]+=in[idxReps].getCenter()[k]*dw[i][j][l];
            F += A * this->Atilde[i][j];

            if( useElastons.getValue())
            {
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_x[k][l]+=in[idxReps].getCenter()[k]*ddw[i][j][l][0];
                S_x += A * this->Stilde_x[i][j];
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_y[k][l]+=in[idxReps].getCenter()[k]*ddw[i][j][l][1];
                S_y += A * this->Stilde_y[i][j];
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_z[k][l]+=in[idxReps].getCenter()[k]*ddw[i][j][l][2];
                S_z += A * this->Stilde_z[i][j];
            }
        }

        // strain and determinant
        this->det[i] = determinant ( F );
        for ( unsigned int k = 0; k < 3; ++k )
        {
            for ( unsigned int j = 0; j < 3; ++j )
                for ( unsigned int l = 0; l < 3; ++l )
                    E[k][j] += F[l][j] * F[l][k];

            E[k][k] -= 1.;
        }
        E /= 2.; // update E=1/2(F^TF-I)
        M33toV6(this->deformationTensors[i],E); // column form

        if(useElastons.getValue())
        {
            Vec6 v6;
            Mat610 &e_elastons=this->deformationTensorsElaston[i];
            unsigned int k,j,m;

            for(k=0; k<6; k++) e_elastons[k][9]=this->deformationTensors[i][k];
            E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=F[m][j]*S_x[m][k];
            M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][0]=v6[k];
            E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=F[m][j]*S_y[m][k];
            M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][1]=v6[k];
            E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=F[m][j]*S_z[m][k];
            M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][2]=v6[k];
            // extended elastons
            {
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_x[m][j]*S_x[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][3]=v6[k]/2;
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_y[m][j]*S_y[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][4]=v6[k]/2;
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_z[m][j]*S_z[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][5]=v6[k]/2;
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_x[m][j]*S_y[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][6]=v6[k];
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_y[m][j]*S_z[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][7]=v6[k];
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_z[m][j]*S_x[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][8]=v6[k];
            }
        }

        // update B and ddet
        for ( unsigned int j = 0 ; j < nbRef; ++j )
        {
            const int& idx = nbRef * i + j;
            const int& idxReps = m_reps[idx];

            Mat6xIn& B = this->B[i][j];
            const Mat33& At = this->Atilde[i][j];
            const Quat& rot = in[idxReps ].getOrientation();
            const Vec3& dWeight = dw[i][j];

            Mat33 Ma, Mb, Mc, Mw;
            ComputeMa( Ma, rot);
            ComputeMb( Mb, rot);
            ComputeMc( Mc, rot);
            ComputeMw( Mw, rot);

            Mat67 dE;
            dE.fill(0); strainDeriv<In>(Ma,Mb,Mc,Mw,dWeight,At,F,dE);
            B=dE*L[j];

            if(useElastons.getValue())
            {
                MatInx610& Be = this->B_Elaston[i][j];
                const Mat33& Stx = this->Stilde_x[i][j];
                const Mat33& Sty = this->Stilde_y[i][j];
                const Mat33& Stz = this->Stilde_z[i][j];
                const Vec3 ddx (ddw[i][j][0][0],ddw[i][j][1][0],ddw[i][j][2][0]);
                const Vec3 ddy (ddw[i][j][0][1],ddw[i][j][1][1],ddw[i][j][2][1]);
                const Vec3 ddz (ddw[i][j][0][2],ddw[i][j][1][2],ddw[i][j][2][2]);
                unsigned int k,m;

                Mat66 S;
                for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][9]=B[k][m];
                dE.fill(0); strainDeriv<In>(Ma,Mb,Mc,Mw,dWeight,At,S_x,dE); strainDeriv<In>(Ma,Mb,Mc,Mw,ddx,Stx,F,dE); S=dE*L[j];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][0]=S[k][m];
                dE.fill(0); strainDeriv<In>(Ma,Mb,Mc,Mw,dWeight,At,S_y,dE); strainDeriv<In>(Ma,Mb,Mc,Mw,ddy,Sty,F,dE); S=dE*L[j];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][1]=S[k][m];
                dE.fill(0); strainDeriv<In>(Ma,Mb,Mc,Mw,dWeight,At,S_z,dE); strainDeriv<In>(Ma,Mb,Mc,Mw,ddz,Stz,F,dE); S=dE*L[j];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][2]=S[k][m];

                // extended elastons
                {
                    dE.fill(0); strainDeriv<In>(Ma,Mb,Mc,Mw,ddx,Stx,S_x,dE); S=dE*L[j];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][3]=S[k][m];
                    dE.fill(0); strainDeriv<In>(Ma,Mb,Mc,Mw,ddy,Sty,S_y,dE); S=dE*L[j];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][4]=S[k][m];
                    dE.fill(0); strainDeriv<In>(Ma,Mb,Mc,Mw,ddz,Stz,S_z,dE); S=dE*L[j];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][5]=S[k][m];
                    dE.fill(0); strainDeriv<In>(Ma,Mb,Mc,Mw,ddx,Stx,S_y,dE); strainDeriv<In>(Ma,Mb,Mc,Mw,ddy,Sty,S_x,dE); S=dE*L[j];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][6]=S[k][m];
                    dE.fill(0); strainDeriv<In>(Ma,Mb,Mc,Mw,ddy,Sty,S_z,dE); strainDeriv<In>(Ma,Mb,Mc,Mw,ddz,Stz,S_y,dE); S=dE*L[j];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][7]=S[k][m];
                    dE.fill(0); strainDeriv<In>(Ma,Mb,Mc,Mw,ddz,Stz,S_x,dE); strainDeriv<In>(Ma,Mb,Mc,Mw,ddx,Stx,S_z,dE); S=dE*L[j];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][8]=S[k][m];
                }

            }
            // Compute ddet
            // if(computevolpres)
            {
                invertMatrix ( Finv, F );
                Vec6 &ddet = this->ddet[i][j];
                ddet.fill(0);
                Vec7 u7; u7.fill(0);

                Ma=Ma*At;
                Mb=Mb*At;
                Mc=Mc*At;
                Mw=Mw*At;
                for(unsigned int k=0; k<3; k++)
                {
                    u7[0]+=Finv[k][0]*dWeight[k]; u7[1]+=Finv[k][1]*dWeight[k]; u7[2]+=Finv[k][2]*dWeight[k];
                    for(unsigned int m=0; m<3; m++) { u7[3]+=2*Finv[k][m]*Ma[m][k]; u7[4]+=2*Finv[k][m]*Mb[m][k]; u7[5]+=2*Finv[k][m]*Mc[m][k]; u7[6]+=2*Finv[k][m]*Mw[m][k]; }
                }
                for(unsigned int k=0; k<6; k++) for(unsigned int m=0; m<7; m++) ddet[k]+=u7[m]*L[j][m][k];
                for(unsigned int k=0; k<6; k++) ddet[k]=this->det[i] * ddet[k];
            }

        }
#endif
    }

}

#ifdef SOFA_DEV

// Apply for Affine types
template <class TIn, class TOut>
template<class TCoord>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::AffineType::Coord, TCoord> >::type SkinningMapping<TIn, TOut>::_apply( typename Out::VecCoord& out, const sofa::helper::vector<typename AffineType::Coord>& in)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    const VVD& m_weights = weights.getValue();

    rotatedPoints.resize ( initPos.size() );
    out.resize ( initPos.size() / nbRef );

    // Resize matrices in case of Frame insertion
    if ( this->computeAllMatrices.getValue())
    {
        this->det.resize(out.size());
        this->deformationTensors.resize(out.size());
        this->B.resize(out.size());
        for(unsigned int i = 0; i < out.size(); ++i)
            this->B[i].resize(nbRef);
    }

    for ( unsigned int i = 0 ; i < out.size(); i++ )
    {
        // Point transformation (apply)
        out[i] = Coord();

        for ( unsigned int j = 0 ; j < nbRef; ++j )
        {
            const int& idx = nbRef * i + j;
            const int& idxReps = m_reps[idx];

            // Save rotated points for applyJ/JT
            rotatedPoints[idx] = in[idxReps].getAffine() * initPos[idx];

            // And add each reference frames contributions to the new position out[i]
            out[i] += ( in[idxReps ].getCenter() + rotatedPoints[idx] ) * m_weights[i][j];
        }

        // Physical computations
        if ( !this->computeAllMatrices.getValue()) continue;

        const SVector<SVector<GeoCoord> >& dw = this->weightGradients.getValue();
        VVMat33& ddw = this->weightGradients2;

        Mat33 F, Finv, E,A,S_x,S_y,S_z;
        F.fill ( 0 ); S_x.fill(0); S_y.fill(0); S_z.fill(0);
        E.fill ( 0 );

        for ( unsigned int j = 0 ; j < nbRef; ++j )
        {
            const int& idx = nbRef * i + j;
            const int& idxReps = m_reps[idx];

            for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) F[k][l]+=in[idxReps].getCenter()[k]*dw[i][j][l];
            F += in[idxReps].getAffine() * this->Atilde[i][j];

            if( useElastons.getValue())
            {
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_x[k][l]+=in[idxReps].getCenter()[k]*ddw[i][j][l][0];
                S_x += in[idxReps ].getAffine() * this->Stilde_x[i][j];
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_y[k][l]+=in[idxReps].getCenter()[k]*ddw[i][j][l][1];
                S_y += in[idxReps ].getAffine() * this->Stilde_y[i][j];
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_z[k][l]+=in[idxReps].getCenter()[k]*ddw[i][j][l][2];
                S_z += in[idxReps ].getAffine() * this->Stilde_z[i][j];
            }
        }

        // strain and determinant
        this->det[i] = determinant ( F );
        for ( unsigned int k = 0; k < 3; ++k )
        {
            for ( unsigned int j = 0; j < 3; ++j )
                for ( unsigned int l = 0; l < 3; ++l )
                    E[k][j] += F[l][j] * F[l][k];

            E[k][k] -= 1.;
        }
        E /= 2.; // update E=1/2(U^TU-I)
        M33toV6(this->deformationTensors[i],E); // column form

        if(useElastons.getValue())
        {
            Vec6 v6;
            Mat610 &e_elastons=this->deformationTensorsElaston[i];
            unsigned int k,j,m;

            for(k=0; k<6; k++) e_elastons[k][9]=this->deformationTensors[i][k];
            E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=F[m][j]*S_x[m][k];
            M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][0]=v6[k];
            E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=F[m][j]*S_y[m][k];
            M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][1]=v6[k];
            E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=F[m][j]*S_z[m][k];
            M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][2]=v6[k];
            // extended elastons
            {
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_x[m][j]*S_x[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][3]=v6[k]/2;
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_y[m][j]*S_y[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][4]=v6[k]/2;
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_z[m][j]*S_z[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][5]=v6[k]/2;
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_x[m][j]*S_y[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][6]=v6[k];
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_y[m][j]*S_z[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][7]=v6[k];
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_z[m][j]*S_x[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][8]=v6[k];
            }
        }

        // update B and ddet
        for ( unsigned int j = 0 ; j < nbRef; ++j )
        {
            Mat6xIn& B = this->B[i][j];
            const Vec3& dWeight = dw[i][j];
            const Mat33& At = this->Atilde[i][j];
            B.fill(0); strainDeriv<In>(dWeight,At,F,B);

            if(useElastons.getValue())
            {
                MatInx610& Be = this->B_Elaston[i][j];
                const MatInAtx3& Stx = this->Stilde_x[i][j];
                const MatInAtx3& Sty = this->Stilde_y[i][j];
                const MatInAtx3& Stz = this->Stilde_z[i][j];
                const Vec3 ddx (ddw[i][j][0][0],ddw[i][j][1][0],ddw[i][j][2][0]);
                const Vec3 ddy (ddw[i][j][0][1],ddw[i][j][1][1],ddw[i][j][2][1]);
                const Vec3 ddz (ddw[i][j][0][2],ddw[i][j][1][2],ddw[i][j][2][2]);
                unsigned int k,m;

                Mat6xIn S;
                for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][9]=B[k][m];
                S.fill(0); strainDeriv<In>(dWeight,At,S_x,S); strainDeriv<In>(ddx,Stx,F,S); for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][0]=S[k][m];
                S.fill(0); strainDeriv<In>(dWeight,At,S_y,S); strainDeriv<In>(ddy,Sty,F,S); for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][1]=S[k][m];
                S.fill(0); strainDeriv<In>(dWeight,At,S_z,S); strainDeriv<In>(ddz,Stz,F,S); for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][2]=S[k][m];

                // extended elastons
                {
                    S.fill(0); strainDeriv<In>(ddx,Stx,S_x,S);	for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][3]=S[k][m];
                    S.fill(0); strainDeriv<In>(ddy,Sty,S_y,S);	for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][4]=S[k][m];
                    S.fill(0); strainDeriv<In>(ddz,Stz,S_z,S);	for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][5]=S[k][m];
                    S.fill(0); strainDeriv<In>(ddx,Stx,S_y,S); strainDeriv<In>(ddy,Sty,S_x,S);	for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][6]=S[k][m];
                    S.fill(0); strainDeriv<In>(ddy,Sty,S_z,S); strainDeriv<In>(ddz,Stz,S_y,S); 	for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][7]=S[k][m];
                    S.fill(0); strainDeriv<In>(ddz,Stz,S_x,S); strainDeriv<In>(ddx,Stx,S_z,S); 	for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][8]=S[k][m];
                }
            }

            // Compute ddet
            // if(computevolpres)
            {
                invertMatrix ( Finv, F );
                VecIn &ddet = this->ddet[i][j];
                ddet.fill(0);
                for (unsigned int k = 0; k < 3; k++ ) for (unsigned int l = 0; l < 3; l++ ) ddet[k] += dWeight [l] * Finv[l][k];
                for (unsigned int k = 0; k < 3; k++ ) for (unsigned int m = 0; m < 3; m++ ) for (unsigned int l = 0; l < 3; l++ ) ddet[m+3*k+3] += At[m][l] * Finv[l][k];
                for(unsigned int k=0; k<12; k++) ddet[k]=this->det[i] * ddet[k];
            }
        }
    }
}


// Apply for Quadratic types
template <class TIn, class TOut>
template<class TCoord>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::QuadraticType::Coord, TCoord> >::type SkinningMapping<TIn, TOut>::_apply( typename Out::VecCoord& out, const sofa::helper::vector<typename QuadraticType::Coord>& in)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    const VVD& m_weights = weights.getValue();

    rotatedPoints.resize ( initPos.size() );
    out.resize ( initPos.size() / nbRef );

    // Resize matrices in case of Frame insertion
    if ( this->computeAllMatrices.getValue())
    {
        this->det.resize(out.size());
        this->deformationTensors.resize(out.size());
        this->B.resize(out.size());
        for(unsigned int i = 0; i < out.size(); ++i)
            this->B[i].resize(nbRef);
    }

    for ( unsigned int i = 0 ; i < out.size(); i++ )
    {
        // Point transformation (apply)
        out[i] = Coord();

        for ( unsigned int j = 0 ; j < nbRef; ++j )
        {
            const int& idx = nbRef * i + j;
            const int& idxReps = m_reps[idx];

            const Vec3& p0 = initPos[idx];
            Vec9 p2 = Vec9( p0[0], p0[1], p0[2], p0[0]*p0[0], p0[1]*p0[1], p0[2]*p0[2], p0[0]*p0[1], p0[1]*p0[2], p0[0]*p0[2]);

            // Save rotated points for applyJ/JT
            rotatedPoints[idx] = in[idxReps].getQuadratic() * p2;

            // And add each reference frames contributions to the new position out[i]
            out[i] += ( in[idxReps ].getCenter() + rotatedPoints[idx] ) * m_weights[i][j];
        }

        // Physical computations
        if ( !this->computeAllMatrices.getValue()) continue;

        const SVector<SVector<GeoCoord> >& dw = this->weightGradients.getValue();
        VVMat33& ddw = this->weightGradients2;

        Mat33 F, Finv, E,A,S_x,S_y,S_z;
        F.fill ( 0 ); S_x.fill(0); S_y.fill(0); S_z.fill(0);
        E.fill ( 0 );
        for ( unsigned int j = 0 ; j < nbRef; ++j )
        {
            const int& idx = nbRef * i + j;
            const int& idxReps = m_reps[idx];

            for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) F[k][l]+=in[idxReps].getCenter()[k]*dw[i][j][l];
            F += in[idxReps ].getQuadratic() * this->Atilde[i][j];

            if( useElastons.getValue())
            {
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_x[k][l]+=in[idxReps].getCenter()[k]*ddw[i][j][l][0];
                S_x += in[idxReps].getQuadratic() * this->Stilde_x[i][j];
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_y[k][l]+=in[idxReps].getCenter()[k]*ddw[i][j][l][1];
                S_y += in[idxReps].getQuadratic() * this->Stilde_y[i][j];
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_z[k][l]+=in[idxReps].getCenter()[k]*ddw[i][j][l][2];
                S_z += in[idxReps].getQuadratic() * this->Stilde_z[i][j];
            }
        }

        // strain and determinant
        this->det[i] = determinant ( F );

        for ( unsigned int k = 0; k < 3; ++k )
        {
            for ( unsigned int j = 0; j < 3; ++j )
                for ( unsigned int l = 0; l < 3; ++l )
                    E[k][j] += F[l][j] * F[l][k];

            E[k][k] -= 1.;
        }
        E /= 2.; // update E=1/2(U^TU-I)
        M33toV6(this->deformationTensors[i],E); // column form

        if(useElastons.getValue())
        {
            Vec6 v6;
            Mat610 &e_elastons=this->deformationTensorsElaston[i];
            unsigned int k,j,m;

            for(k=0; k<6; k++) e_elastons[k][9]=this->deformationTensors[i][k];
            E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=F[m][j]*S_x[m][k];
            M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][0]=v6[k];
            E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=F[m][j]*S_y[m][k];
            M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][1]=v6[k];
            E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=F[m][j]*S_z[m][k];
            M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][2]=v6[k];
            // extended elastons
            {
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_x[m][j]*S_x[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][3]=v6[k]/2;
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_y[m][j]*S_y[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][4]=v6[k]/2;
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_z[m][j]*S_z[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][5]=v6[k]/2;
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_x[m][j]*S_y[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][6]=v6[k];
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_y[m][j]*S_z[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][7]=v6[k];
                E.fill(0); for(k=0; k<3; k++) for(j=0; j<3; j++) for(m=0; m<3; m++) E[k][j]+=S_z[m][j]*S_x[m][k];
                M33toV6(v6,E); for(k=0; k<6; k++) e_elastons[k][8]=v6[k];
            }
        }

        // update B and ddet
        for ( unsigned int j = 0 ; j < nbRef; ++j )
        {
            Mat6xIn& B = this->B[i][j];
            const Vec3& dWeight = dw[i][j];
            const MatInAtx3& At = this->Atilde[i][j];
            B.fill(0); strainDeriv<In>(dWeight,At,F,B);


            if(useElastons.getValue())
            {
                MatInx610& Be = this->B_Elaston[i][j];
                const MatInAtx3& Stx = this->Stilde_x[i][j];
                const MatInAtx3& Sty = this->Stilde_y[i][j];
                const MatInAtx3& Stz = this->Stilde_z[i][j];
                const Vec3 ddx (ddw[i][j][0][0],ddw[i][j][1][0],ddw[i][j][2][0]);
                const Vec3 ddy (ddw[i][j][0][1],ddw[i][j][1][1],ddw[i][j][2][1]);
                const Vec3 ddz (ddw[i][j][0][2],ddw[i][j][1][2],ddw[i][j][2][2]);
                unsigned int k,m;
                Mat6xIn S;

                for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][9]=B[k][m];
                S.fill(0); strainDeriv<In>(dWeight,At,S_x,S); strainDeriv<In>(ddx,Stx,F,S); for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][0]=S[k][m];
                S.fill(0); strainDeriv<In>(dWeight,At,S_y,S); strainDeriv<In>(ddy,Sty,F,S); for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][1]=S[k][m];
                S.fill(0); strainDeriv<In>(dWeight,At,S_z,S); strainDeriv<In>(ddz,Stz,F,S); for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][2]=S[k][m];

                // extended elastons
                {
                    S.fill(0); strainDeriv<In>(ddx,Stx,S_x,S);	for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][3]=S[k][m];
                    S.fill(0); strainDeriv<In>(ddy,Sty,S_y,S);	for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][4]=S[k][m];
                    S.fill(0); strainDeriv<In>(ddz,Stz,S_z,S);	for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][5]=S[k][m];
                    S.fill(0); strainDeriv<In>(ddx,Stx,S_y,S); strainDeriv<In>(ddy,Sty,S_x,S);	for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][6]=S[k][m];
                    S.fill(0); strainDeriv<In>(ddy,Sty,S_z,S); strainDeriv<In>(ddz,Stz,S_y,S);	for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][7]=S[k][m];
                    S.fill(0); strainDeriv<In>(ddz,Stz,S_x,S); strainDeriv<In>(ddx,Stx,S_z,S);	for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][8]=S[k][m];
                }
            }

            // Compute ddet
            // if(computevolpres)
            {
                invertMatrix ( Finv, F );
                VecIn &ddet = this->ddet[i][j];
                ddet.fill(0);
                for (unsigned int k = 0; k < 3; k++ ) for (unsigned int l = 0; l < 3; l++ ) ddet[k] += dWeight [l] * Finv[l][k];
                for (unsigned int k = 0; k < 3; k++ ) for (unsigned int m = 0; m < 9; m++ ) for (unsigned int l = 0; l < 3; l++ ) ddet[m+9*k+3] += At[m][l] * Finv[l][k];
                for(unsigned int k=0; k<30; k++) ddet[k]=this->det[i] * ddet[k];
            }
        }
    }
}

#endif

template <class TIn, class TOut>
template<class TDeriv>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::RigidType::Deriv, TDeriv> >::type SkinningMapping<TIn, TOut>::_applyJ( typename Out::VecDeriv& out, const sofa::helper::vector<typename RigidType::Deriv>& in)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    const VVD& m_weights = weights.getValue();
    const VecCoord& xto = *this->toModel->getX();
    out.resize ( xto.size() );
    Deriv v,omega;

    if ( ! ( maskTo->isInUse() ) )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            out[i] = Deriv();
            for ( unsigned int j=0 ; j<nbRef; j++ )
            {
                const int idx=nbRef *i+j;
                const int idxReps=m_reps[idx];

                v = getVCenter(in[idxReps]);
                omega = getVOrientation(in[idxReps]);
                out[i] += ( v - cross ( rotatedPoints[idx],omega ) ) * m_weights[i][j];
            }
        }
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;
        for ( it=indices.begin(); it!=indices.end(); it++ )
        {
            const int i= ( int ) ( *it );
            out[i] = Deriv();
            for ( unsigned int j=0 ; j<nbRef; j++ )
            {
                const int idx=nbRef *i+j;
                const int idxReps=m_reps[idx];

                v = getVCenter(in[idxReps]);
                omega = getVOrientation(in[idxReps]);
                out[i] += ( v - cross ( rotatedPoints[idx],omega ) ) * m_weights[i][j];
            }
        }
    }
}

#ifdef SOFA_DEV

template <class TIn, class TOut>
template<class TDeriv>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::AffineType::Deriv, TDeriv> >::type SkinningMapping<TIn, TOut>::_applyJ( typename Out::VecDeriv& out, const sofa::helper::vector<typename AffineType::Deriv>& in)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    const VecCoord& xto = *this->toModel->getX();
    out.resize ( xto.size() );
    Deriv v;
    //    typename In::Deriv::Affine omega;

    if ( ! ( maskTo->isInUse() ) )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            out[i] = Deriv();
            for ( unsigned int j=0 ; j<nbRef; j++ )
            {
                /*                VecIn speed;
                for (unsigned int k = 0; k < InDOFs; ++k)
                speed[k]  = in[j][k];

                Vec3 f = ( this->J[i][j] * speed );

                out[i] += Deriv ( f[0], f[1], f[2] );*/
                const int idxReps=m_reps[nbRef *i+j];
                out[i] += this->J[i][j] * in[idxReps];
            }
        }
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;
        for ( it=indices.begin(); it!=indices.end(); it++ )
        {
            const int i= ( int ) ( *it );
            out[i] = Deriv();
            for ( unsigned int j=0 ; j<nbRef; j++ )
            {
                /*                VecIn speed;
                for (unsigned int k = 0; k < InDOFs; ++k)
                speed[k]  = in[j][k];

                Vec3 f = ( this->J[i][j] * speed );

                out[i] += Deriv ( f[0], f[1], f[2] );*/
                const int idxReps=m_reps[nbRef *i+j];
                out[i] += this->J[i][j] * in[idxReps];
            }
        }
    }
}


template <class TIn, class TOut>
template<class TDeriv>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::QuadraticType::Deriv, TDeriv> >::type SkinningMapping<TIn, TOut>::_applyJ( typename Out::VecDeriv& out, const sofa::helper::vector<typename QuadraticType::Deriv>& in)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    const VecCoord& xto = *this->toModel->getX();
    out.resize ( xto.size() );
    Deriv v;
    //    typename In::Deriv::Quadratic omega;

    if ( ! ( maskTo->isInUse() ) )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            out[i] = Deriv();
            for ( unsigned int j=0 ; j<nbRef; j++ )
            {
                const int idxReps=m_reps[nbRef *i+j];
                VecIn speed;
                for (unsigned int k = 0; k < InDOFs; ++k)
                    speed[k]  = in[idxReps][k];

                Vec3 f = ( this->J[i][j] * speed );

                out[i] += Deriv ( f[0], f[1], f[2] );
            }
        }
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;
        for ( it=indices.begin(); it!=indices.end(); it++ )
        {
            const int i= ( int ) ( *it );
            out[i] = Deriv();
            for ( unsigned int j=0 ; j<nbRef; j++ )
            {
                const int idxReps=m_reps[nbRef *i+j];
                VecIn speed;
                for (unsigned int k = 0; k < InDOFs; ++k)
                    speed[k]  = in[idxReps][k];

                Vec3 f = ( this->J[i][j] * speed );

                out[i] += Deriv ( f[0], f[1], f[2] );
            }
        }
    }
}

#endif

template <class TIn, class TOut>
template<class TDeriv>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::RigidType::Deriv, TDeriv> >::type SkinningMapping<TIn, TOut>::_applyJT( sofa::helper::vector<typename RigidType::Deriv>& out, const typename Out::VecDeriv& in)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    const VVD& m_weights = weights.getValue();

    Deriv v,omega;
    if ( ! ( maskTo->isInUse() ) )
    {
        maskFrom->setInUse ( false );
        for ( unsigned int i=0; i<in.size(); i++ )
        {
            for ( unsigned int j=0 ; j<nbRef; j++ )
            {
                Deriv f = in[i];
                v = f;
                const int idx=nbRef *i+j;
                const int idxReps=m_reps[idx];
                omega = cross ( rotatedPoints[idx],f );
                getVCenter(out[idxReps]) += v * m_weights[i][j];
                getVOrientation(out[idxReps]) += omega * m_weights[i][j];
            }
        }
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;
        for ( it=indices.begin(); it!=indices.end(); it++ )
        {
            const int i= ( int ) ( *it );
            for ( unsigned int j=0 ; j<nbRef; j++ )
            {
                Deriv f = in[i];
                v = f;
                const int idx=nbRef *i+j;
                const int idxReps=m_reps[idx];
                omega = cross ( rotatedPoints[idx],f );
                getVCenter(out[idxReps]) += v * m_weights[i][j];
                getVOrientation(out[idxReps]) += omega * m_weights[i][j];

                maskFrom->insertEntry ( idxReps );
            }
        }
    }
}

#ifdef SOFA_DEV

template <class TIn, class TOut>
template<class TDeriv>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::AffineType::Deriv, TDeriv> >::type SkinningMapping<TIn, TOut>::_applyJT( sofa::helper::vector<typename AffineType::Deriv>& out, const typename Out::VecDeriv& in)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    Deriv v;
    //    typename In::Deriv::Affine omega;
    if ( ! ( maskTo->isInUse() ) )
    {
        maskFrom->setInUse ( false );
        for ( unsigned int i=0; i<in.size(); i++ ) // VecType
        {
            for ( unsigned int j=0 ; j<nbRef; j++ ) // AffineType
            {
                const int idxReps=m_reps[nbRef *i+j];
                out[idxReps] += this->J[i][j].multTranspose( in[i] );
            }
        }
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;
        for ( it=indices.begin(); it!=indices.end(); it++ ) // VecType
        {
            const int i= ( int ) ( *it );
            for ( unsigned int j=0 ; j<nbRef; j++ ) // AffineType
            {
                const int idxReps=m_reps[nbRef *i+j];
                out[idxReps] += this->J[i][j].multTranspose( in[i] );
            }
        }
    }
}


template <class TIn, class TOut>
template<class TDeriv>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::QuadraticType::Deriv, TDeriv> >::type SkinningMapping<TIn, TOut>::_applyJT( sofa::helper::vector<typename QuadraticType::Deriv>& out, const typename Out::VecDeriv& in)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    Deriv v;
    //    typename In::Deriv::Quadratic omega;
    if ( ! ( maskTo->isInUse() ) )
    {
        maskFrom->setInUse ( false );
        for ( unsigned int i=0; i<in.size(); i++ ) // VecType
        {
            for ( unsigned int j=0 ; j<nbRef; j++ ) // QuadraticType
            {
                const int idxReps=m_reps[nbRef *i+j];
                out[idxReps] += this->J[i][j].multTranspose(in[i]);
            }
        }
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;
        for ( it=indices.begin(); it!=indices.end(); it++ ) // VecType
        {
            const int i= ( int ) ( *it );
            for ( unsigned int j=0 ; j<nbRef; j++ ) // QuadraticType
            {
                const int idxReps=m_reps[nbRef *i+j];
                out[idxReps] += this->J[i][j].multTranspose(in[i]);
            }
        }
    }
}

#endif

template <class TIn, class TOut>
template<class TMatrixDeriv>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::RigidType::MatrixDeriv, TMatrixDeriv> >::type SkinningMapping<TIn, TOut>::_applyJT_Matrix( typename RigidType::MatrixDeriv& out, const typename Out::MatrixDeriv& in)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    const VVD& m_weights = weights.getValue();
    const unsigned int nbp = this->fromModel->getX()->size();
    Deriv omega;
    typename In::VecDeriv v;
    vector<bool> flags;

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        v.clear();
        v.resize(nbp);
        flags.clear();
        flags.resize(nbp);

        typename In::MatrixDeriv::RowIterator o = out.end();

        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        for (typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            unsigned int indexPoint = colIt.index();
            Deriv data = ( Deriv ) colIt.val();

            for (unsigned int j = 0 ; j < nbRef; j++)
            {
                const int idxReps=m_reps[nbRef *indexPoint+j];
                omega = cross(rotatedPoints[nbRef * indexPoint + j], data);
                flags[idxReps] = true;
                getVCenter(v[idxReps]) += data * m_weights[indexPoint][j];
                getVOrientation(v[idxReps]) += omega * m_weights[indexPoint][j];
            }

            for (unsigned int j = 0 ; j < nbp; j++)
            {
                if (flags[j])
                {
                    // Create an unique new line for each contraint
                    if (o == out.end())
                    {
                        o = out.writeLine(rowIt.index());
                    }

                    o.addCol(j, v[j]);
                }
            }
        }
    }
}

#ifdef SOFA_DEV

template <class TIn, class TOut>
template<class TMatrixDeriv>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::AffineType::MatrixDeriv, TMatrixDeriv> >::type SkinningMapping<TIn, TOut>::_applyJT_Matrix( typename AffineType::MatrixDeriv& out, const typename Out::MatrixDeriv& in)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    //    typename In::Deriv::Affine omega;
    typename In::VecDeriv v;

    if ( !this->enableSkinning.getValue())
        return;

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename In::MatrixDeriv::RowIterator o = out.end();

        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        for (typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            const unsigned int indexPoint = colIt.index(); // Point
            const Deriv data = colIt.val();

            for (unsigned int j=0; j<nbRef; ++j) // Affine
            {
                const int idxReps=m_reps[nbRef *indexPoint+j];
                InDeriv value = this->J[indexPoint][j].multTranspose(data);
                o.addCol(idxReps, value);
            }
        }
    }
}


template <class TIn, class TOut>
template<class TMatrixDeriv>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::QuadraticType::MatrixDeriv, TMatrixDeriv> >::type SkinningMapping<TIn, TOut>::_applyJT_Matrix( typename QuadraticType::MatrixDeriv& out, const typename Out::MatrixDeriv& in)
{
    const unsigned int& nbRef = this->nbRefs.getValue();
    const vector<unsigned int>& m_reps = this->repartition.getValue();
    //    typename In::Deriv::Quadratic omega;
    typename In::VecDeriv v;

    if ( !this->enableSkinning.getValue())
        return;

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename In::MatrixDeriv::RowIterator o = out.end();

        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        for (typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            const unsigned int indexPoint = colIt.index();
            const Deriv data = colIt.val();

            for (unsigned int j=0; j<nbRef; ++j)
            {
                const int idxReps=m_reps[nbRef *indexPoint+j];
                InDeriv value = this->J[indexPoint][j].multTranspose(data);
                o.addCol(idxReps, value);
            }
        }
    }
}

#endif



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
