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
#include <sofa/component/topology/HexahedronGeodesicalDistance.inl>
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
    , repartition ( initData ( &repartition,"repartition","repartition between input DOFs and skinned vertices" ) )
    , weights ( initData ( &weights,"weights","weights list for the influences of the references Dofs" ) )
    , weightGradients ( initData ( &weightGradients,"weightGradients","weight gradients list for the influences of the references Dofs" ) )
    , nbRefs ( initData ( &nbRefs, ( unsigned ) 3,"nbRefs","nb references for skinning" ) )
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
    geoDist = NULL;
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

    const vector<int>& m_reps = repartition.getValue();

    initPos.resize ( xto.size() * nbRefs.getValue() );
    for ( unsigned int i = 0; i < xto.size(); i++ )
        for ( unsigned int m = 0; m < nbRefs.getValue(); m++ )
        {
            const int& idx=nbRefs.getValue() *i+m;
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
            sofa::helper::OptionsGroup* geoDistanceTypeOption = geoDist->distanceType.beginEdit();
            if ( distanceType.getValue().getSelectedId() == DISTANCE_GEODESIC) geoDistanceTypeOption->setSelectedItem(TYPE_GEODESIC);
            if ( distanceType.getValue().getSelectedId() == DISTANCE_HARMONIC) geoDistanceTypeOption->setSelectedItem(TYPE_HARMONIC);
            if ( distanceType.getValue().getSelectedId() == DISTANCE_STIFFNESS_DIFFUSION) geoDistanceTypeOption->setSelectedItem(TYPE_STIFFNESS_DIFFUSION);
            if ( distanceType.getValue().getSelectedId() == DISTANCE_HARMONIC_STIFFNESS) geoDistanceTypeOption->setSelectedItem(TYPE_HARMONIC_STIFFNESS);
            geoDist->distanceType.endEdit();
        }
        geoDist->computeDistanceMap ( tmpFrom );
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
        geoDist->getDistances ( distances, distGradients, goals );
        break;
    }
#endif
    default: {}
    }
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::sortReferences( vector<int>& references)
{
    const VecCoord& xto = ( this->toModel->getX0()->size() == 0)?*this->toModel->getX():*this->toModel->getX0();
    const VecInCoord& xfrom = *this->fromModel->getX0();
    const unsigned int& nbRef = nbRefs.getValue();

    references.clear();
    references.resize ( nbRefs.getValue() *xto.size() );
    for ( unsigned int i=0; i<nbRefs.getValue() *xto.size(); i++ )
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
    const unsigned int& nbRef = nbRefs.getValue();
    VVD& m_weights = * ( weights.beginEdit() );
    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());
    VVMat33& m_ddweight = this->weightGradients2;

    // Normalise weights & dweights
    for (unsigned int j = 0; j < xtoSize; ++j)
    {
        double sumWeights = 0,wn;
        Vec3 sumGrad,dwn;			sumGrad.fill(0);
        Mat33 sumGrad2,ddwn;		sumGrad2.fill(0);

        // Compute norm
        for (unsigned int i = 0; i < nbRef; ++i)
        {
            sumWeights += m_weights[i][j];
            sumGrad += m_dweight[i][j];
            sumGrad2 += m_ddweight[i][j];
        }

        // Normalise
        if(sumWeights!=0)
            for (unsigned int i = 0; i < nbRef; ++i)
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
        this->getContext()->get ( geoDist, core::objectmodel::BaseContext::SearchRoot );
        if ( !geoDist )
        {
            serr << "Can not find the geodesical distance component: distances used are euclidian." << sendl;
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
        /* Temporary remove optimistaion. TODO: reactivate this when the different types will be instanciated
          if ( wheightingType.getValue().getSelectedId() == WEIGHT_LINEAR || wheightingType.getValue().getSelectedId() == WEIGHT_HERMITE )
              nbRefs.setValue ( 2 );

          if ( wheightingType.getValue().getSelectedId() == WEIGHT_SPLINE)
              nbRefs.setValue ( 4 );

          if ( xfrom.size() < nbRefs.getValue())
              nbRefs.setValue ( xfrom.size() );
        */
        nbRefs.setValue ( xfrom.size() );

        computeDistances();
        vector<int>& m_reps = * ( repartition.beginEdit() );
        sortReferences ( m_reps);
        repartition.endEdit();
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

    const vector<int>& m_reps = repartition.getValue();

    m_weights.resize ( xfrom.size() );    for ( unsigned int i=0; i<xfrom.size(); i++ )        m_weights[i].resize ( xto.size() );
    m_dweight.resize ( xfrom.size() );    for ( unsigned int i=0; i<xfrom.size(); i++ )        m_dweight[i].resize ( xto.size() );
    m_ddweight.resize( xfrom.size() );    for ( unsigned int i=0; i<xfrom.size(); i++ )        m_ddweight[i].resize( xto.size() );

    switch ( wheightingType.getValue().getSelectedId() )
    {
    case WEIGHT_NONE:
    {
        for ( unsigned int j=0; j<xto.size(); j++ )
            for ( unsigned int i=0; i<nbRefs.getValue(); i++ )
            {
                int indexFrom = m_reps[nbRefs.getValue() *j + i];
#ifdef SOFA_DEV
                if ( distanceType.getValue().getSelectedId()  == DISTANCE_HARMONIC)
                {
                    m_weights[indexFrom][j] = geoDist->harmonicMaxValue.getValue() - distances[indexFrom][j];
                    if ( distances[indexFrom][j] < 0.0) distances[indexFrom][j] = 0.0;
                    if ( distances[indexFrom][j] > geoDist->harmonicMaxValue.getValue()) distances[indexFrom][j] = geoDist->harmonicMaxValue.getValue();
                    if(distances[indexFrom][j]==0 || distances[indexFrom][j]==geoDist->harmonicMaxValue.getValue()) m_dweight[indexFrom][j]=Coord();
                    else m_dweight[indexFrom][j] = - distGradients[indexFrom][j];
                }
                else
                {
#endif
                    m_weights[indexFrom][j] = distances[indexFrom][j];
                    m_dweight[indexFrom][j] = distGradients[indexFrom][j];
#ifdef SOFA_DEV
                }
#endif
            }
        break;
    }
    case WEIGHT_LINEAR:
    {
        vector<int> tmpReps;
        sortReferences( tmpReps);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            for ( unsigned int j=0; j<xfrom.size(); j++ )
            {
                m_weights[j][i] = 0.0;
                m_dweight[j][i] = Coord();
                m_ddweight[j][i].fill(0);
            }
            Vec3d r1r2, r1p;
            r1r2 = xfrom[tmpReps[nbRefs.getValue() *i+1]].getCenter() - xfrom[tmpReps[nbRefs.getValue() *i+0]].getCenter();
            r1p  = xto[i] - xfrom[tmpReps[nbRefs.getValue() *i+0]].getCenter();
            double r1r2NormSquare = r1r2.norm()*r1r2.norm();
            double wi = ( r1r2*r1p ) / ( r1r2NormSquare);

            // Abscisse curviligne
            m_weights[tmpReps[nbRefs.getValue() *i+0]][i] = ( 1 - wi );
            m_weights[tmpReps[nbRefs.getValue() *i+1]][i] = wi;
            m_dweight[tmpReps[nbRefs.getValue() *i+0]][i] = -r1r2 / r1r2NormSquare;
            m_dweight[tmpReps[nbRefs.getValue() *i+1]][i] = r1r2 / r1r2NormSquare;
        }
        break;
    }
    case WEIGHT_INVDIST_SQUARE:
    {
        for ( unsigned int j=0; j<xto.size(); j++ )
        {
            for ( unsigned int i=0; i<nbRefs.getValue(); i++ )
            {
                int indexFrom = m_reps[nbRefs.getValue() *j + i];
                double d2=distances[indexFrom][j]*distances[indexFrom][j];
                double d3=d2*distances[indexFrom][j];
                double d4=d3*distances[indexFrom][j];

                m_ddweight[indexFrom][j].fill(0);
                if (d2)
                {
                    m_weights[indexFrom][j] = 1 / d2;
                    m_dweight[indexFrom][j] = - distGradients[indexFrom][j] / d3* 2.0;
                    m_ddweight[indexFrom][j][0][0]-=2.0/d4;	m_ddweight[indexFrom][j][1][1]-=2.0/d4;	m_ddweight[indexFrom][j][2][2]-=2.0/d4;
                    for(unsigned int k=0; k<3; k++) for(unsigned int m=0; m<3; m++) m_ddweight[indexFrom][j][k][m]+=distGradients[indexFrom][j][k]*distGradients[indexFrom][j][m]*8.0/d2;
                }
                else
                {
                    m_weights[indexFrom][j] = 0xFFF;
                    m_dweight[indexFrom][j] = Coord();
                }

            }
        }

        break;
    }
    case WEIGHT_HERMITE:
    {
        vector<int> tmpReps;
        sortReferences( tmpReps);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            for ( unsigned int j=0; j<xfrom.size(); j++ )
            {
                m_weights[j][i] = 0.0;
                m_dweight[j][i] = Coord();
            }
            Vec3d r1r2, r1p;
            double wi;
            r1r2 = xfrom[tmpReps[nbRefs.getValue() *i+1]].getCenter() - xfrom[tmpReps[nbRefs.getValue() *i+0]].getCenter();
            r1p  = xto[i] - xfrom[tmpReps[nbRefs.getValue() *i+0]].getCenter();
            double r1r2NormSquare = r1r2.norm()*r1r2.norm();
            wi = ( r1r2*r1p ) / r1r2NormSquare;

            // Fonctions d'Hermite
            m_weights[tmpReps[nbRefs.getValue() *i+0]][i] = 1-3*wi*wi+2*wi*wi*wi;
            m_weights[tmpReps[nbRefs.getValue() *i+1]][i] = 3*wi*wi-2*wi*wi*wi;

            r1r2.normalize();
            m_dweight[tmpReps[nbRefs.getValue() *i+0]][i] = -r1r2 * (6*wi-6*wi*wi) / (r1r2NormSquare);
            m_dweight[tmpReps[nbRefs.getValue() *i+1]][i] = r1r2 * (6*wi-6*wi*wi) / (r1r2NormSquare);
        }
        break;
    }
    case WEIGHT_SPLINE:
    {
        if( xfrom.size() < 4 || nbRefs.getValue() < 4)
        {
            serr << "Error ! To use WEIGHT_SPLINE, you must use at least 4 DOFs and set nbRefs to 4.\n WEIGHT_SPLINE requires also the DOFs are ordered along z-axis." << sendl;
            return;
        }
        vector<int> tmpReps;
        sortReferences( tmpReps);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            // Clear all weights and dweights.
            for ( unsigned int j=0; j<xfrom.size(); j++ )
            {
                m_weights[j][i] = 0.0;
                m_dweight[j][i] = Coord();
            }
            // Get the 4 nearest DOFs.
            vector<unsigned int> sortedFrames;
            for( unsigned int j = 0; j < 4; ++j)
                sortedFrames.push_back( tmpReps[nbRefs.getValue() *i+j]);
            std::sort( sortedFrames.begin(), sortedFrames.end());

            if( xto[i][2] < xfrom[sortedFrames[1]].getCenter()[2])
            {
                sortedFrames.clear();
                sortedFrames.push_back( 0);
                sortedFrames.push_back( 0);
                sortedFrames.push_back( 1);
                sortedFrames.push_back( 2);
            }
            else if( xto[i][2] > xfrom[sortedFrames[2]].getCenter()[2])
            {
                sortedFrames.clear();
                sortedFrames.push_back( sortedFrames[1]);
                sortedFrames.push_back( sortedFrames[2]);
                sortedFrames.push_back( sortedFrames[3]);
                sortedFrames.push_back( sortedFrames[3]);
            }

            // Compute u
            Vec3d r1r2 = xfrom[sortedFrames[2]].getCenter() - xfrom[sortedFrames[1]].getCenter();
            Vec3d r1p  = xto[i] - xfrom[sortedFrames[1]].getCenter();
            double r1r2NormSquare = r1r2.norm()*r1r2.norm();
            double u = ( r1r2*r1p ) / r1r2NormSquare;

            // Set weights and dweights.
            m_weights[sortedFrames[0]][i] = 1-3*u*u+2*u*u*u;
            m_weights[sortedFrames[1]][i] = u*u*u - 2*u*u + u;
            m_weights[sortedFrames[2]][i] = 3*u*u-2*u*u*u;
            m_weights[sortedFrames[3]][i] = u*u*u - u*u;

            r1r2.normalize();
            m_dweight[sortedFrames[0]][i] = -r1r2 * (6*u - 6*u*u) / (r1r2NormSquare);
            m_dweight[sortedFrames[1]][i] = r1r2 * (3*u*u - 4*u + 1) / (r1r2NormSquare);
            m_dweight[sortedFrames[2]][i] = r1r2 * (6*u - 6*u*u) / (r1r2NormSquare);
            m_dweight[sortedFrames[3]][i] = r1r2 * (3*u*u - 2*u) / (r1r2NormSquare);
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
    vector<int> * m_reps = repartition.beginEdit();
    m_reps->clear();
    m_reps->insert ( m_reps->begin(), rep.begin(), rep.end() );;
    repartition.endEdit();
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
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_weights = weights.getValue();
    const SVector<SVector<GeoCoord> >& dw = weightGradients.getValue();
    const unsigned int nbRef = nbRefs.getValue();
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
                double coef = m_weights[idxReps][i];
                if ( coef > 0.0 )
                {
                    glColor4d ( coef,coef,0,1 );
                    helper::gl::glVertexT ( xfrom[m_reps[nbRef *i+m] ].getCenter() );
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
            sofa::helper::gl::GlText::draw ( (int)(distances[m_reps[i*nbRef+showFromIndex.getValue()%distances.size()]][i]*scale), xto[i], textScale );
    }

    // Display distance gradients values for each points
    if ( showGradientsValues.getValue())
    {
        char txt[100];
        glColor3f( 0.5, 0.5, 0.5);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            const Vec3& grad = dw[m_reps[i*nbRef+showFromIndex.getValue()%dw.size()]][i];
            sprintf( txt, "( %i, %i, %i)", (int)(grad[0]*scale), (int)(grad[1]*scale), (int)(grad[2]*scale));
            sofa::helper::gl::GlText::draw ( txt, xto[i], textScale );
        }
    }

    // Display weights for each points
    if ( showWeightsValues.getValue())
    {
        glColor3f( 1.0, 1.0, 1.0);
        for ( unsigned int i=0; i<xto.size(); i++ )
            sofa::helper::gl::GlText::draw ( (int)(m_weights[m_reps[i*nbRef+showFromIndex.getValue()%m_weights.size()]][i]*scale), xto[i], textScale );
    }

    // Display weights gradients for each points
    if ( showGradients.getValue())
    {
        glColor3f ( 0.0, 1.0, 0.3 );
        glBegin ( GL_LINES );
        for ( unsigned int j = 0; j < xto.size(); j++ )
        {
            const GeoCoord& gradMap = dw[m_reps[j*nbRef+showFromIndex.getValue()%dw.size()]][j];
            const Coord& point = xto[j];
            glVertex3f ( point[0], point[1], point[2] );
            glVertex3f ( point[0] + gradMap[0] * showGradientsScaleFactor.getValue(), point[1] + gradMap[1] * showGradientsScaleFactor.getValue(), point[2] + gradMap[2] * showGradientsScaleFactor.getValue() );
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
        for ( unsigned int j = 0; j < xto.size(); j++)
        {
            if ( m_weights[m_reps[j*nbRef+showFromIndex.getValue()%m_weights.size()]][j] < minValue && m_weights[m_reps[j*nbRef+showFromIndex.getValue()%m_weights.size()]][j] != 0xFFF) minValue = m_weights[m_reps[j*nbRef+showFromIndex.getValue()%m_weights.size()]][j];
            if ( m_weights[m_reps[j*nbRef+showFromIndex.getValue()%m_weights.size()]][j] > maxValue && m_weights[m_reps[j*nbRef+showFromIndex.getValue()%m_weights.size()]][j] != 0xFFF) maxValue = m_weights[m_reps[j*nbRef+showFromIndex.getValue()%m_weights.size()]][j];
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
                    double color = (m_weights[m_reps[tri[i][j]*nbRef+showFromIndex.getValue()%m_weights.size()]][tri[i][j]] - minValue) / (maxValue - minValue);
                    color = pow(color, showGammaCorrection.getValue());
                    points.push_back(defaulttype::Vector3(xto[tri[i][j]][0],xto[tri[i][j]][1],xto[tri[i][j]][2]));
                    colors.push_back(defaulttype::Vec<4,float>(color, 0.0, 0.0,1.0));
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
                double color = (m_weights[m_reps[i*nbRef+showFromIndex.getValue()%m_weights.size()]][i] - minValue) / (maxValue - minValue);
                color = pow(color, showGammaCorrection.getValue());
                glColor3f( color, 0.0, 0.0);
                glVertex3f( xto[i][0], xto[i][1], xto[i][2]);
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
        this->geoDist->addElt( newX0.getCenter(), beginPointSet, distMax);
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
        this->geoDist->getDistances ( dist, ddist, goals );
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
                dw[i][0] = Coord();
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
                        dw[i][j] = Coord();
                    }
                    else
                    {
                        if( maximizeWeightDist != 0.0)
                        {
                          if( this->distances[i][j] < maximizeWeightDist)
                          {
                            m_weights[i][j] = 0xFFF;
                            dw[i][j] = Coord();
                          }
                          else if( this->distances[i][j] > radius[i])
                          {
                            m_weights[i][j] = 0.0;
                            dw[i][j] = Coord();
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
                            dw[i][j] = Coord();
                          }
                          else
                            dw[i][j] = - this->distGradients[i][j] / (this->distances[i][j]*this->distances[i][j]*this->distances[i][j]) * 2.0;
                        }
                    }
                }
                else
                {
                    m_weights[i][j] = 0xFFF;
                    dw[i][j] = Coord();
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
    if( this->geoDist)
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
void SkinningMapping<TIn, TOut>::getLocalCoord( Coord& result, const typename defaulttype::StdRigidTypes<N, InReal>::Coord& inCoord, const Coord& coord) const
{
    result = inCoord.getOrientation().inverseRotate ( coord - inCoord.getCenter() );
}


#ifdef SOFA_DEV
template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::getLocalCoord( Coord& result, const typename defaulttype::StdAffineTypes<N, InReal>::Coord& inCoord, const Coord& coord) const
{
    Mat33 affineInv;
    affineInv.invert( inCoord.getAffine() );
    result = affineInv * ( coord - inCoord.getCenter() );
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::getLocalCoord( Coord& result, const typename defaulttype::StdQuadraticTypes<N, InReal>::Coord& inCoord, const Coord& coord) const
{
    result = coord - inCoord.getCenter();
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
    L[0][0]=q[3]/2.; L[0][1]=q[2]/2.; L[0][2]=-q[1]/2.; L[0][3]=0;
    L[0][4]=0; L[0][5]=0;
    L[1][0]=-q[2]/2.; L[1][1]=q[3]/2.; L[1][2]=q[0]/2.; L[1][3]=0;
    L[1][4]=0; L[1][5]=0;
    L[2][0]=q[1]/2.; L[2][1]=-q[0]/2.; L[2][2]=q[3]/2.; L[2][3]=0;
    L[2][4]=0; L[2][5]=0;
    L[3][0]=-q[0]/2.; L[3][1]=-q[1]/2.; L[3][2]=-q[2]/2.; L[3][3]=0;
    L[3][4]=0; L[3][5]=0;
    L[4][0]=0; L[4][1]=0; L[4][2]=0; L[4][3]=1; L[4][4]=0; L[4][5]=0;
    L[5][0]=0; L[5][1]=0; L[5][2]=0; L[5][3]=0; L[5][4]=1; L[5][5]=0;
    L[6][0]=0; L[6][1]=0; L[6][2]=0; L[6][3]=0; L[6][4]=0; L[6][5]=1;
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::ComputeQ(Mat37& Q, const Quat& q, const Vec3& p) const
{
    Q[0][0]=2*(q[1]*p[1]+q[2]*p[2]); Q[0][1]=2*(-2*q[1]*p[0]+q[0]*p[1]+q[3]*p[2]);
    Q[0][2]=2*(-2*q[2]*p[0]-q[3]*p[1]+q[0]*p[2]);
    Q[0][3]=2*(-q[2]*p[1]+q[1]*p[2]); Q[0][4]=1; Q[0][5]=0; Q[0][6]=0;
    Q[1][0]=2*(q[1]*p[0]-2*q[0]*p[1]-q[3]*p[2]); Q[1][1]=2*(q[0]*p[0]+q[2]*p[2]);
    Q[1][2]=2*(q[3]*p[0]-2*q[2]*p[1]+q[1]*p[2]); Q[1][3]=2*(q[2]*p[0]-q[0]*p[2]);
    Q[1][4]=0; Q[1][5]=1; Q[1][6]=0;
    Q[2][0]=2*(q[2]*p[0]+q[3]*p[1]-2*q[0]*p[2]);
    Q[2][1]=2*(-q[3]*p[0]+q[2]*p[1]-2*q[1]*p[2]); Q[2][2]=2*(q[0]*p[0]+q[1]*p[1]);
    Q[2][3]=2*(-q[1]*p[0]+q[0]*p[1]); Q[2][4]=0; Q[2][5]=0; Q[2][6]=1;
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
void SkinningMapping<TIn, TOut>:: StrainDeriv_rigid(Mat33 Ma,Mat33 Mb,Mat33 Mc,Mat33 Mw,Vec3 dw,Mat33 At,Mat33 F,Mat67 &B) const
{
    unsigned int k,m;
    Mat33 D; Mat33 FT=F.transposed();
    D=FT*Ma*At; B[0][0]+=2*D[0][0]; B[1][0]+=2*D[1][1]; B[2][0]+=2*D[2][2]; B[3][0]+=D[0][1]+D[1][0]; B[4][0]+=D[1][2]+D[2][1]; B[5][0]+=D[0][2]+D[2][0];
    D=FT*Mb*At; B[0][1]+=2*D[0][0]; B[1][1]+=2*D[1][1]; B[2][1]+=2*D[2][2]; B[3][1]+=D[0][1]+D[1][0]; B[4][1]+=D[1][2]+D[2][1]; B[5][1]+=D[0][2]+D[2][0];
    D=FT*Mc*At; B[0][2]+=2*D[0][0]; B[1][2]+=2*D[1][1]; B[2][2]+=2*D[2][2]; B[3][2]+=D[0][1]+D[1][0]; B[4][2]+=D[1][2]+D[2][1]; B[5][2]+=D[0][2]+D[2][0];
    D=FT*Mw*At; B[0][3]+=2*D[0][0]; B[1][3]+=2*D[1][1]; B[2][3]+=2*D[2][2]; B[3][3]+=D[0][1]+D[1][0]; B[4][3]+=D[1][2]+D[2][1]; B[5][3]+=D[0][2]+D[2][0];
    for(k=0; k<3; k++)  for(m=0; m<3; m++) B[m][k+4]+=dw[m]*F[k][m];
    for(k=0; k<3; k++)  B[3][k+4]+=(dw[0]*F[k][1]+dw[1]*F[k][0])/2.;
    for(k=0; k<3; k++)  B[4][k+4]+=(dw[1]*F[k][2]+dw[2]*F[k][1])/2.;
    for(k=0; k<3; k++)  B[5][k+4]+=(dw[0]*F[k][2]+dw[2]*F[k][0])/2.;
}

//affine
template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::StrainDeriv_affine(Vec3 dw,MatInAtx3 At,Mat33 F,Mat6xIn &B) const
{
    unsigned int k,l,m;
// stretch
    for(k=0; k<3; k++)  for(m=0; m<3; m++) B[m][9+k]+=dw[m]*F[k][m];
    for(k=0; k<3; k++)  for(m=0; m<3; m++) for(l=0; l<3; l++) B[m][3*l+k]+=F[l][m]*At[k][m];
// shear
    for(k=0; k<3; k++)  for(l=0; l<3; l++) B[3][3*l+k]+=0.5*(F[l][0]*At[k][1] + F[l][1]*At[k][0]);
    for(k=0; k<3; k++)  B[3][9+k]+=0.5*(dw[0]*F[k][1] + dw[1]*F[k][0]);
    for(k=0; k<3; k++)  for(l=0; l<3; l++) B[4][3*l+k]+=0.5*(F[l][1]*At[k][2]+F[l][2]*At[k][1]);
    for(k=0; k<3; k++)  B[4][9+k]+=0.5*(dw[1]*F[k][2] + dw[2]*F[k][1]);
    for(k=0; k<3; k++)  for(l=0; l<3; l++) B[5][3*l+k]+=0.5*(F[l][2]*At[k][0]+F[l][0]*At[k][2]);
    for(k=0; k<3; k++)  B[5][9+k]+=0.5*(dw[2]*F[k][0] + dw[0]*F[k][2]);
}

// quadratic
template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::StrainDeriv_quadratic(Vec3 dw,MatInAtx3 At,Mat33 F,Mat6xIn &B) const
{
    unsigned int k,l,m;
// stretch
    for(k=0; k<3; k++)  for(m=0; m<3; m++) B[m][27+k]+=dw[m]*F[k][m];
    for(k=0; k<9; k++)  for(m=0; m<3; m++) for(l=0; l<3; l++) B[m][9*l+k]+=F[l][m]*At[k][m];
// shear
    for(k=0; k<9; k++)  for(l=0; l<3; l++) B[3][9*l+k]+=0.5*(F[l][0]*At[k][1] + F[l][1]*At[k][0]);
    for(k=0; k<3; k++)  B[3][27+k]+=0.5*(dw[0]*F[k][1] + dw[1]*F[k][0]);
    for(k=0; k<9; k++)  for(l=0; l<3; l++) B[4][9*l+k]+=0.5*(F[l][1]*At[k][2]+F[l][2]*At[k][1]);
    for(k=0; k<3; k++)  B[4][27+k]+=0.5*(dw[1]*F[k][2] + dw[2]*F[k][1]);
    for(k=0; k<9; k++)  for(l=0; l<3; l++) B[5][9*l+k]+=0.5*(F[l][2]*At[k][0]+F[l][0]*At[k][2]);
    for(k=0; k<3; k++)  B[5][27+k]+=0.5*(dw[2]*F[k][0] + dw[0]*F[k][2]);
}

template <class TIn, class TOut>
template<class T>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::RigidType, T> >::type SkinningMapping<TIn, TOut>::precomputeMatrices(const RigidType&)
{
    const VecInCoord& xfrom0 = *this->fromModel->getX0();
    const VecCoord& xto0 = *this->toModel->getX0();
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_weights = weights.getValue();
    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());
    VVMat33& m_ddweight = this->weightGradients2;

    // vol and massDensity
    sofa::component::topology::DynamicSparseGridTopologyContainer* hexaContainer;
    this->getContext()->get( hexaContainer);
    double volume = this->voxelVolume.getValue();
    if ( hexaContainer && this->geoDist) volume = this->geoDist->initTargetStep.getValue()*this->geoDist->initTargetStep.getValue()*this->geoDist->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[0]*hexaContainer->voxelSize.getValue()[1]*hexaContainer->voxelSize.getValue()[2];
    const VecCoord& xto = *this->toModel->getX();
    this->vol.resize( xto.size());
    for ( unsigned int i = 0; i < xto.size(); i++) this->vol[i] = volume;
    this->massDensity.resize( xto.size());
    for ( unsigned int i = 0; i < xto.size(); i++) this->massDensity[i] = 1.0;

    // Resize matrices
    this->det.resize(xto.size());
    this->deformationTensors.resize(xto.size());
    this->ddet.resize(xfrom0.size());
    for(unsigned int i = 0; i < xfrom0.size(); ++i)      this->ddet[i].resize(xto.size());
    this->B.resize(xfrom0.size());
    for(unsigned int i = 0; i < xfrom0.size(); ++i)      this->B[i].resize(xto.size());
    this->Atilde.resize(xfrom0.size());
    for (unsigned int i = 0; i < xfrom0.size(); ++i)      this->Atilde[i].resize(xto0.size());
    this->J0.resize ( xfrom0.size() );
    for (unsigned int i = 0; i < xfrom0.size(); ++i)      this->J0[i].resize(xto0.size());
    this->J.resize(xfrom0.size());
    for (unsigned int i = 0; i < xfrom0.size(); ++i)      this->J[i].resize(xto0.size());

    if( useElastons.getValue())
    {
        this->integ_Elaston.resize( xto.size());
        for ( unsigned int i = 0; i < xto.size(); i++) // to update
        {
            double lx,ly,lz;
            if ( hexaContainer && this->geoDist)
            {
                lx=this->geoDist->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[0];
                ly=this->geoDist->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[1];
                lz=this->geoDist->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[2];
            }
            else lx=ly=lz=pow(this->voxelVolume.getValue(),1/3.);
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

        this->deformationTensorsElaston.resize(xto.size());
        this->Stilde_x.resize(xfrom0.size());
        this->Stilde_y.resize(xfrom0.size());
        this->Stilde_z.resize(xfrom0.size());
        for (unsigned int i = 0; i < xfrom0.size(); ++i)
        {
            this->Stilde_x[i].resize(xto0.size());
            this->Stilde_y[i].resize(xto0.size());
            this->Stilde_z[i].resize(xto0.size());
        }
        this->B_Elaston.resize(xfrom0.size());
        for(unsigned int i = 0; i < xfrom0.size(); ++i)      this->B_Elaston[i].resize(xto.size());
    }

    // Precompute matrices
    for ( unsigned int i=0 ; i<xto0.size(); i++ )
    {
        for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
        {
            const int& idx=nbRefs.getValue() *i+m;
            const int& idxReps=m_reps[idx];

            const InCoord& xi0 = xfrom0[idxReps];
            Mat33 R0;	 QtoR( R0, xi0.getOrientation());
            Mat33 R0Inv;  R0Inv.invert (R0);

            for(int k=0; k<3; k++) for(int l=0; l<3; l++) (this->Atilde[idxReps][i])[k][l] = (initPos[idx])[k] * (m_dweight[idxReps][i])[l]  +  m_weights[idxReps][i] * (R0Inv)[k][l];

            if( useElastons.getValue())
            {
                for(int k=0; k<3; k++) for(int l=0; l<3; l++) (this->Stilde_x[idxReps][i])[k][l] = (initPos[idx])[k] * (m_ddweight[idxReps][i])[l][0] + (R0Inv)[k][0] * (m_dweight[idxReps][i])[l]  +  m_dweight[idxReps][i][0] * (R0Inv)[k][l];
                for(int k=0; k<3; k++) for(int l=0; l<3; l++) (this->Stilde_y[idxReps][i])[k][l] = (initPos[idx])[k] * (m_ddweight[idxReps][i])[l][1] + (R0Inv)[k][1] * (m_dweight[idxReps][i])[l]  +  m_dweight[idxReps][i][1] * (R0Inv)[k][l];
                for(int k=0; k<3; k++) for(int l=0; l<3; l++) (this->Stilde_z[idxReps][i])[k][l] = (initPos[idx])[k] * (m_ddweight[idxReps][i])[l][2] + (R0Inv)[k][2] * (m_dweight[idxReps][i])[l]  +  m_dweight[idxReps][i][2] * (R0Inv)[k][l];
            }

            Mat76 L; ComputeL(L, xi0.getOrientation());
            Mat37 Q; ComputeQ(Q, xi0.getOrientation(), initPos[idx]);

            this->J[idxReps][i] = (InReal)m_weights[idxReps][i] * Q * L;
            this->J0[idxReps][i] = this->J[idxReps][i];
        }
    }
}


template <class TIn, class TOut>
template<class T>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::AffineType, T> >::type SkinningMapping<TIn, TOut>::precomputeMatrices(const AffineType&)
{
    const VecInCoord& xfrom0 = *this->fromModel->getX0();
    const VecCoord& xto0 = *this->toModel->getX0();
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_weights = weights.getValue();
    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());
    VVMat33& m_ddweight = this->weightGradients2;

    // vol and massDensity
    sofa::component::topology::DynamicSparseGridTopologyContainer* hexaContainer;
    this->getContext()->get( hexaContainer);
    double volume = this->voxelVolume.getValue();
    if ( hexaContainer && this->geoDist) volume = this->geoDist->initTargetStep.getValue()*this->geoDist->initTargetStep.getValue()*this->geoDist->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[0]*hexaContainer->voxelSize.getValue()[1]*hexaContainer->voxelSize.getValue()[2];
    const VecCoord& xto = *this->toModel->getX();
    this->vol.resize( xto.size());
    for ( unsigned int i = 0; i < xto.size(); i++) this->vol[i] = volume;
    this->massDensity.resize( xto.size());
    for ( unsigned int i = 0; i < xto.size(); i++) this->massDensity[i] = 1.0;

    // Resize matrices
    this->det.resize(xto.size());
    this->deformationTensors.resize(xto.size());
    this->ddet.resize(xfrom0.size());
    for(unsigned int i = 0; i < xfrom0.size(); ++i)      this->ddet[i].resize(xto.size());
    this->B.resize(xfrom0.size());
    for(unsigned int i = 0; i < xfrom0.size(); ++i)      this->B[i].resize(xto.size());
    this->Atilde.resize(xfrom0.size());
    for (unsigned int i = 0; i < xfrom0.size(); ++i)      this->Atilde[i].resize(xto0.size());
    this->J.resize(xfrom0.size());
    for (unsigned int i = 0; i < xfrom0.size(); ++i)      this->J[i].resize(xto0.size());

    if( useElastons.getValue())
    {
        this->integ_Elaston.resize( xto.size());
        for ( unsigned int i = 0; i < xto.size(); i++) // to update
        {
            double lx,ly,lz;
            if ( hexaContainer && this->geoDist)
            {
                lx=this->geoDist->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[0];
                ly=this->geoDist->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[1];
                lz=this->geoDist->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[2];
            }
            else lx=ly=lz=pow(this->voxelVolume.getValue(),1/3.);
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

        this->deformationTensorsElaston.resize(xto.size());
        this->Stilde_x.resize(xfrom0.size());
        this->Stilde_y.resize(xfrom0.size());
        this->Stilde_z.resize(xfrom0.size());
        for (unsigned int i = 0; i < xfrom0.size(); ++i)
        {
            this->Stilde_x[i].resize(xto0.size());
            this->Stilde_y[i].resize(xto0.size());
            this->Stilde_z[i].resize(xto0.size());
        }
        this->B_Elaston.resize(xfrom0.size());
        for(unsigned int i = 0; i < xfrom0.size(); ++i)      this->B_Elaston[i].resize(xto.size());
    }

    // Precompute matrices
    for ( unsigned int i=0 ; i<xto0.size(); i++ )
    {
        for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
        {
            const int& idx = nbRefs.getValue() *i+m;
            const int& idxReps = m_reps[idx];

            const InCoord& xi0 = xfrom0[idxReps];
            const Mat33& A0 = xi0.getAffine();
            Mat33 A0Inv; A0Inv.invert (A0);

            for(int k=0; k<3; k++) for(int l=0; l<3; l++) (this->Atilde[idxReps][i])[k][l] = (initPos[idx])[k] * (m_dweight[idxReps][i])[l]  +  m_weights[idxReps][i] * (A0Inv)[k][l];

            if( useElastons.getValue())
            {
                for(int k=0; k<3; k++) for(int l=0; l<3; l++) (this->Stilde_x[idxReps][i])[k][l] = (initPos[idx])[k] * (m_ddweight[idxReps][i])[l][0] + (A0Inv)[k][0] * (m_dweight[idxReps][i])[l]  +  m_dweight[idxReps][i][0] * (A0Inv)[k][l];
                for(int k=0; k<3; k++) for(int l=0; l<3; l++) (this->Stilde_y[idxReps][i])[k][l] = (initPos[idx])[k] * (m_ddweight[idxReps][i])[l][1] + (A0Inv)[k][1] * (m_dweight[idxReps][i])[l]  +  m_dweight[idxReps][i][1] * (A0Inv)[k][l];
                for(int k=0; k<3; k++) for(int l=0; l<3; l++) (this->Stilde_z[idxReps][i])[k][l] = (initPos[idx])[k] * (m_ddweight[idxReps][i])[l][2] + (A0Inv)[k][2] * (m_dweight[idxReps][i])[l]  +  m_dweight[idxReps][i][2] * (A0Inv)[k][l];
            }

            Mat3xIn& Ji = this->J[idxReps][i];
            Ji.fill(0);
            double val;
            for(int k=0; k<3; k++)
            {
                val = m_weights[idxReps][i] * initPos[idx][k];
                Ji[0][k]=val;
                Ji[1][k+3]=val;
                Ji[2][k+6]=val;
                Ji[k][k+9]=m_weights[idxReps][i];
            }
        }
    }
}


template <class TIn, class TOut>
template<class T>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::QuadraticType, T> >::type SkinningMapping<TIn, TOut>::precomputeMatrices(const QuadraticType&)
{
    const VecInCoord& xfrom0 = *this->fromModel->getX0();
    const VecCoord& xto0 = *this->toModel->getX0();
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_weights = weights.getValue();
    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());
    VVMat33& m_ddweight = this->weightGradients2;

    // vol and massDensity
    sofa::component::topology::DynamicSparseGridTopologyContainer* hexaContainer;
    this->getContext()->get( hexaContainer);
    double volume = this->voxelVolume.getValue();
    if ( hexaContainer && this->geoDist) volume = this->geoDist->initTargetStep.getValue()*this->geoDist->initTargetStep.getValue()*this->geoDist->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[0]*hexaContainer->voxelSize.getValue()[1]*hexaContainer->voxelSize.getValue()[2];
    const VecCoord& xto = *this->toModel->getX();
    this->vol.resize( xto.size());
    for ( unsigned int i = 0; i < xto.size(); i++) this->vol[i] = volume;
    this->massDensity.resize( xto.size());
    for ( unsigned int i = 0; i < xto.size(); i++) this->massDensity[i] = 1.0;

    // Resize matrices
    this->det.resize(xto.size());
    this->deformationTensors.resize(xto.size());
    this->ddet.resize(xfrom0.size());
    for(unsigned int i = 0; i < xfrom0.size(); ++i)      this->ddet[i].resize(xto.size());
    this->B.resize(xfrom0.size());
    for(unsigned int i = 0; i < xfrom0.size(); ++i)      this->B[i].resize(xto.size());
    this->Atilde.resize(xfrom0.size());
    for (unsigned int i = 0; i < xfrom0.size(); ++i)      this->Atilde[i].resize(xto0.size());
    this->J.resize(xfrom0.size());
    for (unsigned int i = 0; i < xfrom0.size(); ++i)      this->J[i].resize(xto0.size());

    if( useElastons.getValue())
    {
        this->integ_Elaston.resize( xto.size());
        for ( unsigned int i = 0; i < xto.size(); i++) // to update
        {
            double lx,ly,lz;
            if ( hexaContainer && this->geoDist)
            {
                lx=this->geoDist->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[0];
                ly=this->geoDist->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[1];
                lz=this->geoDist->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[2];
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

        this->deformationTensorsElaston.resize(xto.size());
        this->Stilde_x.resize(xfrom0.size());
        this->Stilde_y.resize(xfrom0.size());
        this->Stilde_z.resize(xfrom0.size());
        for (unsigned int i = 0; i < xfrom0.size(); ++i)
        {
            this->Stilde_x[i].resize(xto0.size());
            this->Stilde_y[i].resize(xto0.size());
            this->Stilde_z[i].resize(xto0.size());
        }
        this->B_Elaston.resize(xfrom0.size());
        for(unsigned int i = 0; i < xfrom0.size(); ++i)      this->B_Elaston[i].resize(xto.size());
    }
    // Precompute matrices ( suppose that A0=I )
    for ( unsigned int i=0 ; i<xto0.size(); i++ )
    {
        for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
        {
            const int& idx = nbRefs.getValue() *i+m;
            const int& idxReps = m_reps[idx];

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
            for (int k=0; k<9; k++) for (int l=0; l<3; l++) (this->Atilde[idxReps][i])[k][l] = p2[k] * m_dweight[idxReps][i][l] + m_weights[idxReps][i] * (A0Inv)[k][l];

            if( useElastons.getValue())
            {
                for(int k=0; k<9; k++) for(int l=0; l<3; l++) (this->Stilde_x[idxReps][i])[k][l] = p2[k] * (m_ddweight[idxReps][i])[l][0] + (A0Inv)[k][0] * (m_dweight[idxReps][i])[l]  +  m_dweight[idxReps][i][0] * (A0Inv)[k][l];
                for(int k=0; k<9; k++) for(int l=0; l<3; l++) (this->Stilde_y[idxReps][i])[k][l] = p2[k] * (m_ddweight[idxReps][i])[l][1] + (A0Inv)[k][1] * (m_dweight[idxReps][i])[l]  +  m_dweight[idxReps][i][1] * (A0Inv)[k][l];
                for(int k=0; k<9; k++) for(int l=0; l<3; l++) (this->Stilde_z[idxReps][i])[k][l] = p2[k] * (m_ddweight[idxReps][i])[l][2] + (A0Inv)[k][2] * (m_dweight[idxReps][i])[l]  +  m_dweight[idxReps][i][2] * (A0Inv)[k][l];
            }


            Mat3xIn& Ji = this->J[idxReps][i];
            Ji.fill(0);
            double val;
            for(int k=0; k<9; k++)
            {
                val = m_weights[idxReps][i] * p2[k];
                Ji[0][k] = val;
                Ji[1][k+9] = val;
                Ji[2][k+18] = val;
            }
            for(int k=0; k<3; k++)
                Ji[k][k+27] = m_weights[idxReps][i];
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
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_weights = weights.getValue();

    rotatedPoints.resize ( initPos.size() );
    out.resize ( initPos.size() / nbRefs.getValue() );
    for ( unsigned int i=0 ; i<out.size(); i++ )
    {
        out[i] = Coord();
        for ( unsigned int j = 0; j < nbRefs.getValue(); ++j)
        {
            const int& idx=nbRefs.getValue() *i+j;
            const int& idxReps=m_reps[idx];

            // Save rotated points for applyJ/JT
            rotatedPoints[idx] = in[idxReps].getOrientation().rotate ( initPos[idx] );

            // And add each reference frames contributions to the new position out[i]
            out[i] += ( in[idxReps ].getCenter() + rotatedPoints[idx] ) * m_weights[idxReps][i];
        }

#ifdef SOFA_DEV
        if ( !(this->isMechanical() || this->computeAllMatrices.getValue())) continue;

        // update J
        Mat37 Q;
        VMat76 L;
        L.resize(nbRefs.getValue());
        for(unsigned int j = 0; j < nbRefs.getValue(); j++)
        {
            const int& idx = nbRefs.getValue() * i + j;
            const int& idxReps = m_reps[idx];

            ComputeQ( Q, in[idxReps ].getOrientation(), initPos[idx]);
            ComputeL( L[idxReps], in[idxReps ].getOrientation());
            this->J[idxReps][i] = (InReal)m_weights[idxReps][i] * Q * L[idxReps];
        }

        // Physical computations
        if ( !this->computeAllMatrices.getValue()) continue;

        const SVector<SVector<GeoCoord> >& dw = this->weightGradients.getValue();
        VVMat33& ddw = this->weightGradients2;

        Mat33 F, Finv, E,A,S_x,S_y,S_z;
        F.fill ( 0 ); S_x.fill(0); S_y.fill(0); S_z.fill(0);
        E.fill ( 0 );
        for ( unsigned int j = 0 ; j < nbRefs.getValue(); ++j )
        {
            const int& idx = nbRefs.getValue() * i + j;
            const int& idxReps = m_reps[idx];

            in[idxReps ].getOrientation().toMatrix(A);
            for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) F[k][l]+=in[idxReps ].getCenter()[k]*dw[idxReps][i][l];
            F += A * this->Atilde[idxReps][i];

            if( useElastons.getValue())
            {
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_x[k][l]+=in[idxReps ].getCenter()[k]*ddw[idxReps][i][l][0];
                S_x += A * this->Stilde_x[idxReps][i];
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_y[k][l]+=in[idxReps ].getCenter()[k]*ddw[idxReps][i][l][1];
                S_y += A * this->Stilde_y[idxReps][i];
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_z[k][l]+=in[idxReps ].getCenter()[k]*ddw[idxReps][i][l][2];
                S_z += A * this->Stilde_z[idxReps][i];
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
        for ( unsigned int j = 0 ; j < nbRefs.getValue(); ++j )
        {
            const int& idx = nbRefs.getValue() * i + j;
            const int& idxReps = m_reps[idx];

            Mat6xIn& B = this->B[idxReps][i];
            const Mat33& At = this->Atilde[idxReps][i];
            const Quat& rot = in[idxReps ].getOrientation();
            const Vec3& dWeight = dw[idxReps][i];

            Mat33 Ma, Mb, Mc, Mw;
            ComputeMa( Ma, rot);
            ComputeMb( Mb, rot);
            ComputeMc( Mc, rot);
            ComputeMw( Mw, rot);

            Mat67 dE;
            dE.fill(0); StrainDeriv_rigid(Ma,Mb,Mc,Mw,dWeight,At,F,dE);
            B=dE*L[idxReps];

            if(useElastons.getValue())
            {
                MatInx610& Be = this->B_Elaston[idxReps][i];
                const Mat33& Stx = this->Stilde_x[idxReps][i];
                const Mat33& Sty = this->Stilde_y[idxReps][i];
                const Mat33& Stz = this->Stilde_z[idxReps][i];
                const Vec3 ddx (ddw[idxReps][i][0][0],ddw[idxReps][i][1][0],ddw[idxReps][i][2][0]);
                const Vec3 ddy (ddw[idxReps][i][0][1],ddw[idxReps][i][1][1],ddw[idxReps][i][2][1]);
                const Vec3 ddz (ddw[idxReps][i][0][2],ddw[idxReps][i][1][2],ddw[idxReps][i][2][2]);
                unsigned int k,m;

                Mat66 S;
                for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][9]=B[k][m];
                dE.fill(0); StrainDeriv_rigid(Ma,Mb,Mc,Mw,dWeight,At,S_x,dE); StrainDeriv_rigid(Ma,Mb,Mc,Mw,ddx,Stx,F,dE); S=dE*L[idxReps];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][0]=S[k][m];
                dE.fill(0); StrainDeriv_rigid(Ma,Mb,Mc,Mw,dWeight,At,S_y,dE); StrainDeriv_rigid(Ma,Mb,Mc,Mw,ddy,Sty,F,dE); S=dE*L[idxReps];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][1]=S[k][m];
                dE.fill(0); StrainDeriv_rigid(Ma,Mb,Mc,Mw,dWeight,At,S_z,dE); StrainDeriv_rigid(Ma,Mb,Mc,Mw,ddz,Stz,F,dE); S=dE*L[idxReps];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][2]=S[k][m];

                // extended elastons
                {
                    dE.fill(0); StrainDeriv_rigid(Ma,Mb,Mc,Mw,ddx,Stx,S_x,dE); S=dE*L[idxReps];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][3]=S[k][m];
                    dE.fill(0); StrainDeriv_rigid(Ma,Mb,Mc,Mw,ddy,Sty,S_y,dE); S=dE*L[idxReps];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][4]=S[k][m];
                    dE.fill(0); StrainDeriv_rigid(Ma,Mb,Mc,Mw,ddz,Stz,S_z,dE); S=dE*L[idxReps];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][5]=S[k][m];
                    dE.fill(0); StrainDeriv_rigid(Ma,Mb,Mc,Mw,ddx,Stx,S_y,dE); StrainDeriv_rigid(Ma,Mb,Mc,Mw,ddy,Sty,S_x,dE); S=dE*L[idxReps];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][6]=S[k][m];
                    dE.fill(0); StrainDeriv_rigid(Ma,Mb,Mc,Mw,ddy,Sty,S_z,dE); StrainDeriv_rigid(Ma,Mb,Mc,Mw,ddz,Stz,S_y,dE); S=dE*L[idxReps];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][7]=S[k][m];
                    dE.fill(0); StrainDeriv_rigid(Ma,Mb,Mc,Mw,ddz,Stz,S_x,dE); StrainDeriv_rigid(Ma,Mb,Mc,Mw,ddx,Stx,S_z,dE); S=dE*L[idxReps];	for(k=0; k<6; k++) for(m=0; m<6; m++) Be[m][k][8]=S[k][m];
                }

            }
            // Compute ddet
            // if(computevolpres)
            {
                invertMatrix ( Finv, F );
                Vec6 &ddet = this->ddet[idxReps][i];
                ddet.fill(0);
                Vec7 u7; u7.fill(0);

                Ma=Ma*At;			Mb=Mb*At;			Mc=Mc*At;			Mw=Mw*At;
                for(unsigned int k=0; k<3; k++)
                {
                    for(unsigned int m=0; m<3; m++) { u7[0]+=2*Finv[k][m]*Ma[m][k]; u7[1]+=2*Finv[k][m]*Mb[m][k]; u7[2]+=2*Finv[k][m]*Mc[m][k]; u7[3]+=2*Finv[k][m]*Mw[m][k]; }
                    u7[4]+=Finv[k][0]*dWeight[k]; u7[5]+=Finv[k][1]*dWeight[k]; u7[6]+=Finv[k][2]*dWeight[k];
                }
                for(unsigned int k=0; k<6; k++) for(unsigned int m=0; m<7; m++) ddet[k]+=u7[m]*L[idxReps][m][k];
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
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::AffineType::Coord, TCoord> >::type
SkinningMapping<TIn, TOut>::_apply( typename Out::VecCoord& out, const sofa::helper::vector<typename AffineType::Coord>& in)
{
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_weights = weights.getValue();

    rotatedPoints.resize ( initPos.size() );
    out.resize ( initPos.size() / nbRefs.getValue() );

    // Resize matrices  // pourquoi ??
    if ( this->computeAllMatrices.getValue())
    {
        this->det.resize(out.size());
        this->deformationTensors.resize(out.size());
        this->B.resize(in.size());
        for(unsigned int i = 0; i < in.size(); ++i)
            this->B[i].resize(out.size());
    }

    for ( unsigned int i = 0 ; i < out.size(); i++ )
    {
        // Point transformation (apply)
        out[i] = Coord();

        for ( unsigned int j = 0 ; j < nbRefs.getValue(); ++j )
        {
            const int& idx = nbRefs.getValue() * i + j;
            const int& idxReps = m_reps[idx];

            // Save rotated points for applyJ/JT
            rotatedPoints[idx] = in[idxReps].getAffine() * initPos[idx];

            // And add each reference frames contributions to the new position out[i]
            out[i] += ( in[idxReps ].getCenter() + rotatedPoints[idx] ) * m_weights[idxReps][i];
        }

        // Physical computations
        if ( !this->computeAllMatrices.getValue()) continue;

        const SVector<SVector<GeoCoord> >& dw = this->weightGradients.getValue();
        VVMat33& ddw = this->weightGradients2;

        Mat33 F, Finv, E,A,S_x,S_y,S_z;
        F.fill ( 0 ); S_x.fill(0); S_y.fill(0); S_z.fill(0);
        E.fill ( 0 );

        for ( unsigned int j = 0 ; j < nbRefs.getValue(); ++j )
        {
            const int& idx = nbRefs.getValue() * i + j;
            const int& idxReps = m_reps[idx];

            for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) F[k][l]+=in[idxReps ].getCenter()[k]*dw[idxReps][i][l];
            F += in[idxReps ].getAffine() * this->Atilde[idxReps][i];

            if( useElastons.getValue())
            {
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_x[k][l]+=in[idxReps ].getCenter()[k]*ddw[idxReps][i][l][0];
                S_x += in[idxReps ].getAffine() * this->Stilde_x[idxReps][i];
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_y[k][l]+=in[idxReps ].getCenter()[k]*ddw[idxReps][i][l][1];
                S_y += in[idxReps ].getAffine() * this->Stilde_y[idxReps][i];
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_z[k][l]+=in[idxReps ].getCenter()[k]*ddw[idxReps][i][l][2];
                S_z += in[idxReps ].getAffine() * this->Stilde_z[idxReps][i];
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
        for ( unsigned int j = 0 ; j < nbRefs.getValue(); ++j )
        {
            const int& idx = nbRefs.getValue() * i + j;
            const int& idxReps = m_reps[idx];

            Mat6xIn& B = this->B[idxReps][i];
            const Vec3& dWeight = dw[idxReps][i];
            const Mat33& At = this->Atilde[idxReps][i];
            B.fill(0); StrainDeriv_affine(dWeight,At,F,B);

            if(useElastons.getValue())
            {
                MatInx610& Be = this->B_Elaston[idxReps][i];
                const MatInAtx3& Stx = this->Stilde_x[idxReps][i];
                const MatInAtx3& Sty = this->Stilde_y[idxReps][i];
                const MatInAtx3& Stz = this->Stilde_z[idxReps][i];
                const Vec3 ddx (ddw[idxReps][i][0][0],ddw[idxReps][i][1][0],ddw[idxReps][i][2][0]);
                const Vec3 ddy (ddw[idxReps][i][0][1],ddw[idxReps][i][1][1],ddw[idxReps][i][2][1]);
                const Vec3 ddz (ddw[idxReps][i][0][2],ddw[idxReps][i][1][2],ddw[idxReps][i][2][2]);
                unsigned int k,m;

                Mat6xIn S;
                for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][9]=B[k][m];
                S.fill(0); StrainDeriv_affine(dWeight,At,S_x,S); StrainDeriv_affine(ddx,Stx,F,S); for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][0]=S[k][m];
                S.fill(0); StrainDeriv_affine(dWeight,At,S_y,S); StrainDeriv_affine(ddy,Sty,F,S); for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][1]=S[k][m];
                S.fill(0); StrainDeriv_affine(dWeight,At,S_z,S); StrainDeriv_affine(ddz,Stz,F,S); for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][2]=S[k][m];

                // extended elastons
                {
                    S.fill(0); StrainDeriv_affine(ddx,Stx,S_x,S);	for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][3]=S[k][m];
                    S.fill(0); StrainDeriv_affine(ddy,Sty,S_y,S);	for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][4]=S[k][m];
                    S.fill(0); StrainDeriv_affine(ddz,Stz,S_z,S);	for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][5]=S[k][m];
                    S.fill(0); StrainDeriv_affine(ddx,Stx,S_y,S); StrainDeriv_affine(ddy,Sty,S_x,S);	for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][6]=S[k][m];
                    S.fill(0); StrainDeriv_affine(ddy,Sty,S_z,S); StrainDeriv_affine(ddz,Stz,S_y,S); 	for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][7]=S[k][m];
                    S.fill(0); StrainDeriv_affine(ddz,Stz,S_x,S); StrainDeriv_affine(ddx,Stx,S_z,S); 	for(k=0; k<6; k++) for(m=0; m<12; m++) Be[m][k][8]=S[k][m];
                }
            }

            // Compute ddet
            // if(computevolpres)
            {
                invertMatrix ( Finv, F );
                VecIn &ddet = this->ddet[idxReps][i];
                ddet.fill(0);
                for (unsigned int k = 0; k < 3; k++ ) for (unsigned int m = 0; m < 3; m++ ) for (unsigned int l = 0; l < 3; l++ ) ddet[m+3*k] += At[m][l] * Finv[l][k];
                for (unsigned int k = 0; k < 3; k++ ) for (unsigned int l = 0; l < 3; l++ ) ddet[9+k] += dWeight [l] * Finv[l][k];
                for(unsigned int k=0; k<12; k++) ddet[k]=this->det[i] * ddet[k];
            }
        }
    }
}


// Apply for Quadratic types
template <class TIn, class TOut>
template<class TCoord>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::QuadraticType::Coord, TCoord> >::type
SkinningMapping<TIn, TOut>::_apply( typename Out::VecCoord& out, const sofa::helper::vector<typename QuadraticType::Coord>& in)
{
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_weights = weights.getValue();

    rotatedPoints.resize ( initPos.size() );
    out.resize ( initPos.size() / nbRefs.getValue() );

    // Resize matrices  // pourquoi ??
    if ( this->computeAllMatrices.getValue())
    {
        this->det.resize(out.size());
        this->deformationTensors.resize(out.size());
        this->B.resize(in.size());
        for(unsigned int i = 0; i < in.size(); ++i)
            this->B[i].resize(out.size());
    }

    for ( unsigned int i = 0 ; i < out.size(); i++ )
    {
        // Point transformation (apply)
        out[i] = Coord();

        for ( unsigned int j = 0 ; j < nbRefs.getValue(); ++j )
        {
            const int& idx = nbRefs.getValue() * i + j;
            const int& idxReps = m_reps[idx];

            const Vec3& p0 = initPos[idx];
            Vec9 p2 = Vec9( p0[0], p0[1], p0[2], p0[0]*p0[0], p0[1]*p0[1], p0[2]*p0[2], p0[0]*p0[1], p0[1]*p0[2], p0[0]*p0[2]);

            // Save rotated points for applyJ/JT
            rotatedPoints[idx] = in[idxReps].getQuadratic() * p2;

            // And add each reference frames contributions to the new position out[i]
            out[i] += ( in[idxReps ].getCenter() + rotatedPoints[idx] ) * m_weights[idxReps][i];
        }

        // Physical computations
        if ( !this->computeAllMatrices.getValue()) continue;

        const SVector<SVector<GeoCoord> >& dw = this->weightGradients.getValue();
        VVMat33& ddw = this->weightGradients2;

        Mat33 F, Finv, E,A,S_x,S_y,S_z;
        F.fill ( 0 ); S_x.fill(0); S_y.fill(0); S_z.fill(0);
        E.fill ( 0 );
        for ( unsigned int j = 0 ; j < nbRefs.getValue(); ++j )
        {
            const int& idx = nbRefs.getValue() * i + j;
            const int& idxReps = m_reps[idx];

            for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) F[k][l]+=in[idxReps ].getCenter()[k]*dw[idxReps][i][l];
            F += in[idxReps ].getQuadratic() * this->Atilde[idxReps][i];

            if( useElastons.getValue())
            {
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_x[k][l]+=in[idxReps ].getCenter()[k]*ddw[idxReps][i][l][0];
                S_x += in[idxReps ].getQuadratic() * this->Stilde_x[idxReps][i];
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_y[k][l]+=in[idxReps ].getCenter()[k]*ddw[idxReps][i][l][1];
                S_y += in[idxReps ].getQuadratic() * this->Stilde_y[idxReps][i];
                for (unsigned int k=0; k<3; k++) for (unsigned int l=0; l<3; l++) S_z[k][l]+=in[idxReps ].getCenter()[k]*ddw[idxReps][i][l][2];
                S_z += in[idxReps ].getQuadratic() * this->Stilde_z[idxReps][i];
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
        for ( unsigned int j = 0 ; j < nbRefs.getValue(); ++j )
        {
            const int& idx = nbRefs.getValue() * i + j;
            const int& idxReps = m_reps[idx];

            Mat6xIn& B = this->B[idxReps][i];
            const Vec3& dWeight = dw[idxReps][i];
            const MatInAtx3& At = this->Atilde[idxReps][i];
            B.fill(0); StrainDeriv_quadratic(dWeight,At,F,B);


            if(useElastons.getValue())
            {
                MatInx610& Be = this->B_Elaston[idxReps][i];
                const MatInAtx3& Stx = this->Stilde_x[idxReps][i];
                const MatInAtx3& Sty = this->Stilde_y[idxReps][i];
                const MatInAtx3& Stz = this->Stilde_z[idxReps][i];
                const Vec3 ddx (ddw[idxReps][i][0][0],ddw[idxReps][i][1][0],ddw[idxReps][i][2][0]);
                const Vec3 ddy (ddw[idxReps][i][0][1],ddw[idxReps][i][1][1],ddw[idxReps][i][2][1]);
                const Vec3 ddz (ddw[idxReps][i][0][2],ddw[idxReps][i][1][2],ddw[idxReps][i][2][2]);
                unsigned int k,m;
                Mat6xIn S;

                for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][9]=B[k][m];
                S.fill(0); StrainDeriv_quadratic(dWeight,At,S_x,S); StrainDeriv_quadratic(ddx,Stx,F,S); for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][0]=S[k][m];
                S.fill(0); StrainDeriv_quadratic(dWeight,At,S_y,S); StrainDeriv_quadratic(ddy,Sty,F,S); for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][1]=S[k][m];
                S.fill(0); StrainDeriv_quadratic(dWeight,At,S_z,S); StrainDeriv_quadratic(ddz,Stz,F,S); for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][2]=S[k][m];

                // extended elastons
                {
                    S.fill(0); StrainDeriv_quadratic(ddx,Stx,S_x,S);	for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][3]=S[k][m];
                    S.fill(0); StrainDeriv_quadratic(ddy,Sty,S_y,S);	for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][4]=S[k][m];
                    S.fill(0); StrainDeriv_quadratic(ddz,Stz,S_z,S);	for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][5]=S[k][m];
                    S.fill(0); StrainDeriv_quadratic(ddx,Stx,S_y,S); StrainDeriv_quadratic(ddy,Sty,S_x,S);	for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][6]=S[k][m];
                    S.fill(0); StrainDeriv_quadratic(ddy,Sty,S_z,S); StrainDeriv_quadratic(ddz,Stz,S_y,S);	for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][7]=S[k][m];
                    S.fill(0); StrainDeriv_quadratic(ddz,Stz,S_x,S); StrainDeriv_quadratic(ddx,Stx,S_z,S);	for(k=0; k<6; k++) for(m=0; m<30; m++) Be[m][k][8]=S[k][m];
                }
            }

            // Compute ddet
            // if(computevolpres)
            {
                invertMatrix ( Finv, F );
                VecIn &ddet = this->ddet[idxReps][i];
                ddet.fill(0);
                for (unsigned int k = 0; k < 3; k++ ) for (unsigned int m = 0; m < 9; m++ ) for (unsigned int l = 0; l < 3; l++ ) ddet[m+9*k] += At[m][l] * Finv[l][k];
                for (unsigned int k = 0; k < 3; k++ ) for (unsigned int l = 0; l < 3; l++ ) ddet[27+k] += dWeight [l] * Finv[l][k];
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
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_weights = weights.getValue();
    const VecCoord& xto = *this->toModel->getX();
    out.resize ( xto.size() );
    Deriv v,omega;

    if ( ! ( maskTo->isInUse() ) )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            out[i] = Deriv();
            for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
            {
                const int idx=nbRefs.getValue() *i+m;
                const int idxReps=m_reps[idx];

                v = in[idxReps].getVCenter();
                omega = in[idxReps].getVOrientation();
                out[i] += ( v - cross ( rotatedPoints[idx],omega ) ) * m_weights[idxReps][i];
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
            for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
            {
                const int idx=nbRefs.getValue() *i+m;
                const int idxReps=m_reps[idx];

                v = in[idxReps].getVCenter();
                omega = in[idxReps].getVOrientation();
                out[i] += ( v - cross ( rotatedPoints[idx],omega ) ) * m_weights[idxReps][i];
            }
        }
    }
}

#ifdef SOFA_DEV

template <class TIn, class TOut>
template<class TDeriv>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::AffineType::Deriv, TDeriv> >::type SkinningMapping<TIn, TOut>::_applyJ( typename Out::VecDeriv& out, const sofa::helper::vector<typename AffineType::Deriv>& in)
{
    const VecCoord& xto = *this->toModel->getX();
    out.resize ( xto.size() );
    Deriv v;
    typename In::Deriv::Affine omega;

    if ( ! ( maskTo->isInUse() ) )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            out[i] = Deriv();
            for ( unsigned int j=0 ; j<in.size(); j++ )
            {
                VecIn speed;
                for (unsigned int k = 0; k < InDOFs; ++k)
                    speed[k]  = in[j][k];

                Vec3 f = ( this->J[j][i] * speed );

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
            for ( unsigned int j=0 ; j<in.size(); j++ )
            {
                VecIn speed;
                for (unsigned int k = 0; k < InDOFs; ++k)
                    speed[k]  = in[j][k];

                Vec3 f = ( this->J[j][i] * speed );

                out[i] += Deriv ( f[0], f[1], f[2] );
            }
        }
    }
}


template <class TIn, class TOut>
template<class TDeriv>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::QuadraticType::Deriv, TDeriv> >::type SkinningMapping<TIn, TOut>::_applyJ( typename Out::VecDeriv& out, const sofa::helper::vector<typename QuadraticType::Deriv>& in)
{
    const VecCoord& xto = *this->toModel->getX();
    out.resize ( xto.size() );
    Deriv v;
    typename In::Deriv::Quadratic omega;

    if ( ! ( maskTo->isInUse() ) )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            out[i] = Deriv();
            for ( unsigned int j=0 ; j<in.size(); j++ )
            {
                VecIn speed;
                for (unsigned int k = 0; k < InDOFs; ++k)
                    speed[k]  = in[j][k];

                Vec3 f = ( this->J[j][i] * speed );

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
            for ( unsigned int j=0 ; j<in.size(); j++ )
            {
                VecIn speed;
                for (unsigned int k = 0; k < InDOFs; ++k)
                    speed[k]  = in[j][k];

                Vec3 f = ( this->J[j][i] * speed );

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
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_weights = weights.getValue();

    Deriv v,omega;
    if ( ! ( maskTo->isInUse() ) )
    {
        maskFrom->setInUse ( false );
        for ( unsigned int i=0; i<in.size(); i++ )
        {
            for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
            {
                Deriv f = in[i];
                v = f;
                const int idx=nbRefs.getValue() *i+m;
                const int idxReps=m_reps[idx];
                omega = cross ( rotatedPoints[idx],f );
                out[idxReps].getVCenter() += v * m_weights[idxReps][i];
                out[idxReps].getVOrientation() += omega * m_weights[idxReps][i];
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
            for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
            {
                Deriv f = in[i];
                v = f;
                const int idx=nbRefs.getValue() *i+m;
                const int idxReps=m_reps[idx];
                omega = cross ( rotatedPoints[idx],f );
                out[idxReps].getVCenter() += v * m_weights[idxReps][i];
                out[idxReps].getVOrientation() += omega * m_weights[idxReps][i];

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
    Deriv v;
    typename In::Deriv::Affine omega;
    if ( ! ( maskTo->isInUse() ) )
    {
        maskFrom->setInUse ( false );
        for ( unsigned int j=0; j<in.size(); j++ ) // VecType
        {
            for ( unsigned int i=0 ; i<out.size(); i++ ) // AffineType
            {
                MatInx3 Jt;
                Jt.transpose ( this->J[i][j] );

                Vec3 f;
                f[0] = in[j][0];
                f[1] = in[j][1];
                f[2] = in[j][2];
                VecIn speed = Jt * f;

                for (unsigned int k = 0; k < 3; ++k)
                    for (unsigned int l = 0; l < 3; ++l)
                        omega[k][l] = speed[3*k+l];
                v = Deriv ( speed[9], speed[10], speed[11] );

                out[i].getVCenter() += v;
                out[i].getVAffine() += omega;
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
            const int j= ( int ) ( *it );
            for ( unsigned int i=0 ; i<out.size(); i++ ) // AffineType
            {
                MatInx3 Jt;
                Jt.transpose ( this->J[i][j] );

                Vec3 f;
                f[0] = in[j][0];
                f[1] = in[j][1];
                f[2] = in[j][2];
                VecIn speed = Jt * f;

                for (unsigned int k = 0; k < 3; ++k)
                    for (unsigned int l = 0; l < 3; ++l)
                        omega[k][l] = speed[3*k+l];
                v = Deriv ( speed[9], speed[10], speed[11] );

                out[i].getVCenter() += v;
                out[i].getVAffine() += omega;
            }
        }
    }
}


template <class TIn, class TOut>
template<class TDeriv>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::QuadraticType::Deriv, TDeriv> >::type SkinningMapping<TIn, TOut>::_applyJT( sofa::helper::vector<typename QuadraticType::Deriv>& out, const typename Out::VecDeriv& in)
{
    Deriv v;
    typename In::Deriv::Quadratic omega;
    if ( ! ( maskTo->isInUse() ) )
    {
        maskFrom->setInUse ( false );
        for ( unsigned int j=0; j<in.size(); j++ ) // VecType
        {
            for ( unsigned int i=0 ; i<out.size(); i++ ) // QuadraticType
            {
                MatInx3 Jt;
                Jt.transpose ( this->J[i][j] );

                Vec3 f;
                f[0] = in[j][0];
                f[1] = in[j][1];
                f[2] = in[j][2];
                VecIn speed = Jt * f;

                for (unsigned int k = 0; k < 3; ++k)
                    for (unsigned int l = 0; l < 9; ++l)
                        omega[k][l] = speed[9*k+l];
                v = Deriv ( speed[27], speed[28], speed[29] );

                out[i].getVCenter() += v;
                out[i].getVQuadratic() += omega;
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
            const int j= ( int ) ( *it );
            for ( unsigned int i=0 ; i<out.size(); i++ ) // QuadraticType
            {
                MatInx3 Jt;
                Jt.transpose ( this->J[i][j] );

                Vec3 f;
                f[0] = in[j][0];
                f[1] = in[j][1];
                f[2] = in[j][2];
                VecIn speed = Jt * f;

                for (unsigned int k = 0; k < 3; ++k)
                    for (unsigned int l = 0; l < 9; ++l)
                        omega[k][l] = speed[9*k+l];
                v = Deriv ( speed[27], speed[28], speed[29] );

                out[i].getVCenter() += v;
                out[i].getVQuadratic() += omega;
            }
        }
    }
}

#endif

template <class TIn, class TOut>
template<class TMatrixDeriv>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::RigidType::MatrixDeriv, TMatrixDeriv> >::type SkinningMapping<TIn, TOut>::_applyJT_Matrix( typename RigidType::MatrixDeriv& out, const typename Out::MatrixDeriv& in)
{
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_weights = weights.getValue();
    const unsigned int nbr = nbRefs.getValue();
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

            for (unsigned int m = 0 ; m < nbr; m++)
            {
                omega = cross(rotatedPoints[nbr * indexPoint + m], data);
                flags[m_reps[nbr * indexPoint + m]] = true;
                v[m_reps[nbr * indexPoint + m]].getVCenter() += data * m_weights[m_reps[nbr * indexPoint + m]][indexPoint];
                v[m_reps[nbr * indexPoint + m]].getVOrientation() += omega * m_weights[m_reps[nbr * indexPoint + m]][indexPoint];
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
    const unsigned int nbi = this->fromModel->getX()->size();
    typename In::Deriv::Affine omega;
    typename In::VecDeriv v;
    vector<bool> flags;

    if ( !this->enableSkinning.getValue())
        return;
    const unsigned int numOut = this->J.size();

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        v.clear();
        v.resize(nbi);
        flags.clear();
        flags.resize(nbi);

        typename In::MatrixDeriv::RowIterator o = out.end();

        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        for (typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            const unsigned int indexIn = colIt.index(); // Point
            const Deriv data = colIt.val();

            for (unsigned int j=0; j<numOut; ++j) // Affine
            {
                MatInx3 Jt;
                Jt.transpose ( this->J[j][indexIn] );

                VecIn speed = Jt * data;

                typename In::Deriv::Affine affine;
                for (unsigned int k = 0; k < 3; ++k)
                    for (unsigned int l = 0; l < 3; ++l)
                        affine[k][l] = speed[3*k+l];
                const Vec3 pos( speed[9], speed[10], speed[11] );
                InDeriv value(pos,affine);
                o.addCol(j, value);
            }
        }
    }
}


template <class TIn, class TOut>
template<class TMatrixDeriv>
typename enable_if<Equal<typename SkinningMapping<TIn, TOut>::QuadraticType::MatrixDeriv, TMatrixDeriv> >::type SkinningMapping<TIn, TOut>::_applyJT_Matrix( typename QuadraticType::MatrixDeriv& out, const typename Out::MatrixDeriv& in)
{
    const unsigned int nbi = this->fromModel->getX()->size();
    typename In::Deriv::Quadratic omega;
    typename In::VecDeriv v;
    vector<bool> flags;

    if ( !this->enableSkinning.getValue())
        return;
    const unsigned int numOut = this->J.size();

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        v.clear();
        v.resize(nbi);
        flags.clear();
        flags.resize(nbi);

        typename In::MatrixDeriv::RowIterator o = out.end();

        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        for (typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            const unsigned int indexIn = colIt.index();
            const Deriv data = colIt.val();

            for (unsigned int j=0; j<numOut; ++j)
            {
                MatInx3 Jt;
                Jt.transpose ( this->J[j][indexIn] );

                VecIn speed = Jt * data;

                typename In::Deriv::Quadratic quad;
                for (unsigned int k = 0; k < 9; ++k)
                    for (unsigned int l = 0; l < 3; ++l)
                        quad[k][l] = speed[3*k+l];
                const Vec3 pos ( speed[27], speed[28], speed[29] );

                InDeriv value(pos,quad);
                o.addCol(j, value);
            }
        }
    }
}

#endif



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
