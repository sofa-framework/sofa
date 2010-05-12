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
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/core/behavior/MechanicalMapping.inl>
#include <sofa/core/behavior/MechanicalState.h>
#include <string>
#include <iostream>
#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <sofa/helper/gl/glText.inl>


#ifdef SOFA_DEV
#include <sofa/helper/DualQuat.h>
#include <sofa/component/topology/HexahedronGeodesicalDistance.inl>
#include <sofa/component/topology/DynamicSparseGridTopologyContainer.h>
#include <sofa/simulation/common/Simulation.h>
#endif

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using sofa::component::topology::TriangleSetTopologyContainer;


template <class BasicMapping>
SkinningMapping<BasicMapping>::SkinningMapping ( In* from, Out* to )
    : Inherit ( from, to )
    , repartition ( initData ( &repartition,"repartition","repartition between input DOFs and skinned vertices" ) )
    , coefs ( initData ( &coefs,"coefs","weights list for the influences of the references Dofs" ) )
    , weightGradients ( initData ( &weightGradients,"weightGradients","weight gradients list for the influences of the references Dofs" ) )
    , nbRefs ( initData ( &nbRefs, ( unsigned ) 3,"nbRefs","nb references for skinning" ) )
    , showBlendedFrame ( initData ( &showBlendedFrame, false, "showBlendedFrame","weights list for the influences of the references Dofs" ) )
    , showDefTensors ( initData ( &showDefTensors, false, "showDefTensors","show computed deformation tensors." ) )
    , showDefTensorsValues ( initData ( &showDefTensorsValues, false, "showDefTensorsValues","Show Deformation Tensors Values." ) )
    , showDefTensorScale ( initData ( &showDefTensorScale, 1.0, "showDefTensorScale","deformation tensor scale." ) )
    , showFromIndex ( initData ( &showFromIndex, ( unsigned ) 0, "showFromIndex","Displayed From Index." ) )
    , showDistancesValues ( initData ( &showDistancesValues, false, "showDistancesValues","Show dstances values." ) )
    , showCoefs ( initData ( &showCoefs, false, "showCoefs","Show coeficients." ) )
    , showGammaCorrection ( initData ( &showGammaCorrection, 1.0, "showGammaCorrection","Correction of the Gamma by a power" ) )
    , showCoefsValues ( initData ( &showCoefsValues, false, "showCoefsValues","Show coeficients values." ) )
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
#endif
    , wheightingType ( initData ( &wheightingType, "wheightingType","Weighting computation method." ) )
    , interpolationType ( initData ( &interpolationType,  "interpolationType","Interpolation method." ) )
    , distanceType ( initData ( &distanceType, "distanceType","Distance computation method." ) )
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

    sofa::helper::OptionsGroup wheightingTypeOptions(5,"None","Inverse Distance Square","Linear", "Hermite", "Spline");
    wheightingTypeOptions.setSelectedItem(WEIGHT_INVDIST_SQUARE);
    wheightingType.setValue(wheightingTypeOptions);


    sofa::helper::OptionsGroup interpolationTypeOptions(2,"Linear","Dual Quaternion");
    interpolationTypeOptions.setSelectedItem(INTERPOLATION_DUAL_QUATERNION);
    interpolationType.setValue(interpolationTypeOptions);


    sofa::helper::OptionsGroup distanceTypeOptions(3,"Euclidian","Geodesic", "Harmonic");
    distanceTypeOptions.setSelectedItem(DISTANCE_EUCLIDIAN);
    distanceType.setValue(distanceTypeOptions);
}

template <class BasicMapping>
SkinningMapping<BasicMapping>::~SkinningMapping ()
{
}


template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeInitPos ( )
{
    const VecCoord& xto = ( this->toModel->getX0() == NULL)?*this->toModel->getX():*this->toModel->getX0();
    const VecInCoord& xfrom = *this->fromModel->getX0();

    const vector<int>& m_reps = repartition.getValue();

    switch ( interpolationType.getValue().getSelectedId() )
    {
    case INTERPOLATION_LINEAR:
    {
        initPos.resize ( xto.size() * nbRefs.getValue() );
        for ( unsigned int i = 0; i < xto.size(); i++ )
            for ( unsigned int m = 0; m < nbRefs.getValue(); m++ )
            {
                initPos[nbRefs.getValue() *i+m] = xfrom[m_reps[nbRefs.getValue() *i+m]].getOrientation().inverseRotate ( xto[i] - xfrom[m_reps[nbRefs.getValue() *i+m]].getCenter() );
            }
        break;
    }
#ifdef SOFA_DEV
    case INTERPOLATION_DUAL_QUATERNION:
    {
        initPos.resize ( xto.size() );
        for ( unsigned int i = 0; i < xto.size(); i++ )
            initPos[i] = xto[i];
        break;
    }
#endif
    default:
    {}
    }
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeDistances ()
{
#ifdef SOFA_DEV
    if( this->computeAllMatrices.getValue())
    {
        VecInCoord& xfrom0 = *this->fromModel->getX0();

        GeoVecCoord tmpFrom;
        tmpFrom.resize ( xfrom0.size() );
        for ( unsigned int i = 0; i < xfrom0.size(); i++ )
            tmpFrom[i] = xfrom0[i].getCenter();

        if ( distanceType.getValue().getSelectedId() == DISTANCE_GEODESIC && this->computeAllMatrices.getValue()) geoDist->computeGeodesicalDistanceMap ( tmpFrom );
        if ( distanceType.getValue().getSelectedId() == DISTANCE_HARMONIC && this->computeAllMatrices.getValue()) geoDist->computeHarmonicCoordsDistanceMap ( tmpFrom );
    }
#endif

    getDistances( 0);
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::getDistances( int xfromBegin)
{
    const VecCoord& xto0 = ( this->toModel->getX0() == NULL)?*this->toModel->getX():*this->toModel->getX0();
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

template <class BasicMapping>
void SkinningMapping<BasicMapping>::sortReferences( vector<int>& references)
{
    VecCoord& xto = ( this->toModel->getX0() == NULL)?*this->toModel->getX():*this->toModel->getX0();
    VecInCoord& xfrom = *this->fromModel->getX0();
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


template <class BasicMapping>
void SkinningMapping<BasicMapping>::init()
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
    VecInCoord& xfrom = *this->fromModel->getX0();
    if ( this->initPos.empty() && this->toModel!=NULL && computeWeights==true && coefs.getValue().size() ==0 )
    {
        if ( wheightingType.getValue().getSelectedId() == WEIGHT_LINEAR || wheightingType.getValue().getSelectedId() == WEIGHT_HERMITE )
            nbRefs.setValue ( 2 );

        if ( wheightingType.getValue().getSelectedId() == WEIGHT_SPLINE)
            nbRefs.setValue ( 4 );

        if ( xfrom.size() < nbRefs.getValue() || interpolationType.getValue().getSelectedId() == INTERPOLATION_DUAL_QUATERNION)
            nbRefs.setValue ( xfrom.size() );


        computeDistances();
        vector<int>& m_reps = * ( repartition.beginEdit() );
        if ( interpolationType.getValue().getSelectedId() == INTERPOLATION_LINEAR)
        {
            sortReferences ( m_reps);
        }
        else if ( interpolationType.getValue().getSelectedId() == INTERPOLATION_DUAL_QUATERNION)
        {
            VecCoord& xto = ( this->toModel->getX0() == NULL)?*this->toModel->getX():*this->toModel->getX0();
            VecInCoord& xfrom = *this->fromModel->getX0();
            const unsigned int& nbRef = nbRefs.getValue();
            m_reps.clear();
            m_reps.resize ( nbRefs.getValue() *xto.size() );
            for ( unsigned int i=0; i<xfrom.size(); i++ )
                for ( unsigned int j=0; j<xto.size(); j++ )
                    m_reps[nbRef *j+i] = i;
        }
        repartition.endEdit();
        updateWeights ();
        computeInitPos ();

#ifdef SOFA_DEV
        if ( interpolationType.getValue().getSelectedId() == INTERPOLATION_DUAL_QUATERNION )
        {
            if ( this->computeJ.getValue() || this->computeAllMatrices.getValue())
                this->J.resize ( xfrom.size() );

            DUALQUAT qi0;
            this->T.resize ( xfrom.size() );
            for ( unsigned int i = 0; i < xfrom.size(); i++ )
            {
                XItoQ( qi0, xfrom[i]);
                computeDqT ( this->T[i], qi0 );
            }

            sofa::component::topology::DynamicSparseGridTopologyContainer* hexaContainer;
            this->getContext()->get( hexaContainer);
            double volume = voxelVolume.getValue();
            if ( hexaContainer && geoDist) volume = geoDist->initTargetStep.getValue()*geoDist->initTargetStep.getValue()*geoDist->initTargetStep.getValue() * hexaContainer->voxelSize.getValue()[0]*hexaContainer->voxelSize.getValue()[1]*hexaContainer->voxelSize.getValue()[2];
            const VecCoord& xto = *this->toModel->getX();
            this->vol.resize( xto.size());
            for ( unsigned int i = 0; i < xto.size(); i++) this->vol[i] = volume;
            this->volMass.resize( xto.size());
            for ( unsigned int i = 0; i < xto.size(); i++) this->volMass[i] = 1.0;

            // Set the influence of the first frames to infinite
            vector<double>& vRadius = (*newFrameWeightingRadius.beginEdit());
            vRadius.resize( xfrom.size());
            for (unsigned int i = 0; i < xfrom.size(); ++i) vRadius[i] = 0xFFF;
            newFrameWeightingRadius.endEdit();
        }

        apply0();
#endif
    }
    else if ( computeWeights == false || coefs.getValue().size() !=0 )
    {
        computeInitPos();
    }

    this->BasicMapping::init();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::clear()
{
    this->initPos.clear();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setWeightsToHermite()
{
    wheightingType.beginEdit()->setSelectedItem(WEIGHT_HERMITE);
    wheightingType.endEdit();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setWeightsToLinear()
{
    wheightingType.beginEdit()->setSelectedItem(WEIGHT_LINEAR);
    wheightingType.endEdit();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setWeightsToInvDist()
{
    wheightingType.beginEdit()->setSelectedItem(WEIGHT_INVDIST_SQUARE);
    wheightingType.endEdit();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setInterpolationToLinear()
{
    interpolationType.beginEdit()->setSelectedItem(INTERPOLATION_LINEAR);
    interpolationType.endEdit();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setInterpolationToDualQuaternion()
{
    interpolationType.beginEdit()->setSelectedItem(INTERPOLATION_DUAL_QUATERNION);
    interpolationType.endEdit();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::updateWeights ()
{
    VecCoord& xto = ( this->toModel->getX0() == NULL)?*this->toModel->getX():*this->toModel->getX0();
    VecInCoord& xfrom = *this->fromModel->getX0();

    VVD& m_coefs = * ( coefs.beginEdit() );
    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());
    const vector<int>& m_reps = repartition.getValue();

    m_coefs.resize ( xfrom.size() );
    for ( unsigned int i=0; i<xfrom.size(); i++ )
        m_coefs[i].resize ( xto.size() );
    m_dweight.resize ( xfrom.size() );
    for ( unsigned int i=0; i<xfrom.size(); i++ )
        m_dweight[i].resize ( xto.size() );

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
                    m_coefs[indexFrom][j] = geoDist->harmonicMaxValue.getValue() - distances[indexFrom][j];
                    if ( distances[indexFrom][j] < 0.0) distances[indexFrom][j] = 0.0;
                    if ( distances[indexFrom][j] > geoDist->harmonicMaxValue.getValue()) distances[indexFrom][j] = geoDist->harmonicMaxValue.getValue();
                    m_dweight[indexFrom][j] = - distGradients[indexFrom][j];
                }
                else
                {
#endif
                    m_coefs[indexFrom][j] = distances[indexFrom][j];
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
        sortReferences( tmpReps); // We sort references even for DUALQUAT_INTERPOLATION
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            for ( unsigned int j=0; j<xfrom.size(); j++ )
            {
                m_coefs[j][i] = 0.0;
                m_dweight[j][i] = Coord();
            }
            Vec3d r1r2, r1p;
            r1r2 = xfrom[tmpReps[nbRefs.getValue() *i+1]].getCenter() - xfrom[tmpReps[nbRefs.getValue() *i+0]].getCenter();
            r1p  = xto[i] - xfrom[tmpReps[nbRefs.getValue() *i+0]].getCenter();
            double r1r2NormSquare = r1r2.norm()*r1r2.norm();
            double wi = ( r1r2*r1p ) / ( r1r2NormSquare);

            // Abscisse curviligne
            m_coefs[tmpReps[nbRefs.getValue() *i+0]][i] = ( 1 - wi );
            m_coefs[tmpReps[nbRefs.getValue() *i+1]][i] = wi;
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
                if ( distances[indexFrom][j])
                    m_coefs[indexFrom][j] = 1 / (distances[indexFrom][j]*distances[indexFrom][j]);
                else
                    m_coefs[indexFrom][j] = 0xFFF;
                if ( distances[indexFrom][j])
                    m_dweight[indexFrom][j] = - distGradients[indexFrom][j] / (double)(distances[indexFrom][j]*distances[indexFrom][j]*distances[indexFrom][j]) * 2.0;
                else
                    m_dweight[indexFrom][j] = Coord();
            }
        }

        break;
    }
    case WEIGHT_HERMITE:
    {
        vector<int> tmpReps;
        sortReferences( tmpReps); // We sort references even for DUALQUAT_INTERPOLATION
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            for ( unsigned int j=0; j<xfrom.size(); j++ )
            {
                m_coefs[j][i] = 0.0;
                m_dweight[j][i] = Coord();
            }
            Vec3d r1r2, r1p;
            double wi;
            r1r2 = xfrom[tmpReps[nbRefs.getValue() *i+1]].getCenter() - xfrom[tmpReps[nbRefs.getValue() *i+0]].getCenter();
            r1p  = xto[i] - xfrom[tmpReps[nbRefs.getValue() *i+0]].getCenter();
            double r1r2NormSquare = r1r2.norm()*r1r2.norm();
            wi = ( r1r2*r1p ) / r1r2NormSquare;

            // Fonctions d'Hermite
            m_coefs[tmpReps[nbRefs.getValue() *i+0]][i] = 1-3*wi*wi+2*wi*wi*wi;
            m_coefs[tmpReps[nbRefs.getValue() *i+1]][i] = 3*wi*wi-2*wi*wi*wi;

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
        sortReferences( tmpReps); // We sort references even for DUALQUAT_INTERPOLATION
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            // Clear all weights and dweights.
            for ( unsigned int j=0; j<xfrom.size(); j++ )
            {
                m_coefs[j][i] = 0.0;
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
            m_coefs[sortedFrames[0]][i] = 1-3*u*u+2*u*u*u;
            m_coefs[sortedFrames[1]][i] = u*u*u - 2*u*u + u;
            m_coefs[sortedFrames[2]][i] = 3*u*u-2*u*u*u;
            m_coefs[sortedFrames[3]][i] = u*u*u - u*u;

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
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setWeightCoefs ( VVD &weights )
{
    VVD * m_coefs = coefs.beginEdit();
    m_coefs->clear();
    m_coefs->insert ( m_coefs->begin(), weights.begin(), weights.end() );
    coefs.endEdit();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setRepartition ( vector<int> &rep )
{
    vector<int> * m_reps = repartition.beginEdit();
    m_reps->clear();
    m_reps->insert ( m_reps->begin(), rep.begin(), rep.end() );;
    repartition.endEdit();
}


template <class BasicMapping>
void SkinningMapping<BasicMapping>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_coefs = coefs.getValue();

    switch ( interpolationType.getValue().getSelectedId()  )
    {
    case INTERPOLATION_LINEAR:
    {
        rotatedPoints.resize ( initPos.size() );
        out.resize ( initPos.size() / nbRefs.getValue() );
        for ( unsigned int i=0 ; i<out.size(); i++ )
        {
            out[i] = Coord();
            for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
            {

                const int& idx=nbRefs.getValue() *i+m;
                const int& idxReps=m_reps[idx];

                // Save rotated points for applyJ/JT
                rotatedPoints[idx] = in[idxReps].getOrientation().rotate ( initPos[idx] );

                // And add each reference frames contributions to the new position out[i]
                out[i] += ( in[idxReps ].getCenter() + rotatedPoints[idx] ) * m_coefs[idxReps][i];
            }
        }
        break;
    }
#ifdef SOFA_DEV
    case INTERPOLATION_DUAL_QUATERNION:
    {
        if ( this->T.size() != in.size()) apply0();
        if ( !enableSkinning.getValue()) return;
        unsigned int i,j,k,l,m,n, nbDOF=in.size(), nbP=out.size();

        // allocate temporary variables
        VMat86 L ( nbDOF );
        VMat86 TL ( nbDOF );
        VDUALQUAT qrel ( nbDOF );

        DUALQUAT q,b,bn,V;
        Mat33 R,dR,Ainv,E,dE;
        Mat88 N,dN;
        Mat86 NTL;
        Mat38 Q,dQ,QN,QNT;
        Mat83 W,NW,dNW;
        Vec3 t;
        double QEQ0,Q0Q0,Q0,q0V0,q0Ve,qeV0;
        Mat44 q0q0T,q0qeT,qeq0T,q0V0T,V0q0T,q0VeT,Veq0T,qeV0T,V0qeT;
        const VVD& w = coefs.getValue();
        const SVector<SVector<GeoCoord> >& dw = weightGradients.getValue();
        VVec6& e = this->deformationTensors;

        // Resize vectors
        e.resize ( nbP );
        this->det.resize ( nbP );
        this->J.resize ( nbDOF );
        this->B.resize ( nbDOF );
        this->A.resize( nbP);
        this->ddet.resize( nbDOF);
        for ( i=0; i<nbDOF; i++ )
        {
            this->J[i].resize ( nbP );
            this->B[i].resize ( nbP );
            this->ddet[i].resize( nbP);
        }

        VecInCoord& xfrom = *this->fromModel->getX();

        if ( this->T.size() != nbDOF) updateDataAfterInsertion();

        //apply
        for ( i=0; i<nbDOF; i++ )
        {
            XItoQ ( q,xfrom[i] );		//update DOF quats
            computeDqL ( L[i],q,xfrom[i].getCenter() );	//update L=d(q)/Omega
            TL[i]=this->T[i]*L[i];	//update TL
            computeQrel ( qrel[i], this->T[i], q ); //update qrel=Tq
        }

        for ( i=0; i<nbP; i++ )
        {
            BlendDualQuat ( b,bn,QEQ0,Q0Q0,Q0,i,qrel,w );	//skinning: sum(wTq)
            computeDqRigid ( R, t, bn );			//update Rigid
            for ( k=0; k<3; k++ )
            {
                out[i][k]=t[k];  //update skinned points
                for ( j=0; j<3; j++ ) out[i][k]+=R[k][j]*initPos[i][j];
            }

            if ( !(this->computeJ.getValue() || this->computeAllMatrices.getValue())) continue; // if we don't want to compute all the matrices

            computeDqN_constants ( q0q0T, q0qeT, qeq0T, bn );
            computeDqN ( N, q0q0T, q0qeT, qeq0T, QEQ0, Q0Q0, Q0 );	//update N=d(bn)/d(b)
            computeDqQ ( Q, bn, initPos[i] );	//update Q=d(P)/d(bn)
            W.fill ( 0 );
            for ( j=0; j<nbDOF; j++ )  for ( k=0; k<4; k++ ) for ( l=0; l<3; l++ )
                    {
                        W[k][l]+=dw[j][i][l]*qrel[j].q0[k];  //update W=sum(wrel.dw)=d(b)/d(p)
                        W[k+4][l]+=dw[j][i][l]*qrel[j].qe[k];
                    }
            NW=N*W; // update NW
            QN=Q*N; // update QN

            // grad def = d(P)/d(p0)
            this->A[i]=R+Q*NW; // update A=R+QNW

            // strain
            for ( k=0; k<3; k++ )
            {
                for ( l=0; l<3; l++ )
                {
                    E[k][l]=0;
                    for ( m=0; m<3; m++ ) E[k][l]+=this->A[i][m][l]*this->A[i][m][k];
                }
                E[k][k]-=1.;
            }
            for ( k=0; k<3; k++ ) for ( l=0; l<3; l++ ) E[k][l]/=2.; // update E=1/2(A^TA-I)
            e[i][0]=E[0][0];
            e[i][1]=E[1][1];
            e[i][2]=E[2][2];
            e[i][3]=E[0][1];
            e[i][4]=E[0][2];
            e[i][5]=E[1][2]; // column form

            this->det[i]=determinant ( this->A[i] );
            invertMatrix ( Ainv,this->A[i] );

            for ( j=0; j<nbDOF; j++ )
            {
                // J = d(P_i)/Omega_j
                NTL=N*TL[j]; // update NTL
                this->J[j][i]=Q*NTL;
                this->J[j][i]*=w[j][i]; // Update J=wiQNTL

                if ( !this->computeAllMatrices.getValue()) continue; // if we want to compute just the J matrix

                QNT=QN*this->T[j]; // Update QNT

                // B = d(strain)/Omega_j
                this->B[j][i].fill ( 0 );
                for ( l=0; l<6; l++ )
                {
                    if ( w[j][i]==0 ) dE.fill(0);
                    else
                    {
                        // d(grad def)/Omega_j
                        for ( k=0; k<4; k++ )
                        {
                            V.q0[k]=NTL[k][l];
                            V.qe[k]=NTL[k+4][l];
                        }
                        computeDqDR ( dR, bn, V );
                        dE=dR;
                        computeDqDQ ( dQ, initPos[i], V );
                        dE+=dQ*NW;
                        for ( k=0; k<4; k++ )
                        {
                            V.q0[k]=TL[j][k][l];
                            V.qe[k]=TL[j][k+4][l];
                        }
                        computeDqDN_constants ( q0V0T, V0q0T, q0V0, q0VeT, Veq0T, q0Ve, qeV0T, V0qeT, qeV0, bn, V );
                        computeDqDN ( dN, q0q0T, q0qeT, qeq0T, QEQ0, Q0Q0, Q0, q0V0T, V0q0T, q0V0, q0VeT, Veq0T, q0Ve, qeV0T, V0qeT, qeV0, V );
                        dNW=dN*W;
                        dE+=Q*dNW;
                        dE*=w[j][i];
                    }
                    for ( k=0; k<3; k++ )  for ( m=0; m<3; m++ ) for ( n=0; n<8; n++ ) dE[k][m]+=QNT[k][n]*L[j][n][l]*dw[j][i][m];

                    // determinant derivatives
                    this->ddet[j][i][l]=0;
                    for (k=0; k<3; k++)  for (m=0; m<3; m++) this->ddet[j][i][l]+=dE[k][m]*Ainv[m][k];
                    this->ddet[j][i][l]*=this->det[i];

                    // B=1/2(graddef^T.d(graddef)/Omega_j + d(graddef)/Omega_j^T.graddef)
                    for ( n=0; n<3; n++ )
                    {
                        this->B[j][i][0][l]+= ( dE[n][0]*this->A[i][n][0] );
                        this->B[j][i][1][l]+= ( dE[n][1]*this->A[i][n][1] );
                        this->B[j][i][2][l]+= ( dE[n][2]*this->A[i][n][2] );
                        this->B[j][i][3][l]+=0.5* ( dE[n][0]*this->A[i][n][1]+this->A[i][n][0]*dE[n][1] );
                        this->B[j][i][4][l]+=0.5* ( dE[n][0]*this->A[i][n][2]+this->A[i][n][0]*dE[n][2] );
                        this->B[j][i][5][l]+=0.5* ( dE[n][1]*this->A[i][n][2]+this->A[i][n][1]*dE[n][2] );
                    }
                }
            }
        }
        break;
    }
#endif
    default:
    {}
    }
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_coefs = coefs.getValue();
    VecCoord& xto = *this->toModel->getX();
    out.resize ( xto.size() );
    Deriv v,omega;

    if ( ! ( maskTo->isInUse() ) )
    {
        switch ( interpolationType.getValue().getSelectedId()  )
        {
        case INTERPOLATION_LINEAR:
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
                    out[i] += ( v - cross ( rotatedPoints[idx],omega ) ) * m_coefs[idxReps][i];
                }
            }
            break;
        }
#ifdef SOFA_DEV
        case INTERPOLATION_DUAL_QUATERNION:
        {
            if ( !enableSkinning.getValue()) return;
            for ( unsigned int j=0; j<out.size(); j++ )
            {
                out[j] = Deriv();
                for ( unsigned int i=0 ; i<in.size(); i++ )
                {
                    Vec6 speed;
                    speed[0] = in[i].getVOrientation() [0];
                    speed[1] = in[i].getVOrientation() [1];
                    speed[2] = in[i].getVOrientation() [2];
                    speed[3] = in[i].getVCenter() [0];
                    speed[4] = in[i].getVCenter() [1];
                    speed[5] = in[i].getVCenter() [2];

                    Vec3 f = ( this->J[i][j] * speed );

                    out[j] += Deriv ( f[0], f[1], f[2] );
                }
            }
            break;
        }
#endif
        default:
        {}
        }
    }
    else
    {
        typedef core::behavior::BaseMechanicalState::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;
        switch ( interpolationType.getValue().getSelectedId()  )
        {
        case INTERPOLATION_LINEAR:
        {
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
                    out[i] += ( v - cross ( rotatedPoints[idx],omega ) ) * m_coefs[idxReps][i];
                }
            }
            break;
        }
#ifdef SOFA_DEV
        case INTERPOLATION_DUAL_QUATERNION:
        {
            if ( !enableSkinning.getValue()) return;
            for ( it=indices.begin(); it!=indices.end(); it++ )
            {
                const int j= ( int ) ( *it );
                out[j] = Deriv();
                for ( unsigned int i=0 ; i<in.size(); i++ )
                {
                    Vec6 speed;
                    speed[0] = in[i].getVOrientation() [0];
                    speed[1] = in[i].getVOrientation() [1];
                    speed[2] = in[i].getVOrientation() [2];
                    speed[3] = in[i].getVCenter() [0];
                    speed[4] = in[i].getVCenter() [1];
                    speed[5] = in[i].getVCenter() [2];

                    Vec3 f = ( this->J[i][j] * speed );

                    out[j] += Deriv ( f[0], f[1], f[2] );
                }
            }
            break;
        }
#endif
        default:
        {}
        }
    }
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_coefs = coefs.getValue();

    Deriv v,omega;
    if ( ! ( maskTo->isInUse() ) )
    {
        switch ( interpolationType.getValue().getSelectedId()  )
        {
        case INTERPOLATION_LINEAR:
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
                    out[idxReps].getVCenter() += v * m_coefs[idxReps][i];
                    out[idxReps].getVOrientation() += omega * m_coefs[idxReps][i];
                }
            }
            break;
        }
#ifdef SOFA_DEV
        case INTERPOLATION_DUAL_QUATERNION:
        {
            if ( !enableSkinning.getValue()) return;
            for ( unsigned int j=0; j<in.size(); j++ )
            {
                for ( unsigned int i=0 ; i<out.size(); i++ )
                {
                    Mat63 Jt;
                    Jt.transpose ( this->J[i][j] );

                    Vec3 f;
                    f[0] = in[j][0];
                    f[1] = in[j][1];
                    f[2] = in[j][2];
                    Vec6 speed = Jt * f;

                    omega = Deriv ( speed[0], speed[1], speed[2] );
                    v = Deriv ( speed[3], speed[4], speed[5] );

                    out[i].getVCenter() += v;
                    out[i].getVOrientation() += omega;
                }
            }
            break;
        }
#endif
        default:
        {}
        }
    }
    else
    {
        typedef core::behavior::BaseMechanicalState::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;
        switch ( interpolationType.getValue().getSelectedId()  )
        {
        case INTERPOLATION_LINEAR:
        {
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
                    out[idxReps].getVCenter() += v * m_coefs[idxReps][i];
                    out[idxReps].getVOrientation() += omega * m_coefs[idxReps][i];

                    maskFrom->insertEntry ( idxReps );
                }
            }
            break;
        }
#ifdef SOFA_DEV
        case INTERPOLATION_DUAL_QUATERNION:
        {
            if ( !enableSkinning.getValue()) return;
            for ( it=indices.begin(); it!=indices.end(); it++ )
            {
                const int j= ( int ) ( *it );
                for ( unsigned int i=0 ; i<out.size(); i++ )
                {
                    Mat63 Jt;
                    Jt.transpose ( this->J[i][j] );

                    Vec3 f;
                    f[0] = in[j][0];
                    f[1] = in[j][1];
                    f[2] = in[j][2];
                    Vec6 speed = Jt * f;

                    omega = Deriv ( speed[0], speed[1], speed[2] );
                    v = Deriv ( speed[3], speed[4], speed[5] );

                    out[i].getVCenter() += v;
                    out[i].getVOrientation() += omega;
                }
            }
            break;
        }
#endif
        default:
        {}
        }
    }

}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::applyJT ( typename In::VecConst& out, const typename Out::VecConst& in )
{
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_coefs = coefs.getValue();
    const unsigned int nbr = nbRefs.getValue();
    const unsigned int nbi = this->fromModel->getX()->size();
    Deriv omega;
    typename In::VecDeriv v;
    vector<bool> flags;
    int outSize = out.size();
    out.resize ( in.size() + outSize ); // we can accumulate in "out" constraints from several mappings
    switch ( interpolationType.getValue().getSelectedId()  )
    {
    case INTERPOLATION_LINEAR:
    {
        for ( unsigned int i=0; i<in.size(); i++ )
        {
            v.clear();
            v.resize ( nbi );
            flags.clear();
            flags.resize ( nbi );
            OutConstraintIterator itOut;
            std::pair< OutConstraintIterator, OutConstraintIterator > iter=in[i].data();

            for ( itOut=iter.first; itOut!=iter.second; itOut++ )
            {
                unsigned int indexIn = itOut->first;
                Deriv data = ( Deriv ) itOut->second;
                Deriv f = data;
                for ( unsigned int m=0 ; m<nbr; m++ )
                {
                    omega = cross ( rotatedPoints[nbr*indexIn+m],f );
                    flags[m_reps[nbr*indexIn+m] ] = true;
                    v[m_reps[nbr*indexIn+m] ].getVCenter() += f * m_coefs[m_reps[nbr*indexIn+m]][i];
                    v[m_reps[nbr*indexIn+m] ].getVOrientation() += omega * m_coefs[m_reps[nbr*indexIn+m]][i];
                }
            }
            for ( unsigned int j=0 ; j<nbi; j++ )
            {
                //if (!(v[i] == typename In::Deriv()))
                if ( flags[j] )
                    out[outSize+i].add ( j,v[j] );
            }
        }
        break;
    }
#ifdef SOFA_DEV
    case INTERPOLATION_DUAL_QUATERNION:
    {
        if ( !enableSkinning.getValue()) return;
        const unsigned int numOut=this->J.size();

        for ( unsigned int i=0; i<in.size(); i++ )
        {
            OutConstraintIterator itOut;
            std::pair< OutConstraintIterator, OutConstraintIterator > iter=in[i].data();

            for ( itOut=iter.first; itOut!=iter.second; itOut++ )
            {
                unsigned int indexIn = itOut->first;
                Deriv data = ( Deriv ) itOut->second;

                for (unsigned int j=0; j<numOut; ++j)
                {
                    Mat63 Jt;
                    Jt.transpose ( this->J[j][indexIn] );

                    Vec6 speed = Jt * data;

                    const Vec3 pos( speed[3], speed[4], speed[5] );
                    const Vec3 ori(speed[0], speed[1], speed[2] );
                    InDeriv value(pos,ori);
                    out[outSize+i].add(j,value);
                }
            }
        }
        break;
    }
#endif
    default:
    {}
    }
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::draw()
{
    const typename Out::VecCoord& xto = *this->toModel->getX();
    const typename In::VecCoord& xfrom = *this->fromModel->getX();
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_coefs = coefs.getValue();
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
        if ( interpolationType.getValue().getSelectedId()  != INTERPOLATION_DUAL_QUATERNION )
        {
            glDisable ( GL_LIGHTING );
            glPointSize ( 1 );
            glColor4f ( 1,1,0,1 );
            glBegin ( GL_LINES );

            for ( unsigned int i=0; i<xto.size(); i++ )
            {
                for ( unsigned int m=0 ; m<nbRef; m++ )
                {
                    const int idxReps=m_reps[nbRef *i+m];
                    double coef = m_coefs[idxReps][i];
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
    }

    // Display  m_reps for each points
    if ( showReps.getValue())
    {
        for ( unsigned int i=0; i<xto.size(); i++ )
            sofa::helper::gl::GlText::draw ( m_reps[nbRefs.getValue() *i+0]*scale, xto[i], textScale );
    }

    // Display distances for each points
    if ( showDistancesValues.getValue())
    {
        glColor3f( 1.0, 1.0, 1.0);
        for ( unsigned int i=0; i<xto.size(); i++ )
            sofa::helper::gl::GlText::draw ( (int)(distances[showFromIndex.getValue()%distances.size()][i]*scale), xto[i], textScale );
    }

    // Display coefs for each points
    if ( showCoefsValues.getValue())
    {
        glColor3f( 1.0, 1.0, 1.0);
        for ( unsigned int i=0; i<xto.size(); i++ )
            sofa::helper::gl::GlText::draw ( (int)(m_coefs[showFromIndex.getValue()%m_coefs.size()][i]*scale), xto[i], textScale );
    }

    // Display gradient values for each points
    if ( showGradientsValues.getValue())
    {
        char txt[100];
        glColor3f( 0.5, 0.5, 0.5);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            const Vec3& grad = dw[showFromIndex.getValue()%dw.size()][i];
            sprintf( txt, "( %i, %i, %i)", (int)(grad[0]*scale), (int)(grad[1]*scale), (int)(grad[2]*scale));
            sofa::helper::gl::GlText::draw ( txt, xto[i], textScale );
        }
    }

#ifdef SOFA_DEV
    // Display distances values for each points
    if ( showDefTensorsValues.getValue())
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
#endif

    // Display gradient for each points
    if ( showGradients.getValue())
    {
        glColor3f ( 0.0, 1.0, 0.3 );
        glBegin ( GL_LINES );
        const vector<GeoCoord>& gradMap = dw[showFromIndex.getValue()%dw.size()];
        for ( unsigned int j = 0; j < gradMap.size(); j++ )
        {
            const Coord& point = xto[j];
            glVertex3f ( point[0], point[1], point[2] );
            glVertex3f ( point[0] + gradMap[j][0] * showGradientsScaleFactor.getValue(), point[1] + gradMap[j][1] * showGradientsScaleFactor.getValue(), point[2] + gradMap[j][2] * showGradientsScaleFactor.getValue() );
        }
        glEnd();
    }
    //*/

#ifdef SOFA_DEV
    //*  Continuous animation of the reference frames along the beam (between frames i and i+1)
    if ( showBlendedFrame.getValue() && interpolationType.getValue().getSelectedId() == INTERPOLATION_DUAL_QUATERNION )
    {
        bool anim = true;
        static unsigned int step = 0;
        double transfoM4[16];
        unsigned int nbSteps = 500;
        step++;
        if ( step > nbSteps ) step = 0;

        for ( unsigned int i=1; i<xfrom.size(); i++ )
        {
            DualQuat dq1 ( xfrom[i-1].getCenter(), xfrom[i-1].getOrientation() );
            DualQuat dq2 ( xfrom[i].getCenter(), xfrom[i].getOrientation() );

            if ( anim )
            {
                DualQuat dqi = dq1 * ( step/ ( ( float ) nbSteps ) ) + dq2 * ( 1- ( step/ ( ( float ) nbSteps ) ) );
                dqi.normalize();
                dqi.toGlMatrix ( transfoM4 );
                sofa::helper::gl::Axis::draw ( transfoM4, 0.5 );
            }
            else
            {
                unsigned int nbReferenceFrame = 10;
                for ( unsigned int j = 0; j < nbReferenceFrame; j++ )
                {
                    DualQuat dqi = dq1 * ( 1 - ( j/ ( ( float ) nbReferenceFrame ) ) ) + dq2 * ( j/ ( ( float ) nbReferenceFrame ) );
                    dqi.normalize();
                    dqi.toGlMatrix ( transfoM4 );
                    sofa::helper::gl::Axis::draw ( transfoM4, 0.5 );
                }
            }
        }
    }
    //*/

    // Deformation tensor show
    if ( showDefTensors.getValue() && interpolationType.getValue().getSelectedId()  == INTERPOLATION_DUAL_QUATERNION && this->computeAllMatrices.getValue() )
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
                    double color = 0.5 + ( e[0] + e[1] + e[2])/showDefTensorScale.getValue();
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
                double color = 0.5 + ( e[0] + e[1] + e[2])/2.0;
                glColor3f( 0.0, color, 1.0-color);// /*e[0]*/, e[1], e[2]);
                glVertex3f( xto[i][0], xto[i][1], xto[i][2]);
            }
            glEnd();
        }
    }

    // Coefs show
    if ( showCoefs.getValue())
    {
        // Compute min and max values.
        double minValue = 0xFFFF;
        double maxValue = -0xFFF;
        for ( unsigned int j = 0; j < xto.size(); j++)
        {
            if ( m_coefs[showFromIndex.getValue()%m_coefs.size()][j] < minValue && m_coefs[showFromIndex.getValue()%m_coefs.size()][j] != 0xFFF) minValue = m_coefs[showFromIndex.getValue()%m_coefs.size()][j];
            if ( m_coefs[showFromIndex.getValue()%m_coefs.size()][j] > maxValue && m_coefs[showFromIndex.getValue()%m_coefs.size()][j] != 0xFFF) maxValue = m_coefs[showFromIndex.getValue()%m_coefs.size()][j];
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
                    double color = (m_coefs[showFromIndex.getValue()%m_coefs.size()][tri[i][j]] - minValue) / (maxValue - minValue);
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
                double color = (m_coefs[showFromIndex.getValue()%m_coefs.size()][i] - minValue) / (maxValue - minValue);
                color = pow(color, showGammaCorrection.getValue());
                glColor3f( color, 0.0, 0.0);
                glVertex3f( xto[i][0], xto[i][1], xto[i][2]);
            }
            glEnd();
        }
    }
#endif
}

#ifdef SOFA_DEV
template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeDqL ( Mat86& L, const DUALQUAT& qi, const Coord& ti )
{
    L[0][0] = qi.q0[3];
    L[0][1] = qi.q0[2];
    L[0][2] = -qi.q0[1];
    L[1][0] = -qi.q0[2];
    L[1][1] = qi.q0[3];
    L[1][2] = qi.q0[0];
    L[2][0] = qi.q0[1];
    L[2][1] = -qi.q0[0];
    L[2][2] = qi.q0[3];
    L[3][0] = -qi.q0[0];
    L[3][1] = -qi.q0[1];
    L[3][2] = -qi.q0[2];

    L[4][0] = qi.qe[3]+qi.q0[2]*ti[2]+qi.q0[1]*ti[1];
    L[4][1] = qi.qe[2]-qi.q0[3]*ti[2]-qi.q0[1]*ti[0];
    L[4][2] = -qi.qe[1]+qi.q0[3]*ti[1]-qi.q0[2]*ti[0];
    L[5][0] = -qi.qe[2]+qi.q0[3]*ti[2]-qi.q0[0]*ti[1];
    L[5][1] = qi.qe[3]+qi.q0[2]*ti[2]+qi.q0[0]*ti[0];
    L[5][2] = qi.qe[0]-qi.q0[3]*ti[0]-qi.q0[2]*ti[1];
    L[6][0] = qi.qe[1]-qi.q0[3]*ti[1]-qi.q0[0]*ti[2];
    L[6][1] = -qi.qe[0]+qi.q0[3]*ti[0]-qi.q0[1]*ti[2];
    L[6][2] = qi.qe[3]+qi.q0[1]*ti[1]+qi.q0[0]*ti[0];
    L[7][0] = -qi.qe[0]-qi.q0[1]*ti[2]+qi.q0[2]*ti[1];
    L[7][1] = -qi.qe[1]+qi.q0[0]*ti[2]-qi.q0[2]*ti[0];
    L[7][2] = -qi.qe[2]-qi.q0[0]*ti[1]+qi.q0[1]*ti[0];

    int i,j;
    for ( i=0; i<8; i++ ) for ( j=0; j<3; j++ ) L[i][j] *= 0.5;
    for ( i=0; i<4; i++ ) for ( j=3; j<6; j++ ) L[i][j] = 0;
    for ( i=4; i<8; i++ ) for ( j=3; j<6; j++ ) L[i][j] = L[i-4][j-3];

}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::BlendDualQuat ( DUALQUAT& b, DUALQUAT& bn, double& QEQ0, double& Q0Q0, double& Q0, const int& indexp, const VDUALQUAT& qrel, const VVD& w )
{
    int i,j,nbDOF=qrel.size();
    for ( i=0; i<4; i++ )
    {
        b.q0[i]=0;
        b.qe[i]=0;
    }
    for ( j=0; j<nbDOF; j++ ) for ( i=0; i<4; i++ )
        {
            b.q0[i]+=w[j][indexp]*qrel[j].q0[i];
            b.qe[i]+=w[j][indexp]*qrel[j].qe[i];
        }

    Q0Q0 = b.q0 * b.q0;
    Q0= sqrt ( Q0Q0 );
    QEQ0 = b.q0 * b.qe;

    for ( i=0; i<4; i++ )
    {
        bn.q0[i]=b.q0[i]/Q0;
        bn.qe[i]=b.qe[i]/Q0;
    }
    for ( i=0; i<4; i++ ) bn.qe[i]-=QEQ0*bn.q0[i]/Q0Q0;
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeQrel ( DUALQUAT& qrel, const Mat88& T, const DUALQUAT& q )
{
    int k,m;
    for ( m=0; m<4; m++ )
    {
        qrel.q0[m]=0;
        qrel.qe[m]=0;
    }
    for ( k=0; k<4; k++ ) for ( m=0; m<4; m++ )
        {
            qrel.q0[k]+=T[k][m]*q.q0[m];
            qrel.qe[k]+=T[k+4][m]*q.q0[m];
        }
    for ( k=0; k<4; k++ ) for ( m=0; m<4; m++ ) qrel.qe[k]+=T[k+4][m+4]*q.qe[m];
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeDqRigid ( Mat33& R, Vec3& t, const DUALQUAT& bn )
{
    double as = bn.q0[0]*2., bs = bn.q0[1]*2., cs = bn.q0[2]*2.;
    double wa = bn.q0[3]*as, wb = bn.q0[3]*bs, wc = bn.q0[3]*cs;
    double aa = bn.q0[0]*as, ab = bn.q0[0]*bs, ac = bn.q0[0]*cs;
    double bb = bn.q0[1]*bs, bc = bn.q0[1]*cs, cc = bn.q0[2]*cs;

    R[0][0] = 1.0 - ( bb + cc );
    R[0][1]= ab - wc;
    R[0][2] = ac + wb;
    R[1][0] = ab + wc;
    R[1][1] = 1.0 - ( aa + cc );
    R[1][2] = bc - wa;
    R[2][0] = ac - wb;
    R[2][1] = bc + wa;
    R[2][2] = 1.0 - ( aa + bb );

    t[0]= 2* ( -bn.qe[3]*bn.q0[0]+bn.qe[0]*bn.q0[3]-bn.qe[1]*bn.q0[2]+bn.qe[2]*bn.q0[1] );
    t[1]= 2* ( -bn.qe[3]*bn.q0[1]+bn.qe[0]*bn.q0[2]+bn.qe[1]*bn.q0[3]-bn.qe[2]*bn.q0[0] );
    t[2]= 2* ( -bn.qe[3]*bn.q0[2]-bn.qe[0]*bn.q0[1]+bn.qe[1]*bn.q0[0]+bn.qe[2]*bn.q0[3] );
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeDqN_constants ( Mat44& q0q0T, Mat44& q0qeT,Mat44& qeq0T, const DUALQUAT& bn )
{
    getCov ( q0q0T,bn.q0,bn.q0 );
    getCov ( q0qeT,bn.q0,bn.qe );
    getCov ( qeq0T,bn.qe,bn.q0 );
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::XItoQ ( DUALQUAT& q, const InCoord& xi )
{
    // xi: quat(a,b,c,w),tx,ty,tz
    // qi: quat(a,b,c,w),1/2quat(t.q)
    const Quat& roti = xi.getOrientation();
    const Vec3& ti = xi.getCenter();

    q.q0[0]=roti[0];
    q.q0[1]=roti[1];
    q.q0[2]=roti[2];
    q.q0[3]=roti[3];
    q.qe[3]= ( - ( roti[0]*ti[0]+roti[1]*ti[1]+roti[2]*ti[2] ) ) /2.;
    q.qe[0]= ( ti[1]*roti[2]-roti[1]*ti[2]+roti[3]*ti[0] ) /2.;
    q.qe[1]= ( ti[2]*roti[0]-roti[2]*ti[0]+roti[3]*ti[1] ) /2.;
    q.qe[2]= ( ti[0]*roti[1]-roti[0]*ti[1]+roti[3]*ti[2] ) /2.;
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::getCov ( Mat44& q1q2T, const Vec4& q1, const Vec4& q2 )
{
    for ( int i=0; i<4; i++ ) for ( int j=0; j<4; j++ ) q1q2T[i][j] = q1[i]*q2[j];
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeDqN ( Mat88& N, const Mat44& q0q0T, const Mat44& q0qeT, const Mat44& qeq0T , const double& QEQ0, const double& Q0Q0, const double& Q0 )
{
    double c1=-1./Q0;
    double c2=-QEQ0/Q0Q0;

    int i,j;
    for ( i=0; i<4; i++ ) for ( j=4; j<8; j++ ) N[i][j]= 0;
    for ( i=0; i<4; i++ ) for ( j=0; j<4; j++ ) N[i][j]= c1*q0q0T[i][j] ;
    for ( i=0; i<4; i++ ) N[i][i]-=c1;
    for ( i=4; i<8; i++ ) for ( j=4; j<8; j++ ) N[i][j]= N[i-4][j-4];
    for ( i=0; i<4; i++ ) for ( j=0; j<4; j++ ) N[i+4][j]= c1* ( q0qeT[i][j]+qeq0T[i][j] ) + c2*N[i][j];
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeDqQ ( Mat38& Q, const DUALQUAT& bn, const Vec3& p )
{
    Q[0][0] = - bn.qe[3] + bn.q0[0]*p[0] + bn.q0[1]*p[1] + bn.q0[2]*p[2];
    Q[0][1] =   bn.qe[2] - bn.q0[1]*p[0] + bn.q0[0]*p[1] + bn.q0[3]*p[2];
    Q[0][2] = - bn.qe[1] - bn.q0[2]*p[0] - bn.q0[3]*p[1] + bn.q0[0]*p[2];
    Q[0][3] =   bn.qe[0] + bn.q0[3]*p[0] - bn.q0[2]*p[1] + bn.q0[1]*p[2];
    Q[1][0] = -Q[0][1];
    Q[1][1] = Q[0][0];
    Q[1][2] = Q[0][3];
    Q[1][3] = -Q[0][2];
    Q[2][0] = -Q[0][2];
    Q[2][1] = -Q[0][3];
    Q[2][2] = Q[0][0];
    Q[2][3] = Q[0][1];

    Q[0][4] =  bn.q0[3];
    Q[0][5] = -bn.q0[2];
    Q[0][6] =  bn.q0[1];
    Q[0][7] = -bn.q0[0];
    Q[1][4] = -Q[0][5];
    Q[1][5] = Q[0][4];
    Q[1][6] = Q[0][7];
    Q[1][7] = -Q[0][6];
    Q[2][4] = -Q[0][6];
    Q[2][5] = -Q[0][7];
    Q[2][6] = Q[0][4];
    Q[2][7] = Q[0][5];

    for ( int i=0; i<3; i++ ) for ( int j=0; j<8; j++ ) Q[i][j] *= 2.;
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeDqDR ( Mat33& DR, const DUALQUAT& bn, const DUALQUAT& V )
{
    if ( V.q0[0]==0 && V.q0[1]==0 && V.q0[2]==0 && V.q0[3]==0 )
    {
        for ( int i=0; i<3; i++ ) for ( int j=0; j<3; j++ ) DR[i][j]=0;
    }
    else
    {
        DR[0][0] = -2.* ( bn.q0[1]*V.q0[1]+bn.q0[2]*V.q0[2] );
        DR[0][1]= -bn.q0[2]*V.q0[3]+bn.q0[1]*V.q0[0]+bn.q0[0]*V.q0[1]-bn.q0[3]*V.q0[2];
        DR[0][2] = bn.q0[1]*V.q0[3]+bn.q0[2]*V.q0[0]+bn.q0[3]*V.q0[1]+bn.q0[0]*V.q0[2];
        DR[1][0] = bn.q0[2]*V.q0[3]+bn.q0[1]*V.q0[0]+bn.q0[0]*V.q0[1]+bn.q0[3]*V.q0[2];
        DR[1][1] = -2.* ( bn.q0[0]*V.q0[0]+bn.q0[2]*V.q0[2] );
        DR[1][2] = -bn.q0[0]*V.q0[3]-bn.q0[3]*V.q0[0]+bn.q0[2]*V.q0[1]+bn.q0[1]*V.q0[2];
        DR[2][0] = -bn.q0[1]*V.q0[3]+bn.q0[2]*V.q0[0]-bn.q0[3]*V.q0[1]+bn.q0[0]*V.q0[2];
        DR[2][1] = bn.q0[0]*V.q0[3]+bn.q0[3]*V.q0[0]+bn.q0[2]*V.q0[1]+bn.q0[1]*V.q0[2];
        DR[2][2] = -2.* ( bn.q0[0]*V.q0[0]+bn.q0[1]*V.q0[1] );
        for ( int i=0; i<3; i++ ) for ( int j=0; j<3; j++ ) DR[i][j] *= 2.;
    }
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeDqDQ ( Mat38& DQ, const Vec3& p, const DUALQUAT& V )
{
    DQ[0][0] =  - V.qe[3] + V.q0[0]*p[0] + V.q0[1]*p[1] + V.q0[2]*p[2];
    DQ[0][1] =   V.qe[2] - V.q0[1]*p[0] + V.q0[0]*p[1] + V.q0[3]*p[2];
    DQ[0][2] = - V.qe[1] - V.q0[2]*p[0] - V.q0[3]*p[1] + V.q0[0]*p[2];
    DQ[0][3] =  V.qe[0] + V.q0[3]*p[0] - V.q0[2]*p[1] + V.q0[1]*p[2];
    DQ[1][0] = -DQ[0][1];
    DQ[1][1] = DQ[0][0];
    DQ[1][2] = DQ[0][3];
    DQ[1][3] = -DQ[0][2];
    DQ[2][0] = -DQ[0][2];
    DQ[2][1] = -DQ[0][3];
    DQ[2][2] = DQ[0][0];
    DQ[2][3] = DQ[0][1];

    DQ[0][4] =  V.q0[3];
    DQ[0][5] = -V.q0[2];
    DQ[0][6] =  V.q0[1];
    DQ[0][7] = -V.q0[0];
    DQ[1][4] = -DQ[0][5];
    DQ[1][5] = DQ[0][4];
    DQ[1][6] = DQ[0][7];
    DQ[1][7] = -DQ[0][6];
    DQ[2][4] = -DQ[0][6];
    DQ[2][5] = -DQ[0][7];
    DQ[2][6] = DQ[0][4];
    DQ[2][7] = DQ[0][5];

    for ( int i=0; i<3; i++ ) for ( int j=0; j<8; j++ ) DQ[i][j] *= 2.;
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeDqDN_constants ( Mat44& q0V0T, Mat44& V0q0T, double& q0V0, Mat44& q0VeT, Mat44& Veq0T,double& q0Ve, Mat44& qeV0T, Mat44& V0qeT, double& qeV0, const DUALQUAT& bn, const DUALQUAT& V )
{
    getCov ( q0V0T,bn.q0,V.q0 );
    getCov ( V0q0T,V.q0,bn.q0 );
    q0V0 = q0V0T[0][0]+q0V0T[1][1]+q0V0T[2][2]+q0V0T[3][3];
    getCov ( q0VeT,bn.q0,V.qe );
    getCov ( Veq0T,V.qe,bn.q0 );
    q0Ve = q0VeT[0][0]+q0VeT[1][1]+q0VeT[2][2]+q0VeT[3][3];
    getCov ( qeV0T,bn.qe,V.q0 );
    getCov ( V0qeT,V.q0,bn.qe );
    qeV0 = qeV0T[0][0]+qeV0T[1][1]+qeV0T[2][2]+qeV0T[3][3];
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeDqDN ( Mat88& DN, const Mat44& q0q0T, const Mat44& q0qeT, const Mat44& qeq0T, const double& QEQ0, const double& Q0Q0, const double& /*Q0*/, const Mat44& q0V0T, const Mat44& V0q0T, const double& q0V0, const Mat44& q0VeT, const Mat44& Veq0T, const double& q0Ve, const Mat44& qeV0T, const Mat44& V0qeT, const double& qeV0, const DUALQUAT& V )
{
    double c1=-1./Q0Q0;
    double c2=-2*QEQ0/Q0Q0;

    int i,j;
    if ( V.q0[0]==0 && V.q0[1]==0 && V.q0[2]==0 && V.q0[3]==0 )
    {
        for ( i=0; i<8; i++ ) for ( j=0; j<8; j++ ) DN[i][j]= 0;
        for ( i=0; i<4; i++ ) for ( j=0; j<4; j++ ) DN[i+4][j]= c1* ( q0VeT[i][j]+Veq0T[i][j] - 3*q0Ve*q0q0T[i][j] );
        for ( i=0; i<4; i++ ) DN[i+4][i]+=c1*q0Ve;
    }
    else
    {
        for ( i=0; i<4; i++ ) for ( j=4; j<8; j++ ) DN[i][j]= 0;
        for ( i=0; i<4; i++ ) for ( j=0; j<4; j++ ) DN[i][j]= c1* ( q0V0T[i][j]+V0q0T[i][j] - 3*q0V0*q0q0T[i][j] );
        for ( i=0; i<4; i++ ) DN[i][i]+=c1*q0V0;
        for ( i=4; i<8; i++ ) for ( j=4; j<8; j++ ) DN[i][j]= DN[i-4][j-4];

        for ( i=0; i<4; i++ ) for ( j=0; j<4; j++ ) DN[i+4][j]= c1* ( q0VeT[i][j]+Veq0T[i][j] - 3*q0Ve*q0q0T[i][j] + qeV0T[i][j]+V0qeT[i][j] - 3*qeV0*q0q0T[i][j] -3*q0V0* ( qeq0T[i][j]+q0qeT[i][j] ) ) + c2*DN[i][j];
        for ( i=0; i<4; i++ ) DN[i+4][i]+=c1* ( qeV0+q0Ve );
    }
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeDqT ( Mat88& T, const DUALQUAT& qi0 )
{
    T[0][0] = qi0.q0[3];
    T[0][1] = -qi0.q0[2];
    T[0][2] = qi0.q0[1];
    T[0][3] = -qi0.q0[0];
    T[1][0] = qi0.q0[2];
    T[1][1] = qi0.q0[3];
    T[1][2] = -qi0.q0[0];
    T[1][3] = -qi0.q0[1];
    T[2][0] = -qi0.q0[1];
    T[2][1] = qi0.q0[0];
    T[2][2] = qi0.q0[3];
    T[2][3] = -qi0.q0[2];
    T[3][0] = qi0.q0[0];
    T[3][1] = qi0.q0[1];
    T[3][2] = qi0.q0[2];
    T[3][3] = qi0.q0[3];

    T[4][0] = qi0.qe[3];
    T[4][1] = -qi0.qe[2];
    T[4][2] = qi0.qe[1];
    T[4][3] = -qi0.qe[0];
    T[5][0] = qi0.qe[2];
    T[5][1] = qi0.qe[3];
    T[5][2] = -qi0.qe[0];
    T[5][3] = -qi0.qe[1];
    T[6][0] = -qi0.qe[1];
    T[6][1] = qi0.qe[0];
    T[6][2] = qi0.qe[3];
    T[6][3] = -qi0.qe[2];
    T[7][0] = qi0.qe[0];
    T[7][1] = qi0.qe[1];
    T[7][2] = qi0.qe[2];
    T[7][3] = qi0.qe[3];

    int i,j;
    for ( i=0; i<4; i++ ) for ( j=4; j<8; j++ ) T[i][j] = 0;
    for ( i=4; i<8; i++ ) for ( j=4; j<8; j++ ) T[i][j] = T[i-4][j-4];
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::removeFrame( const unsigned int /*index*/)
{
    //VecCoord& xto0 = *this->toModel->getX0();
    VecCoord& xto = *this->toModel->getX();
    //VecInCoord& xfrom0 = *this->fromModel->getX0();
    VecInCoord& xfrom = *this->fromModel->getX();

    // this->T.erase( T.begin()+index);
    // coeffs

    // Recompute matrices
    apply( xto, xfrom);
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::insertFrame( const Coord& pos, const Quat& rot, GeoVecCoord beginPointSet, double distMax)
{
    changeSettingsDueToInsertion();

    if (!this->toModel->getX0()) return;
    // Get references
    VecCoord& xto = *this->toModel->getX();
    VecInCoord& xfrom0 = *this->fromModel->getX0();
    VecInCoord& xfrom = *this->fromModel->getX();
    unsigned int indexFrom = xfrom.size();
    MechanicalState<StdRigidTypes<N, Real> >* mstateFrom = dynamic_cast<MechanicalState<StdRigidTypes<N, Real> >* >( this->fromModel);
    if ( !mstateFrom)
    {
        serr << "Error: try to insert a new frame on fromModel, which is not a mechanical state !" << sendl;
        return;
    }

    // Compute the rest position of the frame.
    InCoord newX, newX0;
    InCoord targetDOF( pos, rot);
    inverseSkinning( newX0, newX, targetDOF);

    // Insert a new DOF
    this->fromModel->resize( indexFrom + 1);
    xfrom[indexFrom] = newX;
    xfrom0[indexFrom] = newX0;
    (*mstateFrom->getXReset())[indexFrom] = newX0;

    if ( distMax == 0.0)
        distMax = newFrameDefaultCutOffDistance.getValue();

    // Compute geodesical/euclidian distance for this frame.
    if ( distanceType.getValue().getSelectedId() == DISTANCE_GEODESIC || distanceType.getValue().getSelectedId() == DISTANCE_HARMONIC)
        geoDist->addElt( newX0.getCenter(), beginPointSet, distMax);
    vector<double>& vRadius = (*newFrameWeightingRadius.beginEdit());
    vRadius.resize( indexFrom + 1);
    vRadius[indexFrom] = distMax;
    newFrameWeightingRadius.endEdit();

    // Recompute matrices
    apply( xto, xfrom);
}


template <class BasicMapping>
void SkinningMapping<BasicMapping>::Multi_Q(Quat& q, const Vec4& q1, const Quat& q2)
{
    Vec3 qv1;
    qv1[0]=q1[0];
    qv1[1]=q1[1];
    qv1[2]=q1[2];
    Vec3 qv2;
    qv2[0]=q2[0];
    qv2[1]=q2[1];
    qv2[2]=q2[2];
    q[3]=q1[3]*q2[3]-qv1*qv2;
    Vec3 cr=cross(qv1,qv2);
    q[0]=cr[0];
    q[1]=cr[1];
    q[2]=cr[2];
    q[0]+=q2[3]*q1[0]+q1[3]*q2[0];
    q[1]+=q2[3]*q1[1]+q1[3]*q2[1];
    q[2]+=q2[3]*q1[2]+q1[3]*q2[2];
}

template <class BasicMapping>
bool SkinningMapping<BasicMapping>::inverseSkinning( InCoord& X0, InCoord& X, const InCoord& Xtarget)
// compute position in reference configuration from deformed position
{
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
    for (i=0; i<nbDOF; i++)
    {
        XItoQ( q, xi[i]); //update DOF quats
        computeQrel( qrel[i], T[i], q); //update qrel=Tq
    }

// get closest material point
    for (i=0; i<nbP; i++)
    {
        t = Xtarget.getCenter() - P[i];
        d = t * t;
        if (d<dmin)
        {
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
        for (k=0; k<3; k++)
        {
            X.getCenter()[k] = t[k];
            for (j=0; j<3; j++) X.getCenter()[k] += R[k][j] * X0.getCenter()[j];
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
            for (j=0; j<nbDOF; j++) for (k=0; k<4; k++) for (l=0; l<3; l++)
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

    return true;
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeWeight( VVD& w, VecVecCoord& dw, const Coord& x0)
{
    VecCoord& xto0 = *this->toModel->getX0();
    VecInCoord& xfrom0 = *this->fromModel->getX0();
    // Get Distances
    VVD dist;
    GeoVecVecCoord ddist;

    switch ( distanceType.getValue().getSelectedId()  )
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
    {
        GeoVecCoord goals;
        goals.push_back( x0);
        geoDist->getDistances ( dist, ddist, goals );
        break;
    }
    default: {}
    }

    // Compute Weights
    switch ( wheightingType.getValue().getSelectedId()  )
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

template <class BasicMapping>
void SkinningMapping<BasicMapping>::updateDataAfterInsertion()
{
    const VecCoord& xto = *this->toModel->getX();
    const VecInCoord& xfrom0 = *this->fromModel->getX0();
    const VecInCoord& xfrom = *this->fromModel->getX();
    VVD& m_coefs = * ( coefs.beginEdit() );
    SVector<SVector<GeoCoord> >& dw = *(weightGradients.beginEdit());
    vector<double>& radius = (*newFrameWeightingRadius.beginEdit());

    changeSettingsDueToInsertion();

    //TODO fix it ! Synchro between many SMapping
    if ( radius.size() != xfrom.size())
    {
        int oldSize = radius.size();
        radius.resize( xfrom.size());
        for ( unsigned int i = oldSize; i < xfrom.size(); ++i) radius[i] = newFrameDefaultCutOffDistance.getValue();
    }

    // Resize T
    unsigned int size = this->T.size();
    DUALQUAT qi0;
    this->T.resize ( xfrom.size() );

    // Get distances
    getDistances( size);

    // for each new frame
    const double& maximizeWeightDist = newFrameDistanceToMaximizeWeight.getValue();
    for ( unsigned int i = size; i < xfrom.size(); ++i )
    {
        // Get T
        XItoQ( qi0, xfrom0[i]);
        computeDqT ( this->T[i], qi0 );

        // Update weights
        m_coefs.resize ( xfrom.size() );
        m_coefs[i].resize ( xto.size() );
        dw.resize ( xfrom.size() );
        dw[i].resize ( xto.size() );
        for ( unsigned int j = 0; j < xto.size(); ++j )
        {
            if ( distances[i][j])
            {
                if ( distances[i][j] == -1.0)
                {
                    m_coefs[i][j] = 0.0;
                    dw[i][j] = Coord();
                }
                else
                {
                    if( maximizeWeightDist != 0.0)
                    {
                        if( distances[i][j] < maximizeWeightDist)
                        {
                            m_coefs[i][j] = 0xFFF;
                            dw[i][j] = Coord();
                        }
                        else if( distances[i][j] > radius[i])
                        {
                            m_coefs[i][j] = 0.0;
                            dw[i][j] = Coord();
                        }
                        else
                        {
                            // linear interpolation from 0 to 0xFFF
                            //m_coefs[i][j] = 0xFFF / (maximizeWeightDist - radius[i]) * (distances[i][j] - radius[i]);
                            //dw[i][j] = distGradients[i][j] * 0xFFF / (maximizeWeightDist - radius[i]);

                            // Hermite between 0 and 0xFFF
                            double dPrime = (distances[i][j] - maximizeWeightDist) / (radius[i]-maximizeWeightDist);
                            m_coefs[i][j] = (1-3*dPrime*dPrime+2*dPrime*dPrime*dPrime) * 0xFFF;
                            dw[i][j] = - distGradients[i][j] * (6*dPrime-6*dPrime*dPrime) / (radius[i]-maximizeWeightDist ) * 0xFFF;
                        }
                    }
                    else
                    {
                        m_coefs[i][j] = 1.0 / (distances[i][j]*distances[i][j]) - 1.0 / (radius[i]*radius[i]);
                        if (m_coefs[i][j] < 0)
                        {
                            m_coefs[i][j] = 0.0;
                            dw[i][j] = Coord();
                        }
                        else
                            dw[i][j] = - distGradients[i][j] / (distances[i][j]*distances[i][j]*distances[i][j]) * 2.0;
                    }
                }
            }
            else
            {
                m_coefs[i][j] = 0xFFF;
                dw[i][j] = Coord();
            }
        }
    }
    coefs.endEdit();
    weightGradients.endEdit();
    newFrameWeightingRadius.endEdit();
}


template <class BasicMapping>
void SkinningMapping<BasicMapping>::changeSettingsDueToInsertion()
{
    setWeightsToInvDist();
    if( geoDist)
    {
        distanceType.beginEdit()->setSelectedItem(DISTANCE_GEODESIC);
    }
    else
    {
        distanceType.beginEdit()->setSelectedItem(DISTANCE_EUCLIDIAN);
    }
}


template <class BasicMapping>
void SkinningMapping<BasicMapping>::apply0()
{
    const VecCoord& p0 = (this->toModel->getX0())?*this->toModel->getX0():*this->toModel->getX();
    const VecInCoord& in = *this->fromModel->getX0();

    unsigned int i,j,k,l,m,n, nbDOF=in.size(), nbP=p0.size();

    // allocate temporary variables
    VMat86 L ( nbDOF );
    VMat86 TL ( nbDOF );

    DUALQUAT q,bn,V;
    Mat33 R,dR,Ainv,E,dE;
    Mat88 N,dN;
    Mat86 NTL;
    Mat38 Q,dQ,QN,QNT;
    Mat83 W,NW,dNW;
    Vec3 t;
    double QEQ0=0,Q0Q0,Q0,q0V0,q0Ve,qeV0;
    Mat44 q0q0T,q0qeT,qeq0T,q0V0T,V0q0T,q0VeT,Veq0T,qeV0T,V0qeT;
    const VVD& w = coefs.getValue();
    const SVector<SVector<GeoCoord> >& dw = weightGradients.getValue();
    VVec6& e = this->deformationTensors;

    // Resize vectors
    e.resize ( nbP );
    this->det.resize ( nbP );
    this->J0.resize ( nbDOF );
    this->B.resize ( nbDOF );
    this->A.resize( nbP);
    this->ddet.resize( nbDOF);
    for ( i=0; i<nbDOF; i++ )
    {
        this->J0[i].resize ( nbP );
        this->B[i].resize ( nbP );
        this->ddet[i].resize( nbP);
    }


    if ( this->T.size() != nbDOF) updateDataAfterInsertion();

    for (unsigned int i = 0; i < 3; ++i) R[i][i] = 1.0;
    bn.q0[3]=1;
    q0q0T[3][3]=1;

    //apply
    for ( i=0; i<nbDOF; i++ )
    {
        XItoQ ( q,in[i] );		//update DOF quats
        computeDqL ( L[i],q,in[i].getCenter() );	//update L=d(q)/Omega
        TL[i]=this->T[i]*L[i];	//update TL
    }

    for ( i=0; i<nbP; i++ )
    {
        Q0=0;
        for ( k=0; k<nbDOF; k++ ) Q0+=w[k][i];
        Q0Q0=Q0*Q0;

        if ( !(this->computeJ.getValue() || this->computeAllMatrices.getValue())) continue; // if we don't want to compute all the matrices

        computeDqN ( N, q0q0T, q0qeT, qeq0T, QEQ0, Q0Q0, Q0 );	//update N=d(bn)/d(b)
        computeDqQ ( Q, bn, initPos[i] );	//update Q=d(P)/d(bn)
        W.fill ( 0 );
        for ( j=0; j<nbDOF; j++ )  for ( l=0; l<3; l++ )
            {
                W[3][l]+=dw[j][i][l];  //update W=sum(wrel.dw)=d(b)/d(p)
                W[3+4][l]+=dw[j][i][l];
            }
        NW=N*W; // update NW
        QN=Q*N; // update QN

        // grad def = d(P)/d(p0)
        this->A[i]=Ainv=R;
        this->det[i]=1;

        for ( j=0; j<nbDOF; j++ )
        {
            // J = d(P_i)/Omega_j
            NTL=N*TL[j]; // update NTL
            this->J0[j][i]=Q*NTL;
            this->J0[j][i]*=w[j][i]; // Update J=wiQNTL

            if ( !this->computeAllMatrices.getValue()) continue; // if we want to compute just the J matrix

            QNT=QN*this->T[j]; // Update QNT

            // B = d(strain)/Omega_j
            this->B[j][i].fill ( 0 );
            for ( l=0; l<6; l++ )
            {
                if ( w[j][i]==0 ) dE.fill(0);
                else
                {
                    // d(grad def)/Omega_j
                    for ( k=0; k<4; k++ )
                    {
                        V.q0[k]=NTL[k][l];
                        V.qe[k]=NTL[k+4][l];
                    }
                    computeDqDR ( dR, bn, V );
                    dE=dR;
                    computeDqDQ ( dQ, initPos[i], V );
                    dE+=dQ*NW;
                    for ( k=0; k<4; k++ )
                    {
                        V.q0[k]=TL[j][k][l];
                        V.qe[k]=TL[j][k+4][l];
                    }
                    computeDqDN_constants ( q0V0T, V0q0T, q0V0, q0VeT, Veq0T, q0Ve, qeV0T, V0qeT, qeV0, bn, V );
                    computeDqDN ( dN, q0q0T, q0qeT, qeq0T, QEQ0, Q0Q0, Q0, q0V0T, V0q0T, q0V0, q0VeT, Veq0T, q0Ve, qeV0T, V0qeT, qeV0, V );
                    dNW=dN*W;
                    dE+=Q*dNW;
                    dE*=w[j][i];
                }
                for ( k=0; k<3; k++ )  for ( m=0; m<3; m++ ) for ( n=0; n<8; n++ ) dE[k][m]+=QNT[k][n]*L[j][n][l]*dw[j][i][m];

                // determinant derivatives
                this->ddet[j][i][l]=0;
                for (k=0; k<3; k++)  for (m=0; m<3; m++) this->ddet[j][i][l]+=dE[k][m]*Ainv[m][k];
                this->ddet[j][i][l]*=this->det[i];

                // B=1/2(graddef^T.d(graddef)/Omega_j + d(graddef)/Omega_j^T.graddef)
                for ( n=0; n<3; n++ )
                {
                    this->B[j][i][0][l]+= ( dE[n][0]*this->A[i][n][0] );
                    this->B[j][i][1][l]+= ( dE[n][1]*this->A[i][n][1] );
                    this->B[j][i][2][l]+= ( dE[n][2]*this->A[i][n][2] );
                    this->B[j][i][3][l]+=0.5* ( dE[n][0]*this->A[i][n][1]+this->A[i][n][0]*dE[n][1] );
                    this->B[j][i][4][l]+=0.5* ( dE[n][0]*this->A[i][n][2]+this->A[i][n][0]*dE[n][2] );
                    this->B[j][i][5][l]+=0.5* ( dE[n][1]*this->A[i][n][2]+this->A[i][n][1]*dE[n][2] );
                }
            }
        }
    }
}
//*/
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
