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
    , nbRefs ( initData ( &nbRefs, ( unsigned ) 3,"nbRefs","Number of primitives influencing each point." ) )
    , repartition ( initData ( &repartition,"repartition","Repartition between input DOFs and skinned vertices." ) )
    , weights ( initData ( &weights,"weights","weights list for the influences of the references Dofs" ) )
    , weightGradients ( initData ( &weightGradients,"weightGradients","weight gradients list for the influences of the references Dofs" ) )
    , showBlendedFrame ( initData ( &showBlendedFrame, false, "showBlendedFrame","weights list for the influences of the references Dofs" ) )
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
    this->getDistances( 0);
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::getDistances( int xfromBegin)
{
    const VecCoord& xto0 = ( this->toModel->getX0()->size() == 0)?*this->toModel->getX():*this->toModel->getX0();
    const VecInCoord& xfrom0 = *this->fromModel->getX0();

    switch ( distanceType.getValue().getSelectedId() )
    {
    case SM_DISTANCE_EUCLIDIAN:
    {
        const unsigned int& toSize = xto0.size();
        const unsigned int& fromSize = xfrom0.size();
        distances.resize( fromSize);
        distGradients.resize( fromSize);
        for ( unsigned int i = xfromBegin; i < fromSize; ++i ) // for each new frame
        {
            distances[i].resize (toSize);
            distGradients[i].resize (toSize);
            for ( unsigned int j=0; j<toSize; ++j )
            {
                distGradients[i][j] = xto0[j] - xfrom0[i].getCenter();
                distances[i][j] = distGradients[i][j].norm();
                distGradients[i][j].normalize();
            }
        }
        break;
    }
#ifdef SOFA_DEV
    case SM_DISTANCE_GEODESIC:
    case SM_DISTANCE_HARMONIC:
    case SM_DISTANCE_STIFFNESS_DIFFUSION:
    case SM_DISTANCE_HARMONIC_STIFFNESS:
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
    const unsigned int& toSize = this->toModel->getX()->size();
    const unsigned int& fromSize = this->fromModel->getX0()->size();
    const unsigned int& nbRef = this->nbRefs.getValue();

    references.clear();
    references.resize (nbRef*toSize);
    for ( unsigned int i=0; i< nbRef *toSize; i++ )
        references[i] = -1;

    for ( unsigned int i=0; i<fromSize; i++ )
        for ( unsigned int j=0; j<toSize; j++ )
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
    const unsigned int& xtoSize = this->toModel->getX()->size();
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
    if ( distanceType.getValue().getSelectedId() != SM_DISTANCE_EUCLIDIAN)
    {
        this->getContext()->get ( distOnGrid, core::objectmodel::BaseContext::SearchRoot );
        if ( !distOnGrid )
        {
            serr << "Can not find the DistanceOnGrid component: distances used are euclidian." << sendl;
            distanceType.setValue( SM_DISTANCE_EUCLIDIAN);
        }
    }
#else
    distanceType.beginEdit()->setSelectedItem(SM_DISTANCE_EUCLIDIAN);
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
                if ( distanceType.getValue().getSelectedId()  == SM_DISTANCE_HARMONIC)
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
    }
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
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

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
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


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
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
#endif
}



template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::getLocalCoord( Coord& result, const typename In::Coord& inCoord, const Coord& coord) const
{
    result = inCoord.pointToChild ( coord );
}


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


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
