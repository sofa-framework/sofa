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
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <string>
#include <iostream>

#ifdef SOFA_DEV
#include <sofa/helper/DualQuat.h>
#endif




namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

template <class BasicMapping>
class SkinningMapping<BasicMapping>::Loader : public helper::io::MassSpringLoader, public helper::io::SphereLoader
{
public:
    SkinningMapping<BasicMapping>* dest;
    Loader ( SkinningMapping<BasicMapping>* dest ) : dest ( dest ) {}
    virtual void addMass ( SReal /*px*/, SReal /*py*/, SReal /*pz*/, SReal, SReal, SReal, SReal, SReal, bool, bool )
    {
    }
    virtual void addSphere ( SReal /*px*/, SReal /*py*/, SReal /*pz*/, SReal )
    {
    }
};

template <class BasicMapping>
SkinningMapping<BasicMapping>::SkinningMapping ( In* from, Out* to )
    : Inherit ( from, to )
    , repartition ( initData ( &repartition,"repartition","repartition between input DOFs and skinned vertices" ) )
    , coefs ( initData ( &coefs,"coefs","weights list for the influences of the references Dofs" ) )
    , nbRefs ( initData ( &nbRefs, ( unsigned ) 3,"nbRefs","nb references for skinning" ) )
    , computeWeights ( true )
    , wheighting ( WEIGHT_INVDIST )
    , interpolation ( INTERPOLATION_LINEAR )
{
    maskFrom = NULL;
    if (core::componentmodel::behavior::BaseMechanicalState *stateFrom = dynamic_cast< core::componentmodel::behavior::BaseMechanicalState *>(from))
        maskFrom = &stateFrom->forceMask;
    maskTo = NULL;
    if (core::componentmodel::behavior::BaseMechanicalState *stateTo = dynamic_cast< core::componentmodel::behavior::BaseMechanicalState *>(to))
        maskTo = &stateTo->forceMask;

}

template <class BasicMapping>
SkinningMapping<BasicMapping>::~SkinningMapping ()
{
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::load ( const char * /*filename*/ )
{
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::computeInitPos ( )
{
    VecCoord& xto = *this->toModel->getX();
    const VecInCoord& xfrom = *this->fromModel->getX();
    initPosDOFs.resize ( xfrom.size() );
    initPos.resize ( xto.size() * nbRefs.getValue() );

    sofa::helper::vector<double> m_coefs = coefs.getValue();
    sofa::helper::vector<unsigned int> m_reps = repartition.getValue();

    for ( unsigned int i = 0; i < xfrom.size(); i++ )
    {
        initPosDOFs[i] = xfrom[i];
    }

    switch ( interpolation )
    {
    case INTERPOLATION_LINEAR:
    {
        for ( unsigned int i = 0; i < xto.size(); i++ )
        {
            for ( unsigned int m = 0; m < nbRefs.getValue(); m++ )
            {
                initPos[nbRefs.getValue() *i+m] = xfrom[m_reps[nbRefs.getValue() *i+m]].getOrientation().inverseRotate ( xto[i] - xfrom[m_reps[nbRefs.getValue() *i+m]].getCenter() );
            }
        }
        break;
    }
#ifdef SOFA_DEV
    case INTERPOLATION_DUAL_QUATERNION:
    {
        initBlendedPos.resize ( xto.size() );
        for ( unsigned int i = 0; i < xto.size(); i++ )
        {
            DualQuat dq;
            for ( unsigned int m = 0 ; m < nbRefs.getValue(); m++ )
            {
                DualQuat dqi ( -initPosDOFs[m_reps[nbRefs.getValue() *i+m]].getCenter(),
                        initPosDOFs[m_reps[nbRefs.getValue() *i+m]].getOrientation().inverse() );

                // Blend all the transformations
                dq += dqi * m_coefs[nbRefs.getValue() *i+m];
                initPos[ nbRefs.getValue() *i+m] = dqi.transform ( xto[i] );
            }
            dq.normalize();
            initBlendedPos[i] = dq.transform ( xto[i] );
        }
        break;
    }
#endif
    default: {}
    }
    repartition.setValue ( m_reps );
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::sortReferences ()
{
    Coord posTo;
    VecCoord& xto = *this->toModel->getX();
    VecInCoord& xfrom = *this->fromModel->getX();

    sofa::helper::vector<unsigned int> m_reps = repartition.getValue();
    m_reps.clear();
    m_reps.resize ( nbRefs.getValue() *xto.size() );

    sofa::helper::vector<double> m_coefs = coefs.getValue();
    m_coefs.clear();
    m_coefs.resize ( nbRefs.getValue() *xto.size() );

    for ( unsigned int i=0; i<xto.size(); i++ )
    {
        posTo = xto[i];
        for ( unsigned int h=0 ; h<nbRefs.getValue() ; h++ )
            m_coefs[nbRefs.getValue() *i+h] = 9999999. ;

        //search the nbRefs nearest "from" dofs of each "to" point
        for ( unsigned int j=0; j<xfrom.size(); j++ )
        {
            Real dist2 = ( posTo - xfrom[j].getCenter() ).norm();

            unsigned int k=0;
            while ( k<nbRefs.getValue() )
            {
                if ( dist2 < m_coefs[nbRefs.getValue() *i+k] )
                {
                    for ( unsigned int m=nbRefs.getValue()-1 ; m>k ; m-- )
                    {
                        m_coefs[nbRefs.getValue() *i+m] = m_coefs[nbRefs.getValue() *i+m-1];
                        m_reps[nbRefs.getValue() *i+m] = m_reps[nbRefs.getValue() *i+m-1];
                    }
                    m_coefs[nbRefs.getValue() *i+k] = dist2;
                    m_reps[nbRefs.getValue() *i+k] = j;
                    k=nbRefs.getValue();
                }
                k++;
            }
        }
    }
    repartition.setValue ( m_reps );
    coefs.setValue ( m_coefs );
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::init()
{
    if ( this->initPos.empty() && this->toModel!=NULL && computeWeights==true && coefs.getValue().size() ==0 )
    {
        sortReferences ();
        updateWeights ();
        computeInitPos ();
    }
    else if ( computeWeights == false || coefs.getValue().size() !=0 )
    {
        computeInitPos();
    }
    this->BasicMapping::init();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::parse ( core::objectmodel::BaseObjectDescription* arg )
{
    if ( arg->getAttribute ( "filename" ) )
        this->load ( arg->getAttribute ( "filename" ) );
    this->Inherit::parse ( arg );
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::clear()
{
    this->initPos.clear();
    this->initPosDOFs.clear();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setWeightsToHermite()
{
    wheighting = WEIGHT_HERMITE;
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setWeightsToLinear()
{
    wheighting = WEIGHT_LINEAR;
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setWieghtsToInvDist()
{
    wheighting = WEIGHT_INVDIST;
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setInterpolationToLinear()
{
    interpolation = INTERPOLATION_LINEAR;
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setInterpolationToDualQuaternion()
{
    interpolation = INTERPOLATION_DUAL_QUATERNION;
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::updateWeights ()
{
    VecCoord& xto = *this->toModel->getX();
    VecInCoord& xfrom = *this->fromModel->getX();

    sofa::helper::vector<double> m_coefs = coefs.getValue();
    sofa::helper::vector<unsigned int> m_reps = repartition.getValue();

    switch ( wheighting )
    {
    case WEIGHT_LINEAR:
    {
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            Vec3d r1r2, r1p;
            if ( nbRefs.getValue() == 1 )
            {
                m_coefs[nbRefs.getValue() *i] = 1;
            }
            else
            {
                double wi;
                r1r2 = xfrom[m_reps[nbRefs.getValue() *i+1]].getCenter() - xfrom[m_reps[nbRefs.getValue() *i+0]].getCenter();
                r1p  = xto[i] - xfrom[m_reps[nbRefs.getValue() *i+0]].getCenter();
                wi = ( r1r2*r1p ) / ( r1r2.norm() *r1r2.norm() );

                // Abscisse curviligne
                m_coefs[nbRefs.getValue() *i+0] = ( 1 - wi );
                m_coefs[nbRefs.getValue() *i+1] = wi;
            }
        }
        break;
    }
    case WEIGHT_INVDIST:
    {
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            for ( unsigned int k=0; k<nbRefs.getValue(); k++ )
            {
                m_coefs[nbRefs.getValue() *i+k] = 1 / m_coefs[nbRefs.getValue() *i+k];
            }
            //m_coefs.normalize();
            //normalize the coefs vector such as the sum is equal to 1
            double norm=0.0;
            for ( unsigned int h=0 ; h<nbRefs.getValue(); h++ )
                norm += m_coefs[nbRefs.getValue() *i+h]*m_coefs[nbRefs.getValue() *i+h];
            norm = helper::rsqrt ( norm );

            for ( unsigned int g=0 ; g<nbRefs.getValue(); g++ )
                m_coefs[nbRefs.getValue() *i+g] /= norm;

            for ( unsigned int m=0; m<nbRefs.getValue(); m++ )
                m_coefs[nbRefs.getValue() *i+m] = m_coefs[nbRefs.getValue() *i+m]*m_coefs[nbRefs.getValue() *i+m];
        }
        break;
    }
    case WEIGHT_HERMITE:
    {
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            Vec3d r1r2, r1p;
            if ( nbRefs.getValue() == 1 )
            {
                m_coefs[nbRefs.getValue() *i] = 1;
            }
            else
            {
                double wi;
                r1r2 = xfrom[m_reps[nbRefs.getValue() *i+1]].getCenter() - xfrom[m_reps[nbRefs.getValue() *i+0]].getCenter();
                r1p  = xto[i] - xfrom[m_reps[nbRefs.getValue() *i+0]].getCenter();
                wi = ( r1r2*r1p ) / ( r1r2.norm() *r1r2.norm() );

                // Fonctions d'Hermite
                m_coefs[nbRefs.getValue() *i+0] = 1-3*wi*wi+2*wi*wi*wi;
                m_coefs[nbRefs.getValue() *i+1] = 3*wi*wi-2*wi*wi*wi;
            }
        }
        break;
    }
    default: {}
    }
    coefs.setValue ( m_coefs );
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setWeightCoefs ( sofa::helper::vector<double> &weights )
{
    sofa::helper::vector<double> * m_coefs = coefs.beginEdit();
    m_coefs->clear();
    m_coefs->insert ( m_coefs->begin(), weights.begin(), weights.end() );
    coefs.endEdit();
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::setRepartition ( sofa::helper::vector<unsigned int> &rep )
{
    sofa::helper::vector<unsigned int> * m_reps = repartition.beginEdit();
    m_reps->clear();
    m_reps->insert ( m_reps->begin(), rep.begin(), rep.end() );;
    repartition.endEdit();
}


template <class BasicMapping>
void SkinningMapping<BasicMapping>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const sofa::helper::vector<unsigned int>& m_reps = repartition.getValue();
    const sofa::helper::vector<double>& m_coefs = coefs.getValue();
    out.resize ( initPos.size() / nbRefs.getValue() );
    rotatedPoints.resize ( initPos.size() );

    switch ( interpolation )
    {
    case INTERPOLATION_LINEAR:
    {
        for ( unsigned int i=0 ; i<out.size(); i++ )
        {
            out[i] = Coord();
            for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
            {

                const int idx=nbRefs.getValue() *i+m;
                const int idxReps=m_reps[idx];

                // Save rotated points for applyJ/JT
                rotatedPoints[idx] = in[idxReps].getOrientation().rotate ( initPos[idx] );

                // And add each reference frames contributions to the new position out[i]
                out[i] += ( in[idxReps ].getCenter() + rotatedPoints[idx] ) * m_coefs[idx];
            }
        }
        break;
    }
#ifdef SOFA_DEV
    case INTERPOLATION_DUAL_QUATERNION:
    {
        for ( unsigned int i=0 ; i<out.size(); i++ )
        {
            DualQuat dq;
            for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
            {
                const int idx=nbRefs.getValue() *i+m;
                const int idxReps=m_reps[idx];
                // Create a rigid transformation from global frame to "in" frame.
                DualQuat dqi ( in[idxReps].getCenter(),
                        in[idxReps].getOrientation() );

                // Save rotated points for applyJ/JT
                rotatedPoints[idx] = dqi.transform ( initPos[idx] );

                // Blend all the transformations
                dq += dqi * m_coefs[idx];
            }
            dq.normalize(); // Normalize it
            out[i] = dq.transform ( initBlendedPos[i] ); // And apply it
        }
        break;
    }
#endif
    default: {}
    }
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    const sofa::helper::vector<unsigned int>& m_reps = repartition.getValue();
    const sofa::helper::vector<double>& m_coefs = coefs.getValue();

    Deriv v,omega;
    out.resize ( initPos.size() / nbRefs.getValue() );

    if (!(maskTo->isInUse()) )
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
                out[i] += ( v - cross ( rotatedPoints[idx],omega ) ) * m_coefs[idx];
            }
        }
    }
    else
    {
        typedef core::componentmodel::behavior::BaseMechanicalState::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;
        for (it=indices.begin(); it!=indices.end(); it++)
        {
            const int i=(int)(*it);
            out[i] = Deriv();
            for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
            {
                const int idx=nbRefs.getValue() *i+m;
                const int idxReps=m_reps[idx];

                v = in[idxReps].getVCenter();
                omega = in[idxReps].getVOrientation();
                out[i] += ( v - cross ( rotatedPoints[idx],omega ) ) * m_coefs[idx];
            }
        }
    }
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const sofa::helper::vector<unsigned int>& m_reps = repartition.getValue();
    const sofa::helper::vector<double>& m_coefs = coefs.getValue();

    Deriv v,omega;
    if ( !(maskTo->isInUse()) )
    {
        maskFrom->setInUse(false);
        for ( unsigned int i=0; i<in.size(); i++ )
        {
            for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
            {
                Deriv f = in[i];
                v = f;
                const int idx=nbRefs.getValue() *i+m;
                const int idxReps=m_reps[idx];
                omega = cross ( rotatedPoints[idx],f );
                out[idxReps].getVCenter() += v * m_coefs[idx];
                out[idxReps].getVOrientation() += omega * m_coefs[idx];
            }
        }
    }
    else
    {

        typedef core::componentmodel::behavior::BaseMechanicalState::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;
        for (it=indices.begin(); it!=indices.end(); it++)
        {
            const int i=(int)(*it);
            for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
            {
                Deriv f = in[i];
                v = f;
                const int idx=nbRefs.getValue() *i+m;
                const int idxReps=m_reps[idx];
                omega = cross ( rotatedPoints[idx],f );
                out[idxReps].getVCenter() += v * m_coefs[idx];
                out[idxReps].getVOrientation() += omega * m_coefs[idx];

                maskFrom->insertEntry(idxReps);
            }
        }
    }

}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::applyJT ( typename In::VecConst& out, const typename Out::VecConst& in )
{
    const sofa::helper::vector<unsigned int>& m_reps = repartition.getValue();
    const sofa::helper::vector<double>& m_coefs = coefs.getValue();
    const unsigned int nbr = nbRefs.getValue();
    const unsigned int nbi = this->fromModel->getX()->size();
    Deriv omega;
    typename In::VecDeriv v;
    sofa::helper::vector<bool> flags;
    int outSize = out.size();
    out.resize ( in.size() + outSize ); // we can accumulate in "out" constraints from several mappings
    for ( unsigned int i=0; i<in.size(); i++ )
    {
        v.clear();
        v.resize ( nbi );
        flags.clear();
        flags.resize ( nbi );
        OutConstraintIterator itOut;
        for (itOut=in[i].getData().begin(); itOut!=in[i].getData().end(); itOut++)
        {
            unsigned int indexIn = itOut->first;
            Deriv data = (Deriv) itOut->second;
            Deriv f = data;
            for ( unsigned int m=0 ; m<nbr; m++ )
            {
                omega = cross ( rotatedPoints[nbr*indexIn+m],f );
                flags[m_reps[nbr*indexIn+m] ] = true;
                v[m_reps[nbr*indexIn+m] ].getVCenter() += f * m_coefs[nbr*indexIn+m];
                v[m_reps[nbr*indexIn+m] ].getVOrientation() += omega * m_coefs[nbr*indexIn+m];
            }
        }
        for ( unsigned int j=0 ; j<nbi; j++ )
        {
            //if (!(v[i] == typename In::Deriv()))
            if ( flags[j] )
                out[outSize+i].insert (j,v[j] );
        }
    }
}

template <class BasicMapping>
void SkinningMapping<BasicMapping>::draw()
{
    if ( !this->getShow() ) return;
    const typename Out::VecCoord& xOut = *this->toModel->getX();
    const typename In::VecCoord& xIn = *this->fromModel->getX();
    sofa::helper::vector<unsigned int> m_reps = repartition.getValue();
    sofa::helper::vector<double> m_coefs = coefs.getValue();

    if ( interpolation != INTERPOLATION_DUAL_QUATERNION )
    {
        glDisable ( GL_LIGHTING );
        glPointSize ( 1 );
        glColor4f ( 1,1,0,1 );
        glBegin ( GL_LINES );

        for ( unsigned int i=0; i<xOut.size(); i++ )
        {
            for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
            {
                if ( m_coefs[nbRefs.getValue() *i+m] > 0.0 )
                {
                    glColor4d ( m_coefs[nbRefs.getValue() *i+m],m_coefs[nbRefs.getValue() *i+m],0,1 );
                    helper::gl::glVertexT ( xIn[m_reps[nbRefs.getValue() *i+m] ].getCenter() );
                    helper::gl::glVertexT ( xOut[i] );
                }
            }
        }
        glEnd();
    }

#ifdef SOFA_DEV
    //*  Animation continue des repères le long de la poutre.
    if ( interpolation == INTERPOLATION_DUAL_QUATERNION )
    {
        bool anim = true;
        static unsigned int step = 0;
        double transfoM4[16];
        unsigned int nbSteps = 500;
        step++;
        if ( step > nbSteps ) step = 0;

        for ( unsigned int i=1; i<xIn.size(); i++ )
        {
            DualQuat dq1 ( Vec3d ( 0,0,0 ), Quat::identity(), xIn[i-1].getCenter(), xIn[i-1].getOrientation() );
            DualQuat dq2 ( Vec3d ( 0,0,0 ), Quat::identity(), xIn[i].getCenter(), xIn[i].getOrientation() );

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
#endif
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
