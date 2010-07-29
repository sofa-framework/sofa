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
#define SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_CPP

#include <sofa/component/mapping/SkinningMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MechanicalMapping.inl>
#include <sofa/core/Mapping.inl>
namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(SkinningMapping);

using namespace defaulttype;
using namespace core;
using namespace core::behavior;



// Register in the Factory
int SkinningMappingClass = core::RegisterObject("skin a model from a set of rigid dofs")

#ifndef SOFA_FLOAT
        .add< SkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > > >()
// .add< SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > > >()

// Affine
        .add< SkinningMapping< MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > > >()
#endif
#ifndef SOFA_DOUBLE
        .add< SkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > > >()
// .add< SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > > >()
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< SkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< SkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3fTypes> > > >()
        .add< SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > > >()
#endif
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > >;
#endif
#endif





///////////////////////////////////////////////////////////////////////////////
//                           Affine Specialization                           //
///////////////////////////////////////////////////////////////////////////////


template <>
void BasicSkinningMapping<MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > >::computeInitPos ( )
{
    const VecCoord& xto = ( this->toModel->getX0() == NULL)?*this->toModel->getX():*this->toModel->getX0();
    const VecInCoord& xfrom = *this->fromModel->getX0();

    const vector<int>& m_reps = repartition.getValue();

    initPos.resize ( xto.size() * nbRefs.getValue() );
    for ( unsigned int i = 0; i < xto.size(); i++ )
        for ( unsigned int m = 0; m < nbRefs.getValue(); m++ )
        {
            Mat33 affineInv;
            affineInv.invert( xfrom[m_reps[nbRefs.getValue() *i+m]].getAffine() );
            initPos[nbRefs.getValue() *i+m] = affineInv * ( xto[i] - xfrom[m_reps[nbRefs.getValue() *i+m]].getCenter() );
        }
}


template <>
void BasicSkinningMapping<MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > >::apply ( Out::VecCoord& out, const In::VecCoord& in )
{
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_coefs = coefs.getValue();

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
            rotatedPoints[idx] = in[idxReps].getAffine() * initPos[idx];

            // And add each reference frames contributions to the new position out[i]
            out[i] += ( in[idxReps ].getCenter() + rotatedPoints[idx] ) * m_coefs[idxReps][i];
        }
    }
}



template <>
void BasicSkinningMapping<MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > >::applyJ ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    //const vector<int>& m_reps = repartition.getValue();
    //const VVD& m_coefs = coefs.getValue();
    VecCoord& xto = *this->toModel->getX();
    out.resize ( xto.size() );
    Deriv v;
    In::Deriv::Affine omega;

    if ( ! ( maskTo->isInUse() ) )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            out[i] = Deriv();
            for ( unsigned int j=0 ; j<in.size(); j++ )
            {
                Vec12 speed;
                speed[0]  = in[i][0];
                speed[1]  = in[i][1];
                speed[2]  = in[i][2];
                speed[3]  = in[i][3];
                speed[4]  = in[i][4];
                speed[5]  = in[i][5];
                speed[6]  = in[i][6];
                speed[7]  = in[i][7];
                speed[8]  = in[i][8];
                speed[9]  = in[i][9];
                speed[10] = in[i][10];
                speed[11] = in[i][11];

                Vec3 f = ( this->J[j][i] * speed );

                out[j] += Deriv ( f[0], f[1], f[2] );
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
                Vec12 speed;
                speed[0]  = in[i][0];
                speed[1]  = in[i][1];
                speed[2]  = in[i][2];
                speed[3]  = in[i][3];
                speed[4]  = in[i][4];
                speed[5]  = in[i][5];
                speed[6]  = in[i][6];
                speed[7]  = in[i][7];
                speed[8]  = in[i][8];
                speed[9]  = in[i][9];
                speed[10] = in[i][10];
                speed[11] = in[i][11];

                Vec3 f = ( this->J[j][i] * speed );

                out[j] += Deriv ( f[0], f[1], f[2] );
            }
        }
    }
}

template <>
void BasicSkinningMapping<MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > >::applyJT ( In::VecDeriv& out, const Out::VecDeriv& in )
{
    //const vector<int>& m_reps = repartition.getValue();
    //const VVD& m_coefs = coefs.getValue();

    Deriv v;
    In::Deriv::Affine omega;
    if ( ! ( maskTo->isInUse() ) )
    {
        maskFrom->setInUse ( false );
        for ( unsigned int j=0; j<in.size(); j++ ) // VecType
        {
            for ( unsigned int i=0 ; i<out.size(); i++ ) // AffineType
            {
                Mat12x3 Jt;
                Jt.transpose ( this->J[i][j] );

                Vec3 f;
                f[0] = in[j][0];
                f[1] = in[j][1];
                f[2] = in[j][2];
                Vec12 speed = Jt * f;

                omega[0][0] = speed[0];
                omega[0][1] = speed[1];
                omega[0][2] = speed[2];
                omega[1][0] = speed[3];
                omega[1][1] = speed[4];
                omega[1][2] = speed[5];
                omega[2][0] = speed[6];
                omega[2][1] = speed[7];
                omega[2][2] = speed[8];
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
                Mat12x3 Jt;
                Jt.transpose ( this->J[i][j] );

                Vec3 f;
                f[0] = in[j][0];
                f[1] = in[j][1];
                f[2] = in[j][2];
                Vec12 speed = Jt * f;

                omega[0][0] = speed[0];
                omega[0][1] = speed[1];
                omega[0][2] = speed[2];
                omega[1][0] = speed[3];
                omega[1][1] = speed[4];
                omega[1][2] = speed[5];
                omega[2][0] = speed[6];
                omega[2][1] = speed[7];
                omega[2][2] = speed[8];
                v = Deriv ( speed[9], speed[10], speed[11] );

                out[i].getVCenter() += v;
                out[i].getVAffine() += omega;
            }
        }
    }

}

template <>
void BasicSkinningMapping<MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > >::applyJT ( In::VecConst& out, const Out::VecConst& in )
{
    //const vector<int>& m_reps = repartition.getValue();
    //const VVD& m_coefs = coefs.getValue();
    //const unsigned int nbr = nbRefs.getValue();
    const unsigned int nbi = this->fromModel->getX()->size();
    In::Deriv::Affine omega;
    In::VecDeriv v;
    vector<bool> flags;
    int outSize = out.size();
    out.resize ( in.size() + outSize ); // we can accumulate in "out" constraints from several mappings

    if ( !this->enableSkinning.getValue()) return;
    const unsigned int numOut=this->J.size();

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

            for (unsigned int j=0; j<numOut; ++j)
            {
                Mat12x3 Jt;
                Jt.transpose ( this->J[j][indexIn] );

                Vec12 speed = Jt * data;

                In::Deriv::Affine affine;
                affine[0][0] = speed[0];
                affine[0][1] = speed[1];
                affine[0][2] = speed[2];
                affine[1][0] = speed[3];
                affine[1][1] = speed[4];
                affine[1][2] = speed[5];
                affine[2][0] = speed[6];
                affine[2][1] = speed[7];
                affine[2][2] = speed[8];
                const Vec3 pos( speed[9], speed[10], speed[11] );
                InDeriv value(pos,affine);
                out[outSize+i].add(j,value);
            }
        }
    }
}

#ifdef SOFA_DEV
template <>
void BasicSkinningMapping<MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > >::precomputeMatrices()
// precomputeMatrices( Vec3& pmt0,Mat3xIn& J,Mat33& Atilde, const Vec3&  p0, const double&  w,Vec3& dw, const typename In::Coord& xi0)
{
    const VecInCoord& xfrom0 = *this->fromModel->getX0();
    const VecCoord& xto0 = *this->toModel->getX0();
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_coefs = coefs.getValue();
    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());

    for ( unsigned int i=0 ; i<xto0.size(); i++ )
    {
        for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
        {
            const int& idx=nbRefs.getValue() *i+m;
            const int& idxReps=m_reps[idx];

            const InCoord& xi0 = xfrom0[idxReps];
            const Mat33& affine = xi0.getAffine();
            Mat33 affineInv;
            affineInv.invert (affine);

            for(int k=0; k<3; k++)
            {
                for(int l=0; l<3; l++)
                {
                    (Atilde[i])[k][l] = (initPos[idx])[k] * (m_dweight[idxReps][i])[l]  +  m_coefs[idxReps][i] * (affineInv)[k][l];
                }
            }


            Mat3xIn Ji = J[idxReps][i];
            Ji.fill(0);
            double val;
            for(int k=0; k<3; k++)
            {
                val = m_coefs[idxReps][i] * initPos[idx][k];
                Ji[0][k]=val;
                Ji[1][k+3]=val;
                Ji[2][k+6]=val;
                Ji[k][k+9]=m_coefs[idxReps][i];
            }
        }
    }
}
#endif



#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > >;
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Affine3dTypes>, MappedModel<Vec3dTypes> > >;
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Affine3dTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_DOUBLE
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Affine3fTypes>, MechanicalState<Vec3fTypes> > >;
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Affine3fTypes>, MappedModel<Vec3fTypes> > >;
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Affine3fTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3fTypes> > >;
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Affine3fTypes>, MechanicalState<Vec3dTypes> > >;
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Affine3dTypes>, MappedModel<Vec3fTypes> > >;
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Affine3fTypes>, MappedModel<Vec3dTypes> > >;
#endif
#endif


} // namespace mapping

} // namespace component

} // namespace sofa

