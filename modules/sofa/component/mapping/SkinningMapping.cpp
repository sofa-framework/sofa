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

// Rigid Types
#ifndef SOFA_FLOAT
        .add< SkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > > >()
// .add< SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > > >()
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


// Affine Types
#ifdef SOFA_DEV
#ifndef SOFA_FLOAT
        .add< SkinningMapping< MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > > >()
//.add< SkinningMapping< Mapping< State<Affine3dTypes>, MappedModel<Vec3dTypes> > > >()
//.add< SkinningMapping< Mapping< State<Affine3dTypes>, MappedModel<ExtVec3dTypes> > > >()
//.add< SkinningMapping< Mapping< State<Affine3dTypes>, MappedModel<ExtVec3fTypes> > > >()
#endif
#ifndef SOFA_DOUBLE
//.add< SkinningMapping< MechanicalMapping< MechanicalState<Affine3fTypes>, MechanicalState<Vec3fTypes> > > >()
//.add< SkinningMapping< Mapping< State<Affine3fTypes>, MappedModel<Vec3fTypes> > > >()
// .add< SkinningMapping< Mapping< State<Affine3fTypes>, MappedModel<ExtVec3dTypes> > > >()
//.add< SkinningMapping< Mapping< State<Affine3fTypes>, MappedModel<ExtVec3fTypes> > > >()
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
//.add< SkinningMapping< MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3fTypes> > > >()
//.add< SkinningMapping< MechanicalMapping< MechanicalState<Affine3fTypes>, MechanicalState<Vec3dTypes> > > >()
//.add< SkinningMapping< Mapping< State<Affine3dTypes>, MappedModel<Vec3fTypes> > > >()
//.add< SkinningMapping< Mapping< State<Affine3fTypes>, MappedModel<Vec3dTypes> > > >()
#endif
#endif
#endif


// Quadratic Types
#ifdef SOFA_DEV
#ifndef SOFA_FLOAT
        .add< SkinningMapping< MechanicalMapping< MechanicalState<Quadratic3dTypes>, MechanicalState<Vec3dTypes> > > >()
//.add< SkinningMapping< Mapping< State<Quadratic3dTypes>, MappedModel<Vec3dTypes> > > >()
//.add< SkinningMapping< Mapping< State<Quadratic3dTypes>, MappedModel<ExtVec3dTypes> > > >()
//.add< SkinningMapping< Mapping< State<Quadratic3dTypes>, MappedModel<ExtVec3fTypes> > > >()
#endif
#ifndef SOFA_DOUBLE
//.add< SkinningMapping< MechanicalMapping< MechanicalState<Quadratic3fTypes>, MechanicalState<Vec3fTypes> > > >()
//.add< SkinningMapping< Mapping< State<Quadratic3fTypes>, MappedModel<Vec3fTypes> > > >()
// .add< SkinningMapping< Mapping< State<Quadratic3fTypes>, MappedModel<ExtVec3dTypes> > > >()
//.add< SkinningMapping< Mapping< State<Quadratic3fTypes>, MappedModel<ExtVec3fTypes> > > >()
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
//.add< SkinningMapping< MechanicalMapping< MechanicalState<Quadratic3dTypes>, MechanicalState<Vec3fTypes> > > >()
//.add< SkinningMapping< MechanicalMapping< MechanicalState<Quadratic3fTypes>, MechanicalState<Vec3dTypes> > > >()
//.add< SkinningMapping< Mapping< State<Quadratic3dTypes>, MappedModel<Vec3fTypes> > > >()
//.add< SkinningMapping< Mapping< State<Quadratic3fTypes>, MappedModel<Vec3dTypes> > > >()
#endif
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




#ifdef SOFA_DEV
///////////////////////////////////////////////////////////////////////////////
//                           Affine Specialization                           //
///////////////////////////////////////////////////////////////////////////////


template <>
void SkinningMapping<MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > >::precomputeMatrices()
{
    const VecInCoord& xfrom0 = *this->fromModel->getX0();
    const VecCoord& xto0 = *this->toModel->getX0();
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_weights = weights.getValue();
    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());

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
    this->B.resize(xfrom0.size());
    for(unsigned int i = 0; i < xfrom0.size(); ++i)
        this->B[i].resize(xto.size());
    Atilde.resize(xfrom0.size());
    for (unsigned int i = 0; i < xfrom0.size(); ++i)
        Atilde[i].resize(xto0.size());
    J.resize(xfrom0.size());
    for (unsigned int i = 0; i < xfrom0.size(); ++i)
        J[i].resize(xto0.size());

    // Precompute matrices
    for ( unsigned int i=0 ; i<xto0.size(); i++ )
    {
        for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
        {
            const int& idx = nbRefs.getValue() *i+m;
            const int& idxReps = m_reps[idx];

            const InCoord& xi0 = xfrom0[idxReps];
            const Mat33& affine = xi0.getAffine();
            Mat33 affineInv;
            affineInv.invert (affine);

            for(int k=0; k<3; k++)
            {
                for(int l=0; l<3; l++)
                {
                    (Atilde[idxReps][i])[k][l] = (initPos[idx])[k] * (m_dweight[idxReps][i])[l]  +  m_weights[idxReps][i] * (affineInv)[k][l];
                }
            }

            Mat3xIn& Ji = J[idxReps][i];
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


template <>
void SkinningMapping<MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > >::apply ( Out::VecCoord& out, const In::VecCoord& in )
{
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_weights = weights.getValue();

    rotatedPoints.resize ( initPos.size() );
    out.resize ( initPos.size() / nbRefs.getValue() );

    // Resize matrices
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

        Mat33 F, Finv, E;
        F.fill ( 0 );
        E.fill ( 0 );
        for ( unsigned int j = 0 ; j < nbRefs.getValue(); ++j )
        {
            const int& idx = nbRefs.getValue() * i + j;
            const int& idxReps = m_reps[idx];

            Mat33 cov;
            getCov33 ( cov, in[idxReps ].getCenter(), dw[idxReps][i] );
            F += cov + in[idxReps ].getAffine() * this->Atilde[idxReps][i];
        }

        // strain and determinant
        this->det[i] = determinant ( F );
        invertMatrix ( Finv, F );
        for ( unsigned int k = 0; k < 3; ++k )
        {
            for ( unsigned int j = 0; j < 3; ++j )
                for ( unsigned int l = 0; l < 3; ++l )
                    E[k][j] += F[l][j] * F[l][k];

            E[k][k] -= 1.;
        }
        E /= 2.; // update E=1/2(U^TU-I)
        this->deformationTensors[i][0] = E[0][0];
        this->deformationTensors[i][1] = E[1][1];
        this->deformationTensors[i][2] = E[2][2];
        this->deformationTensors[i][3] = E[0][1];
        this->deformationTensors[i][4] = E[1][2];
        this->deformationTensors[i][5] = E[0][2]; // column form

        // update B and ddet
        for ( unsigned int j = 0 ; j < nbRefs.getValue(); ++j )
        {
            unsigned int k, l, m;
            const int& idx = nbRefs.getValue() * i + j;
            const int& idxReps = m_reps[idx];

            Mat6xIn& Bij = this->B[idxReps][i];
            const Mat33& At = this->Atilde[idxReps][i];
            const Vec3& dWeight = dw[idxReps][i];

            // stretch
            for ( k = 0; k < 3; k++ ) for ( m = 0; m < 3; m++ ) for ( l = 0; l < 3; l++ ) Bij[m][3*l+k] = F[l][m] * At[k][m];
            for ( k = 0; k < 3; k++ ) for ( m = 0; m < 3; m++ ) Bij[m][9+k] = dWeight [m] * F[k][m];

            // shear
            for ( k = 0; k < 3; k++ ) for ( l = 0; l < 3; l++ ) Bij[3][3*l+k] = 0.5 * ( F[l][0] * At[k][1] +
                            F[l][1] * At[k][0] );
            for ( k = 0; k < 3; k++ ) Bij[3][9+k] = 0.5 * ( dWeight [0] * F[k][1] + dWeight [1] * F[k][0] );
            for ( k = 0; k < 3; k++ ) for ( l = 0; l < 3; l++ )
                    Bij[4][3*l+k] = 0.5 * ( F[l][1] * At[k][2] + F[l][2] * At[k][1] );
            for ( k = 0; k < 3; k++ ) Bij[4][9+k] = 0.5 * ( dWeight [1] * F[k][2] + dWeight [2] * F[k][1] );
            for ( k = 0; k < 3; k++ ) for ( l = 0; l < 3; l++ )
                    Bij[5][3*l+k] = 0.5 * ( F[l][2] * At[k][0] + F[l][0] * At[k][2] );
            for ( k = 0; k < 3; k++ ) Bij[5][9+k] = 0.5 * ( dWeight [2] * F[k][0] + dWeight [0] * F[k][2] );

            // Compute ddet
            /*
            for ( k = 0;k < 12;k++ ) n->ddet[j].affine[k] = 0;
            for ( k = 0;k < 3;k++ ) for ( m = 0;m < 3;m++ ) for ( l = 0;l < 3;l++ ) n->ddet[j].affine[m+3*k] +=
                    At[m][l] * Finv[l][k];
            for ( k = 0;k < 3;k++ ) for ( l = 0;l < 3;l++ ) n->ddet[j].affine[9+k] += dWeight [l] * Finv[l][k];
            this->ddet[idxReps][i] = this->det[i] * this->ddet[idxReps][i];
            */
        }
    }
}


template <>
void SkinningMapping<MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > >::applyJ ( Out::VecDeriv& out, const In::VecDeriv& in )
{
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


template <>
void SkinningMapping<MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > >::applyJT ( In::VecDeriv& out, const Out::VecDeriv& in )
{
    Deriv v;
    In::Deriv::Affine omega;
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


template <>
void SkinningMapping<MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > >::applyJT (In::MatrixDeriv& out, const Out::MatrixDeriv& in )
{
    const unsigned int nbi = this->fromModel->getX()->size();
    In::Deriv::Affine omega;
    In::VecDeriv v;
    vector<bool> flags;

    if ( !this->enableSkinning.getValue())
        return;
    const unsigned int numOut = this->J.size();

    Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        v.clear();
        v.resize(nbi);
        flags.clear();
        flags.resize(nbi);

        In::MatrixDeriv::RowIterator o = out.end();

        Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        for (Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            const unsigned int indexIn = colIt.index(); // Point
            const Deriv data = colIt.val();

            for (unsigned int j=0; j<numOut; ++j) // Affine
            {
                MatInx3 Jt;
                Jt.transpose ( this->J[j][indexIn] );

                VecIn speed = Jt * data;

                In::Deriv::Affine affine;
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




///////////////////////////////////////////////////////////////////////////////
//                           Quadratic Specialization                           //
///////////////////////////////////////////////////////////////////////////////


template <>
void SkinningMapping<MechanicalMapping< MechanicalState<Quadratic3dTypes>, MechanicalState<Vec3dTypes> > >::precomputeMatrices()
{
    const VecInCoord& xfrom0 = *this->fromModel->getX0();
    const VecCoord& xto0 = *this->toModel->getX0();
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_weights = weights.getValue();
    SVector<SVector<GeoCoord> >& m_dweight = * ( weightGradients.beginEdit());

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
    this->B.resize(xfrom0.size());
    for(unsigned int i = 0; i < xfrom0.size(); ++i)
        this->B[i].resize(xto.size());
    Atilde.resize(xfrom0.size());
    for (unsigned int i = 0; i < xfrom0.size(); ++i)
        Atilde[i].resize(xto0.size());
    J.resize(xfrom0.size());
    for (unsigned int i = 0; i < xfrom0.size(); ++i)
        J[i].resize(xto0.size());

    // Precompute matrices ( suppose that A0=I )
    for ( unsigned int i=0 ; i<xto0.size(); i++ )
    {
        for ( unsigned int m=0 ; m<nbRefs.getValue(); m++ )
        {
            const int& idx = nbRefs.getValue() *i+m;
            const int& idxReps = m_reps[idx];

            const Vec3& p0 = initPos[idx];
            Vec9 p2 = Vec9( p0[0], p0[1], p0[2], p0[0]*p0[0], p0[1]*p0[1], p0[2]*p0[2], p0[0]*p0[1], p0[1]*p0[2], p0[0]*p0[2]);

            MatInAtx3& At = Atilde[idxReps][i];
            At[0][0] = 1;       At[0][1]=0;       At[0][2]=0;
            At[1][0] = 0;       At[1][1]=1;       At[1][2]=0;
            At[2][0] = 0;       At[2][1]=0;       At[2][2]=1;
            At[3][0] = 2*p2[0]; At[3][1]=0;       At[3][2]=0;
            At[4][0] = 0;       At[4][1]=2*p2[1]; At[4][2]=0;
            At[5][0] = 0;       At[5][1]=0;       At[5][2]=2*p2[2];
            At[6][0] = p2[1];   At[6][1]=p2[0];   At[6][2]=0;
            At[7][0] = 0;       At[7][1]=p2[2];   At[7][2]=p2[1];
            At[8][0] = p2[2];   At[8][1]=0;       At[8][2]=p2[0];
            At = m_weights[idxReps][i] * At;
            for (int k=0; k<9; k++) for (int l=0; l<3; l++) At[k][l] += p2[k] * m_dweight[idxReps][i][l];

            Mat3xIn& Ji = J[idxReps][i];
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


template <>
void SkinningMapping<MechanicalMapping< MechanicalState<Quadratic3dTypes>, MechanicalState<Vec3dTypes> > >::apply ( Out::VecCoord& out, const In::VecCoord& in )
{
    const vector<int>& m_reps = repartition.getValue();
    const VVD& m_weights = weights.getValue();

    rotatedPoints.resize ( initPos.size() );
    out.resize ( initPos.size() / nbRefs.getValue() );

    // Resize matrices
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

        Mat33 F, FT, Finv, E;
        F.fill ( 0 );
        E.fill ( 0 );
        for ( unsigned int j = 0 ; j < nbRefs.getValue(); ++j )
        {
            const int& idx = nbRefs.getValue() * i + j;
            const int& idxReps = m_reps[idx];

            Mat33 cov;
            getCov33 ( cov, in[idxReps ].getCenter(), dw[idxReps][i] );
            F += cov + in[idxReps ].getQuadratic() * this->Atilde[idxReps][i];
        }
        FT.transpose(F);

        // strain and determinant
        this->det[i] = determinant ( F );
        invertMatrix ( Finv, F );
        for ( unsigned int k = 0; k < 3; ++k )
        {
            for ( unsigned int j = 0; j < 3; ++j )
                for ( unsigned int l = 0; l < 3; ++l )
                    E[k][j] += F[l][j] * F[l][k];

            E[k][k] -= 1.;
        }
        E /= 2.; // update E=1/2(U^TU-I)
        this->deformationTensors[i][0] = E[0][0];
        this->deformationTensors[i][1] = E[1][1];
        this->deformationTensors[i][2] = E[2][2];
        this->deformationTensors[i][3] = E[0][1];
        this->deformationTensors[i][4] = E[1][2];
        this->deformationTensors[i][5] = E[0][2]; // column form

        // update B and ddet
        for ( unsigned int j = 0 ; j < nbRefs.getValue(); ++j )
        {
            unsigned int k, l, m;
            const int& idx = nbRefs.getValue() * i + j;
            const int& idxReps = m_reps[idx];

            Mat6xIn& Bij = this->B[idxReps][i];
            const MatInAtx3& At = this->Atilde[idxReps][i];
            const Vec3& dWeight = dw[idxReps][i];

            // stretch
            for(k=0; k<9; k++) for(m=0; m<3; m++) for(l=0; l<3; l++) Bij[m][9*l+k] = F[l][m] * At[k][m];
            for(k=0; k<3; k++) for(m=0; m<3; m++) Bij[m][27+k]=dWeight[m]*F[k][m];

            // shear
            for(k=0; k<9; k++) for(l=0; l<3; l++) Bij[3][9*l+k]=0.5*(F[l][0]*At[k][1] + F[l][1]*At[k][0]);
            for(k=0; k<3; k++) Bij[3][27+k]=0.5*(dWeight[0]*F[k][1] + dWeight[1]*F[k][0]);
            for(k=0; k<9; k++) for(l=0; l<3; l++) Bij[4][9*l+k]=0.5*(F[l][1]*At[k][2]+F[l][2]*At[k][1]);
            for(k=0; k<3; k++) Bij[4][27+k]=0.5*(dWeight[1]*F[k][2] + dWeight[2]*F[k][1]);
            for(k=0; k<9; k++) for(l=0; l<3; l++) Bij[5][9*l+k]=0.5*(F[l][2]*At[k][0]+F[l][0]*At[k][2]);
            for(k=0; k<3; k++) Bij[5][27+k]=0.5*(dWeight[2]*F[k][0] + dWeight[0]*F[k][2]);

            /*
                            // Compute ddet
                            for(k=0;k<30;k++) n->ddet[j].quadratic[k]=0;
                            for(k=0;k<3;k++) for(m=0;m<9;m++) for(l=0;l<3;l++) n->ddet[j].quadratic[m+9*k]+=
                            (*At2)[m][l]*Finv[l][k];
                            for(k=0;k<3;k++) for(l=0;l<3;l++) n->ddet[j].quadratic[27+k]+= (*dW)[l]*Finv[l][k];
                            n->ddet[j].quadratic=n->det * n->ddet[j].quadratic;
            */
        }
    }
}


template <>
void SkinningMapping<MechanicalMapping< MechanicalState<Quadratic3dTypes>, MechanicalState<Vec3dTypes> > >::applyJ ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    VecCoord& xto = *this->toModel->getX();
    out.resize ( xto.size() );
    Deriv v;
    In::Deriv::Quadratic omega;

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


template <>
void SkinningMapping<MechanicalMapping< MechanicalState<Quadratic3dTypes>, MechanicalState<Vec3dTypes> > >::applyJT ( In::VecDeriv& out, const Out::VecDeriv& in )
{
    Deriv v;
    In::Deriv::Quadratic omega;
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


template <>
void SkinningMapping<MechanicalMapping< MechanicalState<Quadratic3dTypes>, MechanicalState<Vec3dTypes> > >::applyJT (In::MatrixDeriv& out, const Out::MatrixDeriv& in )
{
    const unsigned int nbi = this->fromModel->getX()->size();
    In::Deriv::Quadratic omega;
    In::VecDeriv v;
    vector<bool> flags;

    if ( !this->enableSkinning.getValue())
        return;
    const unsigned int numOut = this->J.size();

    Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        v.clear();
        v.resize(nbi);
        flags.clear();
        flags.resize(nbi);

        In::MatrixDeriv::RowIterator o = out.end();

        Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        for (Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            const unsigned int indexIn = colIt.index();
            const Deriv data = colIt.val();

            for (unsigned int j=0; j<numOut; ++j)
            {
                MatInx3 Jt;
                Jt.transpose ( this->J[j][indexIn] );

                VecIn speed = Jt * data;

                In::Deriv::Quadratic quad;
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





#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Quadratic3dTypes>, MechanicalState<Vec3dTypes> > >;
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Quadratic3dTypes>, MappedModel<Vec3dTypes> > >;
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Quadratic3dTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_DOUBLE
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Quadratic3fTypes>, MechanicalState<Vec3fTypes> > >;
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Quadratic3fTypes>, MappedModel<Vec3fTypes> > >;
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Quadratic3fTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Quadratic3dTypes>, MechanicalState<Vec3fTypes> > >;
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Quadratic3fTypes>, MechanicalState<Vec3dTypes> > >;
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Quadratic3dTypes>, MappedModel<Vec3fTypes> > >;
//template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Quadratic3fTypes>, MappedModel<Vec3dTypes> > >;
#endif
#endif

#endif // ifdef SOFA_DEV

} // namespace mapping

} // namespace component

} // namespace sofa

