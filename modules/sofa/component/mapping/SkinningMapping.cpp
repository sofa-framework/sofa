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

