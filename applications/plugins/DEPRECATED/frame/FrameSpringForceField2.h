/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef FRAME_FRAMESPRINGFORCEFIELD2_H
#define FRAME_FRAMESPRINGFORCEFIELD2_H

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include "FrameStorage.h"
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/SVector.h>
#include <sofa/helper/accessor.h>

#include <sofa/core/objectmodel/DataFileName.h>
#include "initFrame.h"

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using sofa::component::mapping::FrameStorage;
using helper::SVector;


/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>

class FrameSpringForceField2InternalData
{

public:
};

/// Set of simple springs between particles
template<class DataTypes>

class FrameSpringForceField2 : public core::behavior::ForceField<DataTypes>
{

public:
    SOFA_CLASS ( SOFA_TEMPLATE ( FrameSpringForceField2, DataTypes ), SOFA_TEMPLATE ( core::behavior::ForceField, DataTypes ) );

    typedef typename core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename core::behavior::MechanicalState<DataTypes> MState;

    enum { N = DataTypes::spatial_dimensions };
    enum { InDerivDim=DataTypes::Deriv::total_size };
    typedef defaulttype::Mat<N, N, Real> Mat;
    typedef Vec<N, Real> VecN;
    typedef defaulttype::Mat<3, 3, Real> Mat33;
    typedef defaulttype::Mat<3, 6, Real> Mat36;
    typedef SVector<Mat36> VMat36;
    typedef SVector<VMat36> VVMat36;
    typedef defaulttype::Mat<3, 8, Real> Mat38;
    typedef defaulttype::Mat<4, 4, Real> Mat44;
    typedef defaulttype::Mat<6, 3, Real> Mat63;
    typedef defaulttype::Mat<6, 6, Real> Mat66;
    typedef SVector<Mat66> VMat66;
    typedef SVector<VMat66> VVMat66;
    typedef defaulttype::Mat<6,InDerivDim,Real> Mat6xIn;
    typedef SVector<Mat6xIn> VMat6xIn;
    typedef SVector<VMat6xIn> VVMat6xIn;
    typedef defaulttype::Mat<8, 3, Real> Mat83;
    typedef defaulttype::Mat<8, 6, Real> Mat86;
    typedef SVector<Mat86> VMat86;
    typedef defaulttype::Mat<8, 8, Real> Mat88;
    typedef SVector<Mat88> VMat88;
    typedef defaulttype::Mat<InDerivDim,6,Real> MatInx6;
    typedef defaulttype::Mat<InDerivDim,InDerivDim,Real> MatInxIn;
    typedef SVector<MatInxIn> VMatInxIn;
    typedef SVector<VMatInxIn> VVMatInxIn;
    typedef defaulttype::Vec<3, Real> Vec3;
    typedef SVector<Vec3> VVec3;
    typedef SVector<VVec3> VVVec3;
    typedef defaulttype::Vec<4, Real> Vec4;
    typedef defaulttype::Vec<6, Real> Vec6;
    typedef SVector<Vec6> VVec6;
    typedef SVector<VVec6> VVVec6;
    typedef defaulttype::Vec<8, Real> Vec8;
    typedef SVector<Real> VD;
    typedef defaulttype::Vec<InDerivDim,Real> VecIn;
    typedef SVector<SVector<VecIn> > VVVecIn;
    typedef FrameStorage<DataTypes, Real> FStorage;

    typedef struct
    {
        Vec4 q0;
        Vec4 qe;
    } DUALQUAT;
    typedef SVector<DUALQUAT> VDUALQUAT;

protected:
    bool maskInUse;
    SReal m_potentialEnergy;
    Data<double> youngModulus;
    Data<double> poissonRatio;
    Mat66 H;
    VVMatInxIn K;
    VVMatInxIn K0;
    VVMat6xIn* B;
    VVVecIn* ddet;
    VD* det;
    VD* vol;

    FrameSpringForceField2InternalData<DataTypes> data;

    friend class FrameSpringForceField2InternalData<DataTypes>;

public:
    FrameSpringForceField2 ( MState* obj);
    FrameSpringForceField2 ( );

    virtual bool canPrefetch() const
    {
        return false;
    }


    virtual void reinit();
    virtual void init();
    virtual void bwdInit();

    virtual void addForce ( VecDeriv& vf, const VecCoord& vx, const VecDeriv& vv);
    virtual void addDForce ( VecDeriv& df, const VecDeriv& dx );

    void draw();

    // -- Modifiers

    void clear ( )
    {
        K.clear();
        K0.clear();
        H.clear();
    }

    void updateForce ( VecDeriv& Force, VVMatInxIn& K, const VecCoord& xi, const VVMatInxIn& Kref );

    virtual double getPotentialEnergy ( const VecCoord& /*x*/ ) const
    {
        return m_potentialEnergy;
    }

    virtual double getPotentialEnergy ( const VecCoord&, const VecCoord& ) const
    {
        return m_potentialEnergy;
    }

private:
    FStorage* dqInfos;

    void computeK0();
    void QtoR( Mat33& M, const Quat& q);
    void Transform_Q( Vec3& pout, const Vec3& pin, const Quat& q, const bool& invert=false);
    void PostoSpeed( Deriv& Omega, const Coord& xi, const Coord& xi2);
    void Multi_Q( Quat& q, const Quat& q1, const Quat& q2);
    void getH_isotropic ( Mat66& H, const double& E, const double& v );
    void GetCrossproductMatrix(Mat33& C,const Vec3& u);

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FRAME_FRAMESPRINGFORCEFIELD2_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_FRAME_API FrameSpringForceField2<Rigid3dTypes>;
extern template class SOFA_FRAME_API FrameSpringForceField2<Affine3dTypes>;
//extern template class SOFA_FRAME_API FrameSpringForceField2<Quadratic3dTypes>;
#endif
#ifndef SOFA_DOUBLE
//extern template class SOFA_FRAME_API FrameSpringForceField2<Rigid3fTypes>;
//extern template class SOFA_FRAME_API FrameSpringForceField2<Affine3fTypes>;
//extern template class SOFA_FRAME_API FrameSpringForceField2<Quadratic3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
