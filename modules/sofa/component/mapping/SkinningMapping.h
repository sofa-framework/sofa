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
#ifndef SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_H
#define SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_H

#include <sofa/core/behavior/MechanicalMapping.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MappedModel.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/helper/SVector.h>

#include <vector>

#include <sofa/component/component.h>
#include <sofa/helper/OptionsGroup.h>

#ifdef SOFA_DEV
#include <sofa/component/topology/HexahedronGeodesicalDistance.h>

#include <sofa/helper/DualQuat.h>
#include <sofa/helper/Quater.h>
//#include <sofa/component/mapping/DualQuatStorage.h>
#include "DualQuatStorage.h"
#endif

namespace sofa
{

namespace component
{

namespace mapping
{

using sofa::helper::vector;
using sofa::helper::Quater;
using sofa::helper::SVector;
#ifdef SOFA_DEV
using sofa::component::topology::HexahedronGeodesicalDistance;
#endif

#define DISTANCE_EUCLIDIAN 0
#define DISTANCE_GEODESIC 1
#define DISTANCE_HARMONIC 2

#define WEIGHT_NONE 0
#define WEIGHT_INVDIST_SQUARE 1
#define WEIGHT_LINEAR 2
#define WEIGHT_HERMITE 3
#define WEIGHT_SPLINE 4

#define INTERPOLATION_LINEAR 0
#define INTERPOLATION_DUAL_QUATERNION 1

/*
typedef enum
{
	DISTANCE_EUCLIDIAN, DISTANCE_GEODESIC, DISTANCE_HARMONIC
} DistanceType;

typedef enum
{
	WEIGHT_LINEAR, WEIGHT_INVDIST_SQUARE, WEIGHT_HERMITE
} WeightingType;

typedef enum
{
	INTERPOLATION_LINEAR, INTERPOLATION_DUAL_QUATERNION
} InterpolationType;*/


template <class BasicMapping>
#ifdef SOFA_DEV
class SkinningMapping : public BasicMapping, public DualQuatStorage<BasicMapping::Out::DataTypes::spatial_dimensions, typename BasicMapping::Out::DataTypes::Real>
#else
class SkinningMapping : public BasicMapping
#endif
{
public:
    SOFA_CLASS ( SOFA_TEMPLATE ( SkinningMapping,BasicMapping ), BasicMapping );
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::DataTypes DataTypes;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename defaulttype::SparseConstraint<Deriv> OutSparseConstraint;
    typedef typename OutSparseConstraint::const_data_iterator OutConstraintIterator;

    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::Real InReal;
    typedef typename Out::Real Real;
    enum { N=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;
    //typedef defaulttype::Mat<3,1,Real> Mat31;
    typedef defaulttype::Mat<3,3,Real> Mat33;
    typedef defaulttype::Mat<3,6,Real> Mat36;
    typedef vector<Mat36> VMat36;
    typedef vector<VMat36> VVMat36;
    typedef defaulttype::Mat<3,8,Real> Mat38;
    typedef defaulttype::Mat<4,4,Real> Mat44;
    //typedef defaulttype::Mat<6,1,Real> Mat61;
    typedef defaulttype::Mat<6,3,Real> Mat63;
    typedef defaulttype::Mat<6,6,Real> Mat66;
    typedef vector<Mat66> VMat66;
    typedef vector<VMat66> VVMat66;
    //typedef defaulttype::Mat<8,1,Real> Mat81;
    typedef defaulttype::Mat<8,3,Real> Mat83;
    typedef defaulttype::Mat<8,6,Real> Mat86;
    typedef vector<Mat86> VMat86;
    typedef defaulttype::Mat<8,8,Real> Mat88;
    typedef vector<Mat88> VMat88;
    typedef defaulttype::Vec<3,Real> Vec3;
    typedef vector<Vec3> VVec3;
    typedef vector<VVec3> VVVec3;
    typedef defaulttype::Vec<4,Real> Vec4;
    typedef defaulttype::Vec<6,Real> Vec6;
    typedef vector<Vec6> VVec6;
    typedef vector<VVec6> VVVec6;
    typedef defaulttype::Vec<8,Real> Vec8;
    typedef Quater<InReal> Quat;
    typedef sofa::helper::vector< VecCoord > VecVecCoord;
    typedef SVector<double> VD;
    typedef SVector<SVector<double> > VVD;

#ifdef SOFA_DEV
    // These typedef are here to avoid compilation pb encountered with ResizableExtVect Type.
    typedef typename sofa::defaulttype::StdVectorTypes<sofa::defaulttype::Vec<N, double>, sofa::defaulttype::Vec<N, double>, double> GeoType; // = Vec3fTypes or Vec3dTypes
    typedef typename HexahedronGeodesicalDistance< GeoType >::VecCoord GeoVecCoord;
    typedef typename HexahedronGeodesicalDistance< GeoType >::Coord GeoCoord;
    typedef typename HexahedronGeodesicalDistance< GeoType >::VecVecCoord GeoVecVecCoord;

    typedef typename helper::DualQuatd DualQuat;
    typedef typename DualQuatStorage<N, Real>::DUALQUAT DUALQUAT;
    typedef typename DualQuatStorage<N, Real>::VDUALQUAT VDUALQUAT;
#else
    typedef Coord GeoCoord;
    typedef VecCoord GeoVecCoord;
#endif
protected:
    vector<Coord> initPos; // pos: point coord in the world reference frame
    vector<Coord> rotatedPoints;

    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;

    Data<vector<int> > repartition;
    Data<VVD > coefs;
    Data<SVector<SVector<GeoCoord> > > weightGradients;
    Data<unsigned int> nbRefs;
public:
    Data<bool> showBlendedFrame;
    Data<bool> showDefTensors;
    Data<bool> showDefTensorsValues;
    Data<double> showDefTensorScale;
    Data<unsigned int> showFromIndex;
    Data<bool> showDistancesValues;
    Data<bool> showCoefs;
    Data<double> showGammaCorrection;
    Data<bool> showCoefsValues;
    Data<bool> showReps;
    Data<int> showValuesNbDecimals;
    Data<double> showTextScaleFactor;
    Data<bool> showGradients;
    Data<bool> showGradientsValues;
    Data<double> showGradientsScaleFactor;
#ifdef SOFA_DEV
    HexahedronGeodesicalDistance< GeoType>* geoDist;
    Data<double> newFrameMinDist;
    Data<vector<double> > newFrameWeightingRadius;
    Data<double> newFrameDefaultCutOffDistance;
    Data<double> newFrameDistanceToMaximizeWeight;
    Data<bool> enableSkinning;
    Data<double> voxelVolume;
#endif

protected:
    Data<sofa::helper::OptionsGroup> wheightingType;
    Data<sofa::helper::OptionsGroup> interpolationType;
    Data<sofa::helper::OptionsGroup> distanceType;
    bool computeWeights;
    VVD distances;
#ifdef SOFA_DEV
    GeoVecVecCoord distGradients;
#else
    vector<vector<GeoCoord> > distGradients;
#endif

    inline void computeInitPos();
    inline void computeDistances();
    inline void sortReferences( vector<int>& references);

public:
    SkinningMapping ( In* from, Out* to );
    virtual ~SkinningMapping();

    void init();

    void apply ( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT ( typename In::VecConst& out, const typename Out::VecConst& in );

    void draw();
    void clear();

    // Weights
    void setWeightsToHermite();
    void setWeightsToInvDist();
    void setWeightsToLinear();
    inline void updateWeights();
    inline void getDistances( int xfromBegin);
    //inline void temporaryUpdateWeightsAfterInsertion( VVD& w, VecVecCoord& dw, int xfromBegin);

    // Interpolations
    void setInterpolationToLinear();
    void setInterpolationToDualQuaternion();

    // Accessors
    void setNbRefs ( unsigned int nb )
    {
        nbRefs.setValue ( nb );
    }
    void setWeightCoefs ( VVD& weights );
    void setRepartition ( vector<int> &rep );
    void setComputeWeights ( bool val )
    {
        computeWeights=val;
    }
    unsigned int getNbRefs()
    {
        return nbRefs.getValue();
    }
    const VVD& getWeightCoefs()
    {
        return coefs.getValue();
    }
    const vector<int>& getRepartition()
    {
        return repartition.getValue();
    }
    bool getComputeWeights()
    {
        return computeWeights;
    }

#ifdef SOFA_DEV
    void computeDqL ( Mat86& L, const DUALQUAT& qi, const Coord& ti );
    void BlendDualQuat ( DUALQUAT& b, DUALQUAT& bn, double& QEQ0, double& Q0Q0, double& Q0, const int& indexp, const VDUALQUAT& qrel, const VVD& w );

    void computeQrel ( DUALQUAT& qrel, const Mat88& T, const DUALQUAT& q );
    void computeDqRigid ( Mat33& R, Vec3& t, const DUALQUAT& bn );
    void computeDqN ( Mat88& N, const Mat44& q0q0T, const Mat44& q0qeT, const Mat44& qeq0T , const double& QEQ0, const double& Q0Q0, const double& Q0 );
    void computeDqN_constants ( Mat44& q0q0T, Mat44& q0qeT, Mat44& qeq0T, const DUALQUAT& bn );
    void computeDqDN ( Mat88& DN, const Mat44& q0q0T, const Mat44& q0qeT, const Mat44& qeq0T, const double& QEQ0, const double& Q0Q0, const double& Q0, const Mat44& q0V0T, const Mat44& V0q0T, const double& q0V0, const Mat44& q0VeT, const Mat44& Veq0T, const double& q0Ve, const Mat44& qeV0T, const Mat44& V0qeT, const double& qeV0, const DUALQUAT& V );
    void computeDqDN_constants ( Mat44& q0V0T, Mat44& V0q0T, double& q0V0, Mat44& q0VeT, Mat44& Veq0T,double& q0Ve, Mat44& qeV0T, Mat44& V0qeT, double& qeV0, const DUALQUAT& bn, const DUALQUAT& V );
    void XItoQ ( DUALQUAT& q, const InCoord& xi );
    void getCov ( Mat44& q1q2T, const Vec4& q1, const Vec4& q2 );
    void computeDqQ ( Mat38& Q, const DUALQUAT& bn, const Vec3& p );
    void computeDqDR ( Mat33& DR, const DUALQUAT& bn, const DUALQUAT& V );
    void computeDqDQ ( Mat38& DQ, const Vec3& p, const DUALQUAT& V );
    void computeDqT ( Mat88& T, const DUALQUAT& qi0 );
    void Multi_Q(Quat& q, const Vec4& q1, const Quat& q2);

    void removeFrame( const unsigned int index);
    void insertFrame( const Coord& pos, const Quat& rot, GeoVecCoord beginPointSet = GeoVecCoord(), double distMax = 0.0);
    bool inverseSkinning( InCoord& X0, InCoord& X, const InCoord& Xtarget);
    void computeWeight( VVD& w, VecVecCoord& dw, const Coord& x0);
    void updateDataAfterInsertion();
    inline void changeSettingsDueToInsertion();

    void apply0 ();

#endif
};

using core::Mapping;
using core::behavior::MechanicalMapping;
using core::behavior::MappedModel;
using core::behavior::State;
using core::behavior::MechanicalState;

using sofa::defaulttype::Vec2dTypes;
using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec2fTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec2fTypes;
using sofa::defaulttype::ExtVec3fTypes;
using sofa::defaulttype::Rigid2dTypes;
using sofa::defaulttype::Rigid3dTypes;
using sofa::defaulttype::Rigid2fTypes;
using sofa::defaulttype::Rigid3fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > >;
#endif
#endif
#endif




} // namespace mapping

} // namespace component

} // namespace sofa

#endif
