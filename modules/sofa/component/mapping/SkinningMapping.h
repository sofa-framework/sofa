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
#include "DualQuatStorage.h"
#include <../applications/plugins/frame/AffineTypes.h>
#include <../applications/plugins/frame/QuadraticTypes.h>
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

////////////// Definitions to avoid multiple specializations //////////////////
// See "Substitution failure is not an error" desgin patern and boost::enable_if
// http://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error
// http://www.boost.org/doc/libs/1_36_0/libs/utility/enable_if.html
template <bool B, class T = void>
struct enable_if_c
{
    typedef T type;
};

template <class T>
struct enable_if_c<false, T> {};

template <class Cond, class T = void>
struct enable_if : public enable_if_c<Cond::value, T> {};

template <class T, class T2>
struct Equal
{
    typedef char ok;
    typedef long nok;

    static ok test(const T t, const T t2);
    static nok test(...);

    static const bool value=(sizeof(test(T(),T2()))==sizeof(ok));
};
///////////////////////////////////////////////////////////////////////////////


template <class BasicMapping>
#ifdef SOFA_DEV
class SkinningMapping : public BasicMapping, public DualQuatStorage<typename BasicMapping::In::DataTypes, typename BasicMapping::Out::Real>
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
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Real InReal;
    typedef typename Out::Real Real;
    enum { N=DataTypes::spatial_dimensions };
    //enum { InDerivDim=In::DataTypes::deriv_total_size };
#ifdef SOFA_DEV
    enum { InDOFs=sofa::frame::DataTypesInfo<In::DataTypes::spatial_dimensions, typename In::DataTypes::Real, typename In::DataTypes>::degrees_of_freedom };
    enum { InAt=sofa::frame::DataTypesInfo<In::DataTypes::spatial_dimensions, typename In::DataTypes::Real, typename In::DataTypes>::Atilde_nb_column };
#else
    enum { InDOFs=In::DataTypes::deriv_total_size };
    enum { InAt=0 };
#endif
    typedef defaulttype::Mat<N,N,Real> Mat;
    typedef defaulttype::Mat<3,3,Real> Mat33;
    typedef defaulttype::Mat<3,InDOFs,Real> Mat3xIn;
    typedef vector<Mat3xIn> VMat3xIn;
    typedef vector<VMat3xIn> VVMat3xIn;
    typedef defaulttype::Mat<InAt,3,Real> MatInAtx3;
    typedef vector<MatInAtx3> VMatInAtx3;
    typedef vector<VMatInAtx3> VVMatInAtx3;
    typedef defaulttype::Mat<3,6,Real> Mat36;
    typedef vector<Mat36> VMat36;
    typedef vector<VMat36> VVMat36;
    typedef defaulttype::Mat<3,7,Real> Mat37;
    typedef defaulttype::Mat<3,8,Real> Mat38;
    typedef defaulttype::Mat<3,9,Real> Mat39;
    typedef defaulttype::Mat<4,3,Real> Mat43;
    typedef vector<Mat43> VMat43;
    typedef defaulttype::Mat<4,4,Real> Mat44;
    typedef defaulttype::Mat<6,3,Real> Mat63;
    typedef defaulttype::Mat<6,6,Real> Mat66;
    typedef vector<Mat66> VMat66;
    typedef vector<VMat66> VVMat66;
    typedef defaulttype::Mat<6,7,Real> Mat67;
    typedef defaulttype::Mat<6,InDOFs,Real> Mat6xIn;
    typedef defaulttype::Mat<7,6,Real> Mat76;
    typedef vector<Mat76> VMat76;
    typedef defaulttype::Mat<8,3,Real> Mat83;
    typedef defaulttype::Mat<8,6,Real> Mat86;
    typedef vector<Mat86> VMat86;
    typedef defaulttype::Mat<8,8,Real> Mat88;
    typedef vector<Mat88> VMat88;
    typedef defaulttype::Mat<InDOFs,3,Real> MatInx3;

    typedef defaulttype::Vec<3,Real> Vec3;
    typedef vector<Vec3> VVec3;
    typedef vector<VVec3> VVVec3;
    typedef defaulttype::Vec<4,Real> Vec4;
    typedef defaulttype::Vec<6,Real> Vec6;
    typedef vector<Vec6> VVec6;
    typedef vector<VVec6> VVVec6;
    typedef defaulttype::Vec<8,Real> Vec8;
    typedef defaulttype::Vec<9,Real> Vec9;
    typedef defaulttype::Vec<InDOFs,Real> VecIn;
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
#else
    typedef Coord GeoCoord;
    typedef VecCoord GeoVecCoord;
#endif
    typedef defaulttype::StdRigidTypes<N,InReal> RigidType;
#ifdef SOFA_DEV
    typedef defaulttype::StdAffineTypes<N,InReal> AffineType;
    typedef defaulttype::StdQuadraticTypes<N,InReal> QuadraticType;
#endif
protected:
    vector<Coord> initPos; // pos: point coord in the local reference frame of In[i].
    vector<Coord> rotatedPoints;

    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;

    Data<vector<int> > repartition;
    Data<VVD> weights;
    Data<SVector<SVector<GeoCoord> > > weightGradients;
    Data<unsigned int> nbRefs;
public:
    Data<bool> showBlendedFrame;
    Data<bool> showDefTensors;
    Data<bool> showDefTensorsValues;
    Data<double> showDefTensorScale;
    Data<unsigned int> showFromIndex;
    Data<bool> showDistancesValues;
    Data<bool> showWeights;
    Data<double> showGammaCorrection;
    Data<bool> showWeightsValues;
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
    inline void normalizeWeights();

public:
    SkinningMapping ( In* from, Out* to );
    virtual ~SkinningMapping();

    void init();

    void apply ( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in );

    void draw();
    void clear();

    // Weights
    void setWeightsToHermite();
    void setWeightsToInvDist();
    void setWeightsToLinear();
    inline void updateWeights();
    inline void getDistances( int xfromBegin);

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
        return weights.getValue();
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
    void removeFrame( const unsigned int index);
    void insertFrame( const Coord& pos, const Quat& rot, GeoVecCoord beginPointSet = GeoVecCoord(), double distMax = 0.0);
    bool inverseSkinning( InCoord& X0, InCoord& X, const InCoord& Xtarget);
    void computeWeight( VVD& w, VecVecCoord& dw, const Coord& x0);
    void updateDataAfterInsertion();
    inline void changeSettingsDueToInsertion();

protected:
    void precomputeMatrices();
    void getCov33 (Mat33& M, const Vec3& vec1, const Vec3& vec2) const;
    void QtoR(Mat33& M, const Quat& q) const;
    void ComputeL(Mat76& L, const Quat& q) const;
    void ComputeQ(Mat37& Q, const Quat& q, const Vec3& p) const;
    void ComputeMa(Mat33& M, const Quat& q) const;
    void ComputeMb(Mat33& M, const Quat& q) const;
    void ComputeMc(Mat33& M, const Quat& q) const;
    void ComputeMw(Mat33& M, const Quat& q) const;

    // Avoid multiple specializations
    inline void setInCoord( typename defaulttype::StdRigidTypes<N, InReal>::Coord& coord, const Coord& position, const Quat& rotation) const;
    inline void setInCoord( typename defaulttype::StdAffineTypes<N, InReal>::Coord& coord, const Coord& position, const Quat& rotation) const;
    inline void setInCoord( typename defaulttype::StdQuadraticTypes<N, InReal>::Coord& coord, const Coord& position, const Quat& rotation) const;
#endif
    inline void getLocalCoord( Coord& result, const typename defaulttype::StdRigidTypes<N, InReal>::Coord& inCoord, const Coord& coord) const;
#ifdef SOFA_DEV
    inline void getLocalCoord( Coord& result, const typename defaulttype::StdAffineTypes<N, InReal>::Coord& inCoord, const Coord& coord) const;
    inline void getLocalCoord( Coord& result, const typename defaulttype::StdQuadraticTypes<N, InReal>::Coord& inCoord, const Coord& coord) const;
#endif
    template<int InN, class InReal2, class TCoord>
    inline typename enable_if<Equal<typename defaulttype::StdRigidTypes<InN, InReal2>::Coord, TCoord> >::type _apply( typename Out::VecCoord& out, const sofa::helper::vector<typename defaulttype::StdRigidTypes<InN, InReal2>::Coord>& in);
#ifdef SOFA_DEV
    template<int InN, class InReal2, class TCoord>
    inline typename enable_if<Equal<typename defaulttype::StdAffineTypes<InN, InReal2>::Coord, TCoord> >::type _apply( typename Out::VecCoord& out, const sofa::helper::vector<typename defaulttype::StdAffineTypes<InN, InReal2>::Coord>& in);
    template<int InN, class InReal2, class TCoord>
    inline typename enable_if<Equal<typename defaulttype::StdQuadraticTypes<InN, InReal2>::Coord, TCoord> >::type _apply( typename Out::VecCoord& out, const sofa::helper::vector<typename defaulttype::StdQuadraticTypes<InN, InReal2>::Coord>& in);
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


#ifdef SOFA_DEV
///////////////////////////////////////////////////////////////////////////////
//                           Affine Specialization                         //
///////////////////////////////////////////////////////////////////////////////

using sofa::defaulttype::Affine3dTypes;
using sofa::defaulttype::Affine3fTypes;

#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3dTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Affine3dTypes>, MappedModel<Vec3dTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Affine3dTypes>, MappedModel<ExtVec3dTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Affine3dTypes>, MappedModel<ExtVec3fTypes> > >;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Affine3fTypes>, MechanicalState<Vec3fTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Affine3fTypes>, MappedModel<Vec3fTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Affine3fTypes>, MappedModel<ExtVec3dTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Affine3fTypes>, MappedModel<ExtVec3fTypes> > >;
#endif //SOFA_DOUBLE
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Affine3dTypes>, MechanicalState<Vec3fTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Affine3fTypes>, MechanicalState<Vec3dTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Affine3dTypes>, MappedModel<Vec3fTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Affine3fTypes>, MappedModel<Vec3dTypes> > >;
#endif //SOFA_DOUBLE
#endif //SOFA_FLOAT


///////////////////////////////////////////////////////////////////////////////
//                          Quadratic Specialization                         //
///////////////////////////////////////////////////////////////////////////////

using sofa::defaulttype::Quadratic3dTypes;
using sofa::defaulttype::Quadratic3fTypes;

#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Quadratic3dTypes>, MechanicalState<Vec3dTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Quadratic3dTypes>, MappedModel<Vec3dTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Quadratic3dTypes>, MappedModel<ExtVec3dTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Quadratic3dTypes>, MappedModel<ExtVec3fTypes> > >;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Quadratic3fTypes>, MechanicalState<Vec3fTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Quadratic3fTypes>, MappedModel<Vec3fTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Quadratic3fTypes>, MappedModel<ExtVec3dTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Quadratic3fTypes>, MappedModel<ExtVec3fTypes> > >;
#endif //SOFA_DOUBLE
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Quadratic3dTypes>, MechanicalState<Vec3fTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< MechanicalMapping< MechanicalState<Quadratic3fTypes>, MechanicalState<Vec3dTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Quadratic3dTypes>, MappedModel<Vec3fTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Mapping< State<Quadratic3fTypes>, MappedModel<Vec3dTypes> > >;
#endif //SOFA_DOUBLE
#endif //SOFA_FLOAT
#endif // SOFA_DEV
#endif //defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_CPP)



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
