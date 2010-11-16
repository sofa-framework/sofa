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

#include <sofa/core/Mapping.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/helper/SVector.h>

#include <vector>

#include <sofa/component/component.h>
#include <sofa/helper/OptionsGroup.h>

#ifdef SOFA_DEV
#include <sofa/component/topology/HexahedronGeodesicalDistance.h>
#include "FrameStorage.h"
#include <../applications/plugins/frame/AffineTypes.h>
//#include <../applications/plugins/frame/ElastonTypes.h>
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
#define DISTANCE_STIFFNESS_DIFFUSION 3
#define DISTANCE_HARMONIC_STIFFNESS 4

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


template <class TIn, class TOut>
#ifdef SOFA_DEV
class SkinningMapping : public core::Mapping<TIn, TOut>, public FrameStorage<TIn, typename TIn::Real>
#else
class SkinningMapping : public core::Mapping<TIn, TOut>
#endif
{
public:
#ifdef SOFA_DEV
    SOFA_CLASS2(SOFA_TEMPLATE2(SkinningMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut), SOFA_TEMPLATE2(FrameStorage,TIn, typename TIn::Real));
#else
    SOFA_CLASS(SOFA_TEMPLATE2(SkinningMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));
#endif

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef Out DataTypes;

    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real Real;


    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real InReal;

    enum { N=DataTypes::spatial_dimensions };
    //enum { InDerivDim=In::DataTypes::deriv_total_size };
#ifdef SOFA_DEV
    enum { InDOFs=sofa::frame::DataTypesInfo<In::spatial_dimensions, InReal, In>::degrees_of_freedom };
    enum { InAt=sofa::frame::DataTypesInfo<In::spatial_dimensions, InReal, In>::Atilde_nb_column };
#else
    enum { InDOFs=In::deriv_total_size };
    enum { InAt=0 };
#endif
    typedef defaulttype::Mat<N,N,InReal> Mat;
    typedef defaulttype::Mat<3,3,InReal> Mat33;
    typedef vector<Mat33> VMat33;
    typedef vector<VMat33> VVMat33;
    typedef defaulttype::Mat<3,InDOFs,InReal> Mat3xIn;
    typedef vector<Mat3xIn> VMat3xIn;
    typedef vector<VMat3xIn> VVMat3xIn;
    typedef defaulttype::Mat<InAt,3,InReal> MatInAtx3;
    typedef vector<MatInAtx3> VMatInAtx3;
    typedef vector<VMatInAtx3> VVMatInAtx3;
    typedef defaulttype::Mat<3,6,InReal> Mat36;
    typedef vector<Mat36> VMat36;
    typedef vector<VMat36> VVMat36;
    typedef defaulttype::Mat<3,7,InReal> Mat37;
    typedef defaulttype::Mat<3,8,InReal> Mat38;
    typedef defaulttype::Mat<3,9,InReal> Mat39;
    typedef defaulttype::Mat<4,3,InReal> Mat43;
    typedef vector<Mat43> VMat43;
    typedef defaulttype::Mat<4,4,InReal> Mat44;
    typedef defaulttype::Mat<6,3,InReal> Mat63;
    typedef defaulttype::Mat<6,6,InReal> Mat66;
    typedef vector<Mat66> VMat66;
    typedef vector<VMat66> VVMat66;
    typedef defaulttype::Mat<6,7,InReal> Mat67;
    typedef defaulttype::Mat<6,InDOFs,InReal> Mat6xIn;
    typedef defaulttype::Mat<7,6,InReal> Mat76;
    typedef vector<Mat76> VMat76;
    typedef defaulttype::Mat<8,3,InReal> Mat83;
    typedef defaulttype::Mat<8,6,InReal> Mat86;
    typedef vector<Mat86> VMat86;
    typedef defaulttype::Mat<8,8,InReal> Mat88;
    typedef vector<Mat88> VMat88;
    typedef defaulttype::Mat<InDOFs,3,InReal> MatInx3;
    typedef defaulttype::Mat<6,10,InReal> Mat610;
    typedef vector<Mat610> VMat610;
    typedef sofa::helper::fixed_array<Mat610,InDOFs> MatInx610;
    typedef vector<MatInx610> VMatInx610;
    typedef vector<VMatInx610> VVMatInx610;
    typedef defaulttype::Vec<3,InReal> Vec3;
    typedef vector<Vec3> VVec3;
    typedef vector<VVec3> VVVec3;
    typedef defaulttype::Vec<4,InReal> Vec4;
    typedef defaulttype::Vec<6,InReal> Vec6;
    typedef vector<Vec6> VVec6;
    typedef vector<VVec6> VVVec6;
    typedef defaulttype::Vec<7,InReal> Vec7;
    typedef defaulttype::Vec<8,InReal> Vec8;
    typedef defaulttype::Vec<9,InReal> Vec9;
    typedef defaulttype::Vec<InDOFs,InReal> VecIn;
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
//  typedef defaulttype::StdElastonTypes<Coord, Deriv, Real> ElastonType;
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
    Data<bool> useElastons;
#endif

protected:
    Data<sofa::helper::OptionsGroup> wheightingType;
    Data<sofa::helper::OptionsGroup> distanceType;
    bool computeWeights;
    VVD distances;
    VVMat33 weightGradients2;

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
    SkinningMapping (core::State<In>* from, core::State<Out>* to );
    virtual ~SkinningMapping();

    void init();

    void apply(typename Out::VecCoord& out, const typename In::VecCoord& in);
    void applyJ(typename Out::VecDeriv& out, const typename In::VecDeriv& in);
    void applyJT(typename In::VecDeriv& out, const typename Out::VecDeriv& in);
    void applyJT(typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in);

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
    unsigned int getNbRefs() const
    {
        return nbRefs.getValue();
    }
    const VVD& getWeightCoefs() const
    {
        return weights.getValue();
    }
    const vector<int>& getRepartition() const
    {
        return repartition.getValue();
    }
    bool getComputeWeights() const
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
    void M33toV6(Vec6 &v,const Mat33& M) const;
    void QtoR(Mat33& M, const Quat& q) const;
    void ComputeL(Mat76& L, const Quat& q) const;
    void ComputeQ(Mat37& Q, const Quat& q, const Vec3& p) const;
    void ComputeMa(Mat33& M, const Quat& q) const;
    void ComputeMb(Mat33& M, const Quat& q) const;
    void ComputeMc(Mat33& M, const Quat& q) const;
    void ComputeMw(Mat33& M, const Quat& q) const;
    void StrainDeriv_rigid(Mat33 Ma,Mat33 Mb,Mat33 Mc,Mat33 Mw,Vec3 dw,Mat33 At,Mat33 F,Mat67 &B) const;
    void StrainDeriv_affine(Vec3 dw,MatInAtx3 At,Mat33 F,Mat6xIn &B) const;
    void StrainDeriv_quadratic(Vec3 dw,MatInAtx3 At,Mat33 F,Mat6xIn &B) const;

    // Avoid multiple specializations
    inline void setInCoord( typename defaulttype::StdRigidTypes<N, InReal>::Coord& coord, const Coord& position, const Quat& rotation) const;
    inline void setInCoord( typename defaulttype::StdAffineTypes<N, InReal>::Coord& coord, const Coord& position, const Quat& rotation) const;
    inline void setInCoord( typename defaulttype::StdQuadraticTypes<N, InReal>::Coord& coord, const Coord& position, const Quat& rotation) const;
#endif
    inline void getLocalCoord( Coord& result, const typename defaulttype::StdRigidTypes<N, InReal>::Coord& inCoord, const Coord& coord) const;
#ifdef SOFA_DEV
    inline void getLocalCoord( Coord& result, const typename defaulttype::StdAffineTypes<N, InReal>::Coord& inCoord, const Coord& coord) const;
    inline void getLocalCoord( Coord& result, const typename defaulttype::StdQuadraticTypes<N, InReal>::Coord& inCoord, const Coord& coord) const;

    template<class T>
    inline typename enable_if<Equal<RigidType, T> >::type precomputeMatrices(const RigidType&); // Useless parameter here to be compatible with gcc-4.0
    template<class T>
    inline typename enable_if<Equal<AffineType, T> >::type precomputeMatrices(const AffineType&); // Useless parameter here to be compatible with gcc-4.0
    template<class T>
    inline typename enable_if<Equal<QuadraticType, T> >::type precomputeMatrices(const QuadraticType&); // Useless parameter here to be compatible with gcc-4.0
#endif


    // Samples (default)
    template<class TCoord>
    inline typename enable_if<Equal<typename RigidType::Coord, TCoord> >::type _apply( typename Out::VecCoord& out, const sofa::helper::vector<typename RigidType::Coord>& in);
#ifdef SOFA_DEV
    template<class TCoord>
    inline typename enable_if<Equal<typename AffineType::Coord, TCoord> >::type _apply( typename Out::VecCoord& out, const sofa::helper::vector<typename AffineType::Coord>& in);
    template<class TCoord>
    inline typename enable_if<Equal<typename QuadraticType::Coord, TCoord> >::type _apply( typename Out::VecCoord& out, const sofa::helper::vector<typename QuadraticType::Coord>& in);
#endif


    template<class TDeriv>
    inline typename enable_if<Equal<typename RigidType::Deriv, TDeriv> >::type _applyJ( typename Out::VecDeriv& out, const sofa::helper::vector<typename RigidType::Deriv>& in);
#ifdef SOFA_DEV
    template<class TDeriv>
    inline typename enable_if<Equal<typename AffineType::Deriv, TDeriv> >::type _applyJ( typename Out::VecDeriv& out, const sofa::helper::vector<typename AffineType::Deriv>& in);
    template<class TDeriv>
    inline typename enable_if<Equal<typename QuadraticType::Deriv, TDeriv> >::type _applyJ( typename Out::VecDeriv& out, const sofa::helper::vector<typename QuadraticType::Deriv>& in);
#endif

    template<class TDeriv>
    inline typename enable_if<Equal<typename RigidType::Deriv, TDeriv> >::type _applyJT( sofa::helper::vector<typename RigidType::Deriv>& out, const typename Out::VecDeriv& in);
#ifdef SOFA_DEV
    template<class TDeriv>
    inline typename enable_if<Equal<typename AffineType::Deriv, TDeriv> >::type _applyJT( sofa::helper::vector<typename AffineType::Deriv>& out, const typename Out::VecDeriv& in);
    template<class TDeriv>
    inline typename enable_if<Equal<typename QuadraticType::Deriv, TDeriv> >::type _applyJT( sofa::helper::vector<typename QuadraticType::Deriv>& out, const typename Out::VecDeriv& in);
#endif

    template<class TMatrixDeriv>
    inline typename enable_if<Equal<typename RigidType::MatrixDeriv, TMatrixDeriv> >::type _applyJT_Matrix( typename RigidType::MatrixDeriv& out, const typename Out::MatrixDeriv& in);
#ifdef SOFA_DEV
    template<class TMatrixDeriv>
    inline typename enable_if<Equal<typename AffineType::MatrixDeriv, TMatrixDeriv> >::type _applyJT_Matrix( typename AffineType::MatrixDeriv& out, const typename Out::MatrixDeriv& in);
    template<class TMatrixDeriv>
    inline typename enable_if<Equal<typename QuadraticType::MatrixDeriv, TMatrixDeriv> >::type _applyJT_Matrix( typename QuadraticType::MatrixDeriv& out, const typename Out::MatrixDeriv& in);
#endif
};

using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec3fTypes;
using sofa::defaulttype::Rigid3dTypes;
using sofa::defaulttype::Rigid3fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Rigid3dTypes, Vec3dTypes >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Rigid3dTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Rigid3fTypes, Vec3fTypes >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Rigid3fTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Rigid3dTypes, Vec3fTypes >;
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Rigid3fTypes, Vec3dTypes >;
#endif
#endif


#ifdef SOFA_DEV
///////////////////////////////////////////////////////////////////////////////
//                           Affine Specialization                           //
///////////////////////////////////////////////////////////////////////////////

using sofa::defaulttype::Affine3dTypes;
using sofa::defaulttype::Affine3fTypes;

#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Affine3dTypes, Vec3dTypes >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Affine3dTypes, ExtVec3fTypes >;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Affine3fTypes, Vec3fTypes >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Affine3fTypes, ExtVec3fTypes >;
#endif //SOFA_DOUBLE
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Affine3dTypes, Vec3fTypes >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Affine3fTypes, Vec3dTypes >;
#endif //SOFA_DOUBLE
#endif //SOFA_FLOAT


///////////////////////////////////////////////////////////////////////////////
//                          Quadratic Specialization                         //
///////////////////////////////////////////////////////////////////////////////

using sofa::defaulttype::Quadratic3dTypes;
using sofa::defaulttype::Quadratic3fTypes;

#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Quadratic3dTypes, Vec3dTypes >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Quadratic3dTypes, ExtVec3fTypes >;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Quadratic3fTypes, Vec3fTypes >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Quadratic3fTypes, ExtVec3fTypes >;
#endif //SOFA_DOUBLE
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Quadratic3dTypes, Vec3fTypes >;
//extern template class SOFA_COMPONENT_MAPPING_API SkinningMapping< Quadratic3fTypes, Vec3dTypes >;
#endif //SOFA_DOUBLE
#endif //SOFA_FLOAT
#endif // SOFA_DEV

#endif //defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_CPP)



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
