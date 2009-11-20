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

#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <vector>
#include <sofa/component/component.h>
#ifdef SOFA_DEV
#include <sofa/helper/DualQuat.h>
#endif
namespace sofa
{

namespace component
{

namespace topology
{
template<class T>
class HexahedronGeodesicalDistance;
}

namespace mapping
{
using sofa::component::topology::HexahedronGeodesicalDistance;

typedef enum
{
    DISTANCE_EUCLIDIAN, DISTANCE_GEODESIC
} DistanceType;

typedef enum
{
    WEIGHT_LINEAR, WEIGHT_INVDIST_SQUARE, WEIGHT_HERMITE
} WeightingType;

typedef enum
{
    INTERPOLATION_LINEAR, INTERPOLATION_DUAL_QUATERNION
} InterpolationType;

template <class BasicMapping>
class SkinningMapping : public BasicMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SkinningMapping,BasicMapping), BasicMapping);
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename defaulttype::SparseConstraint<Deriv> OutSparseConstraint;
    typedef typename OutSparseConstraint::const_data_iterator OutConstraintIterator;

    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename Coord::value_type Real;
    enum { N=Coord::static_size };
    typedef defaulttype::Mat<N,N,Real> Mat;
    typedef defaulttype::Mat<8,8,Real> Mat88;
    typedef defaulttype::Mat<3,8,Real> Mat38;
    typedef defaulttype::Mat<8,6,Real> Mat86;
    typedef defaulttype::Mat<3,6,Real> Mat36;
    typedef defaulttype::Mat<6,1,Real> Mat61;
    typedef defaulttype::Mat<6,3,Real> Mat63;
    typedef defaulttype::Mat<3,1,Real> Mat31;
    typedef defaulttype::Mat<8,1,Real> Mat81;

#ifdef SOFA_DEV
    // These typedef are here to avoid compilation pb encountered with ResizableExtVect Type.
    typedef typename sofa::defaulttype::StdVectorTypes<sofa::defaulttype::Vec<N, Real>, sofa::defaulttype::Vec<N, Real>, Real> GeoType; // = Vec3fTypes or Vec3dTypes
    typedef typename GeoType::VecCoord GeoVecCoord;
    typedef typename helper::DualQuatd DualQuat;
#endif
protected:
    sofa::helper::vector<InCoord> initPosDOFs; // translation and rotation of the blended reference frame i, where i=0..n.
    sofa::helper::vector<Coord> initPos; // pos: point coord in the world reference frame
    sofa::helper::vector<Coord> rotatedPoints;

    core::componentmodel::behavior::BaseMechanicalState::ParticleMask* maskFrom;
    core::componentmodel::behavior::BaseMechanicalState::ParticleMask* maskTo;

    Data<sofa::helper::vector<unsigned int> > repartition;
    Data<sofa::helper::vector<double> > coefs;
    Data<sofa::helper::vector<double> > distances;
    Data<sofa::helper::vector<Coord> > distGradients;
    Data<unsigned int> nbRefs;
    Data<bool> displayBlendedFrame;
    Data<sofa::helper::vector<Mat36> > matJ;

    bool computeWeights;
    WeightingType wheighting;
    InterpolationType interpolation;
    DistanceType distance;
#ifdef SOFA_DEV
    /*					typename Out::VecCoord x1; //TODO remove after test
    					typename Out::VecCoord x2; //TODO remove after test
    					vector<DualQuat> q1; //TODO remove after test
    					vector<DualQuat> q2; //TODO remove after test
    					vector<Mat81> dqLi_previous, dqLi; //TODO to remove after the convergence test*/
    vector<Mat86> L;
    HexahedronGeodesicalDistance< GeoType>* geoDist;
#endif

    class Loader;
    void load ( const char* filename );
    inline void computeInitPos();
    inline void sortReferences();

public:
    SkinningMapping ( In* from, Out* to );
    virtual ~SkinningMapping();

    void init();
    void parse ( core::objectmodel::BaseObjectDescription* arg );

    void apply ( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT ( typename In::VecConst& out, const typename Out::VecConst& in );

    void draw();
    void clear();

    // Weights
    void setWeightsToHermite();
    void setWieghtsToInvDist();
    void setWeightsToLinear();
    inline void updateWeights();

    // Interpolations
    void setInterpolationToLinear();
    void setInterpolationToDualQuaternion();

    // Accessors
    void setNbRefs ( unsigned int nb ) { nbRefs.setValue ( nb ); }
    void setWeightCoefs ( sofa::helper::vector<double> &weights );
    void setRepartition ( sofa::helper::vector<unsigned int> &rep );
    void setComputeWeights ( bool val ) {computeWeights=val;}
    unsigned int getNbRefs() { return nbRefs.getValue(); }
    const sofa::helper::vector<double>& getWeightCoefs() { return coefs.getValue(); }
    const sofa::helper::vector<unsigned int>& getRepartition() { return repartition.getValue(); }
    bool getComputeWeights() { return computeWeights; }

#ifdef SOFA_DEV
    // Dual quat J matrix components
    void computeDqQ( Mat38& Q, const DualQuat& bn, const Coord& p);
    void computeDqN( Mat88& N, const DualQuat& bn, const DualQuat& b);
    void computeDqT( Mat88& T, const DualQuat& qi0);
    void computeDqL( Mat86& L, const DualQuat& qi, const Coord& ti);
#endif
};

using core::Mapping;
using core::componentmodel::behavior::MechanicalMapping;
using core::componentmodel::behavior::MappedModel;
using core::componentmodel::behavior::State;
using core::componentmodel::behavior::MechanicalState;

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
