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
#include <sofa/component/topology/DistanceOnGrid.h>
//#include <../applications/plugins/frame/ElastonTypes.h>
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
using sofa::component::topology::DistanceOnGrid;
#endif

#define SM_DISTANCE_EUCLIDIAN 0
#define SM_DISTANCE_GEODESIC 1
#define SM_DISTANCE_HARMONIC 2
#define SM_DISTANCE_STIFFNESS_DIFFUSION 3
#define SM_DISTANCE_HARMONIC_STIFFNESS 4

#define WEIGHT_NONE 0
#define WEIGHT_INVDIST_SQUARE 1
#define WEIGHT_LINEAR 2
#define WEIGHT_HERMITE 3
#define WEIGHT_SPLINE 4


template <class TIn, class TOut>
class SkinningMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SkinningMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

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
    typedef defaulttype::Mat<3,3,InReal> Mat33;
    typedef vector<Mat33> VMat33;
    typedef vector<VMat33> VVMat33;
    typedef defaulttype::Vec<3,InReal> Vec3;
    typedef Quater<InReal> Quat;
    typedef SVector<SVector<double> > VVD;

#ifdef SOFA_DEV
    // These typedef are here to avoid compilation pb encountered with ResizableExtVect Type.
    typedef typename sofa::defaulttype::StdVectorTypes<sofa::defaulttype::Vec<N, double>, sofa::defaulttype::Vec<N, double>, double> DoGType; // = Vec3fTypes or Vec3dTypes
    typedef typename DistanceOnGrid< DoGType >::VecCoord GeoVecCoord;
    typedef typename DistanceOnGrid< DoGType >::Coord GeoCoord;
    typedef typename DistanceOnGrid< DoGType >::VecVecCoord GeoVecVecCoord;
#else
    typedef Coord GeoCoord;
    typedef VecCoord GeoVecCoord;
#endif

protected:
    vector<Coord> initPos; // pos: point coord in the local reference frame of In[i].
    vector<Coord> rotatedPoints;

    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;

    Data<unsigned int> nbRefs; // Number of primitives influencing each point.
    Data<vector<unsigned int> > repartition; // indices of primitives influencing each point.
    Data<VVD> weights;
    Data<SVector<SVector<GeoCoord> > > weightGradients;
public:
    Data<bool> showBlendedFrame;
    Data<unsigned int> showFromIndex;
    Data<bool> showDistancesValues;
    Data<bool> showWeights;
    Data<double> showGammaCorrection;
    Data<bool> showWeightsValues;
    Data<bool> showReps;
    Data<int> showValuesNbDecimals;
    Data<double> showTextScaleFactor;
#ifdef SOFA_DEV
    DistanceOnGrid< DoGType>* distOnGrid;
#endif
    Data<bool> showGradients;
    Data<bool> showGradientsValues;
    Data<double> showGradientsScaleFactor;

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
    inline void sortReferences( vector<unsigned int>& references);
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
        this->nbRefs.setValue ( nb );
    }
    void setRepartition ( vector<int> &rep );
    void setWeightCoefs ( VVD& weights );
    void setComputeWeights ( bool val )
    {
        computeWeights=val;
    }
    unsigned int getNbRefs() const
    {
        return this->nbRefs.getValue();
    }
    const VVD& getWeightCoefs() const
    {
        return weights.getValue();
    }
    const vector<unsigned int>& getRepartition() const
    {
        return this->repartition.getValue();
    }
    bool getComputeWeights() const
    {
        return computeWeights;
    }

    inline void reverseRepartition( bool& influenced, unsigned int& realIndex, const unsigned int& pointIndex, const unsigned int& frameIndex);

protected:
    inline void getLocalCoord( Coord& result, const typename In::Coord& inCoord, const Coord& coord) const;
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
#endif //defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_CPP)



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
