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
#ifndef SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_H
#define SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_H

#include "initFrame.h"
#include "FrameStorage.h"
#include "NewMaterial.h"
#include "GridMaterial.h"
#include <sofa/core/Mapping.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/SVector.h>
#include <sofa/component/component.h>
#include <sofa/helper/OptionsGroup.h>
//#include <sofa/component/topology/DistanceOnGrid.h>
#include <vector>


namespace sofa
{

namespace component
{

namespace mapping
{

//#define SM_DISTANCE_EUCLIDIAN 0
//#define SM_DISTANCE_GEODESIC 1
//#define SM_DISTANCE_HARMONIC 2
//#define SM_DISTANCE_STIFFNESS_DIFFUSION 3
//#define SM_DISTANCE_HARMONIC_STIFFNESS 4
//
//#define WEIGHT_NONE 0
//#define WEIGHT_INVDIST_SQUARE 1
//#define WEIGHT_LINEAR 2
//#define WEIGHT_HERMITE 3
//#define WEIGHT_SPLINE 4
//using sofa::component::topology::DistanceOnGrid;

using defaulttype::Vec;
using helper::vector;
using helper::SVector;
using sofa::component::material::MaterialTypes;
using sofa::component::material::GridMaterial;

/** This class is not supposed to be instanciated. It is only used to contain what is common between its derived classes.
  */
template <class TIn, class TOut>
class FrameBlendingMapping : public core::Mapping<TIn, TOut>, public FrameStorage<TIn, typename TIn::Real>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(FrameBlendingMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut), SOFA_TEMPLATE2(FrameStorage,TIn,typename TIn::Real));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef Out DataTypes;
    enum { N=DataTypes::spatial_dimensions };

    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real Real;
    //typedef typename Out::SpatialCoord SpatialCoord;
    typedef Vec<N,Real> SpatialCoord;


    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real InReal;

    typedef SVector<InReal> VD;
    typedef SVector<VD> VVD;

    typedef SVector<SpatialCoord > VSpatialCoord;
    typedef SVector<VSpatialCoord > VVSpatialCoord;

    typedef defaulttype::Mat<N,N,Real> MatNN;
    typedef SVector<MatNN > VMatNN;
    typedef SVector<VMatNN > VVMatNN;

    typedef SVector<unsigned int> VUI;
    typedef SVector<VUI> VVUI;

    typedef MaterialTypes<N,InReal> materialType;
    typedef GridMaterial<materialType> GridMat;
    typedef typename GridMat::Vec3  materialCoordType;
    typedef typename GridMat::VecVec3  materialVecCoordType;
    typedef typename GridMat::Mat33  materialMat33Type;
    typedef typename GridMat::VMat33  materialVecMat33Type;

    //enum { InDerivDim=In::DataTypes::deriv_total_size };
    //enum { InDOFs=In::deriv_total_size };
    //enum { InAt=0 };
    //typedef defaulttype::Mat<N,N,InReal> Mat;
    typedef defaulttype::Mat<3,3,InReal> Mat33;
    //typedef vector<Mat33> VMat33;
    //typedef vector<VMat33> VVMat33;
    //typedef defaulttype::Mat<3,InDOFs,InReal> Mat3xIn;
    //typedef vector<Mat3xIn> VMat3xIn;
    //typedef vector<VMat3xIn> VVMat3xIn;
    //typedef defaulttype::Mat<InAt,3,InReal> MatInAtx3;
    //typedef vector<MatInAtx3> VMatInAtx3;
    //typedef vector<VMatInAtx3> VVMatInAtx3;
    //typedef defaulttype::Mat<3,6,InReal> Mat36;
    //typedef vector<Mat36> VMat36;
    //typedef vector<VMat36> VVMat36;
    //typedef defaulttype::Mat<3,7,InReal> Mat37;
    //typedef defaulttype::Mat<3,8,InReal> Mat38;
    //typedef defaulttype::Mat<3,9,InReal> Mat39;
    //typedef defaulttype::Mat<4,3,InReal> Mat43;
    //typedef vector<Mat43> VMat43;
    //typedef defaulttype::Mat<4,4,InReal> Mat44;
    //typedef defaulttype::Mat<6,3,InReal> Mat63;
    //typedef defaulttype::Mat<6,6,InReal> Mat66;
    //typedef vector<Mat66> VMat66;
    //typedef vector<VMat66> VVMat66;
    //typedef defaulttype::Mat<6,7,InReal> Mat67;
    //typedef defaulttype::Mat<6,InDOFs,InReal> Mat6xIn;
    //typedef defaulttype::Mat<7,6,InReal> Mat76;
    //typedef vector<Mat76> VMat76;
    //typedef defaulttype::Mat<8,3,InReal> Mat83;
    //typedef defaulttype::Mat<8,6,InReal> Mat86;
    //typedef vector<Mat86> VMat86;
    //typedef defaulttype::Mat<8,8,InReal> Mat88;
    //typedef vector<Mat88> VMat88;
    //typedef defaulttype::Mat<InDOFs,3,InReal> MatInx3;
    //typedef defaulttype::Mat<6,10,InReal> Mat610;
    //typedef vector<Mat610> VMat610;
    //typedef sofa::helper::fixed_array<Mat610,InDOFs> MatInx610;
    //typedef vector<MatInx610> VMatInx610;
    //typedef vector<VMatInx610> VVMatInx610;
    //typedef defaulttype::Vec<3,InReal> Vec3;
    //typedef vector<Vec3> VVec3;
    //typedef vector<VVec3> VVVec3;
    //typedef defaulttype::Vec<4,InReal> Vec4;
    typedef defaulttype::Vec<6,InReal> Vec6;
    //typedef vector<Vec6> VVec6;
    //typedef vector<VVec6> VVVec6;
    //typedef defaulttype::Vec<7,InReal> Vec7;
    //typedef defaulttype::Vec<8,InReal> Vec8;
    //typedef defaulttype::Vec<9,InReal> Vec9;
    //typedef defaulttype::Vec<InDOFs,InReal> VecIn;
    //typedef helper::Quater<InReal> Quat;
    //typedef sofa::helper::vector< VecCoord > VecVecCoord;
    //typedef SVector<double> VD;
    //typedef SVector<SVector<double> > VVD;

    // These typedef are here to avoid compilation pb encountered with ResizableExtVect Type.
    //typedef typename sofa::defaulttype::StdVectorTypes<sofa::defaulttype::Vec<N, double>, sofa::defaulttype::Vec<N, double>, double> DoGType; // = Vec3fTypes or Vec3dTypes
    //typedef typename DistanceOnGrid< DoGType >::VecCoord GeoVecCoord;
    //typedef typename DistanceOnGrid< DoGType >::Coord GeoCoord;
    //typedef typename DistanceOnGrid< DoGType >::VecVecCoord GeoVecVecCoord;

protected:
    VSpatialCoord initPos; // pos: point coord in the local reference frame of In[i].
    VSpatialCoord rotatedPoints;

    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;

    Data<unsigned int> nbRefs;	// Maximum number of primitives influencing each point.
    Data<VVUI > repartition; // Map of size nbOut*nbRefs from points to primitive indices.

    Data<VVD> weight;
    Data<VVSpatialCoord> weightDeriv;
    Data<VVMatNN> weightDeriv2;

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

    //DistanceOnGrid< DoGType>* distOnGrid;
    //Data<bool> enableSkinning;
    //Data<double> voxelVolume;
    Data<bool> useElastons; // temporary
    //GeoVecVecCoord distGradients;

    GridMaterial< materialType>* gridMaterial;
    Data<unsigned int> targetFrameNumber;
    Data<unsigned int> targetSampleNumber;

protected:
    //Data<sofa::helper::OptionsGroup> wheightingType;
    //Data<sofa::helper::OptionsGroup> distanceType;
    //bool computeWeights;
    //VVD distances;
    //VVMat33 weightGradients2;

    inline void computeInitPos();
    //inline void computeDistances();
    //inline void sortReferences( vector<unsigned int>& references);
    //inline void normalizeWeights();

public:
    FrameBlendingMapping (core::State<In>* from, core::State<Out>* to );
    virtual ~FrameBlendingMapping();

    void init();

    void draw();
    void clear();

    // Weights
    //void setWeightsToHermite();
    //void setWeightsToInvDist();
    //void setWeightsToLinear();
    //inline void updateWeights();
    //inline void getDistances( int xfromBegin);

    // Accessors
    void setNbRefs ( unsigned int nb )     {    this->nbRefs.setValue ( nb ); }
    unsigned int getNbRefs() const    {   return this->nbRefs.getValue();    }

    //void setRepartition ( vector<int> &rep );
    //void setWeightCoefs ( VVD& weights );
    //void setComputeWeights ( bool val )
    //{
    //    computeWeights=val;
    //}
    //const VVD& getWeightCoefs() const
    //{
    //    return weights.getValue();
    //}
    //const vector<unsigned int>& getRepartition() const
    //{
    //    return this->repartition.getValue();
    //}
    //bool getComputeWeights() const
    //{
    //    return computeWeights;
    //}
    inline void findIndexInRepartition( bool& influenced, unsigned int& realIndex, const unsigned int& pointIndex, const unsigned int& frameIndex);

protected:
    //void removeFrame( const unsigned int index);
    //void insertFrame( const Coord& pos, const Quat& rot, GeoVecCoord beginPointSet = GeoVecCoord(), double distMax = 0.0);
    //bool inverseSkinning( InCoord& X0, InCoord& X, const InCoord& Xtarget);
    //void computeWeight( VVD& w, VecVecCoord& dw, const Coord& x0);
    //void updateDataAfterInsertion();
    //inline void changeSettingsDueToInsertion();

    void M33toV6(Vec6 &v,const Mat33& M) const;

    inline void resizeMatrices();

    inline void initSamples();
    inline void initFrames();
    inline void updateWeights ();
    inline void normalizeWeights();


    // Avoid multiple specializations
    //inline void setInCoord( InCoord& coord, const Coord& position, const Quat& rotation) const;
    inline void getLocalCoord( Coord& result, const typename In::Coord& inCoord, const Coord& coord) const;

    virtual void precomputeMatrices() = 0;


};


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
