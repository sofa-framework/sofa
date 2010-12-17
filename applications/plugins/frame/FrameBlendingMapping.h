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
#include "FrameMass.h"
#include "MappingTypes.h"
#include "NewMaterial.h"
#include "GridMaterial.h"
#include "DeformationGradientTypes.h"
#include <sofa/core/Mapping.h>

#include "QuadraticTypes.h"
#include "AffineTypes.h"
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/SVector.h>
#include <sofa/component/component.h>
#include <sofa/helper/OptionsGroup.h>
#include <vector>


namespace sofa
{

namespace component
{

namespace mapping
{

using defaulttype::Vec;
using helper::vector;
using sofa::component::material::MaterialTypes;
using sofa::component::material::GridMaterial;
using defaulttype::FrameMass;
using defaulttype::DeformationGradient;
using defaulttype::CStrain;


template<class Coord> struct MaterialTraits
{
    typedef typename Coord::VecMaterialCoord VecMaterialCoord;
};

template<class R>
struct MaterialTraits< Vec<3,R> >
{
    typedef vector<Vec<3,R> > VecMaterialCoord;
};


template<class TOut>
class SampleData : public  virtual core::objectmodel::BaseObject
{
public:
    // Output types
    typedef TOut Out;
    typedef typename MaterialTraits<typename Out::Coord>::VecMaterialCoord VecMaterialCoord;

    Data<VecMaterialCoord> f_materialPoints;

    SampleData();

};


/** Linear blend skinning, from a variety of input types to a variety of output types.
 */
template <class TIn, class TOut>
class FrameBlendingMapping : public core::Mapping<TIn, TOut>, public SampleData<TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(FrameBlendingMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;

    // Input types
    typedef TIn In;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real InReal;

    // Output types
    typedef TOut Out;
    static const unsigned num_spatial_dimensions=Out::spatial_dimensions;
    typedef typename Out::VecCoord VecOutCoord;
    typedef typename Out::VecDeriv VecOutDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real OutReal;

    // Material types
    typedef MaterialTypes<num_spatial_dimensions,InReal> materialType;
    typedef GridMaterial<materialType> GridMat;
    static const unsigned num_material_dimensions = GridMat::num_material_dimensions;
    typedef typename GridMat::Coord  MaterialCoord;
    typedef vector<MaterialCoord>  VecMaterialCoord;
    typedef typename GridMat::SCoord  SpatialCoord;
    typedef typename GridMat::SGradient  MaterialDeriv;
    typedef typename GridMat::SHessian  MaterialMat;

    // Mass types
    enum {InVSize= defaulttype::InDataTypesInfo<In,InReal,num_spatial_dimensions>::VSize};
    typedef FrameMass<num_spatial_dimensions,InVSize,InReal> frameMassType;

    // Conversion types
    static const unsigned nbRef = GridMat::nbRef;
    typedef typename defaulttype::LinearBlendTypes<In,Out,GridMat,nbRef, defaulttype::OutDataTypesInfo<Out,OutReal,num_spatial_dimensions>::primitive_order > InOut;


public:
    FrameBlendingMapping (core::State<In>* from, core::State<Out>* to );
    virtual ~FrameBlendingMapping();

    void init();

    void draw();

    void apply(typename Out::VecCoord& out, const typename In::VecCoord& in);
    void applyJ(typename Out::VecDeriv& out, const typename In::VecDeriv& in);
    void applyJT(typename In::VecDeriv& out, const typename Out::VecDeriv& in);
    void applyJT(typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in);

    inline void findIndexInRepartition( bool& influenced, unsigned int& realIndex, const unsigned int& pointIndex, const unsigned int& frameIndex);

protected:
    inline void initSamples();
    inline void initFrames();
    inline void updateWeights ();
    inline void normalizeWeights();
    inline void LumpMassesToFrames ();
    inline void LumpVolumes ();

    vector<InOut> inout;  ///< Data specific to the conversion between the types
    vector<frameMassType> frameMass;
//                vector<Vec<35,InReal> > sampleIntegVector;


    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;

    Data<VecOutCoord> f_initPos;            // initial child coordinates in the world reference frame
    Data< vector<Vec<nbRef,unsigned> > > f_index;   ///< The numChildren * numRefs column indices. index[j][i] is the index of the j-th parent influencing child i.

    Data< vector<Vec<nbRef,InReal> > >       weight;
    Data< vector<Vec<nbRef,MaterialCoord> > > weightDeriv;
    Data< vector<Vec<nbRef,MaterialMat> > >   weightDeriv2;




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

    Data<bool> useElastons; // temporary

    GridMaterial< materialType>* gridMaterial;
    Data<unsigned int> targetFrameNumber;
    Data<unsigned int> targetSampleNumber;
};

using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec3fTypes;
using sofa::defaulttype::Affine3dTypes;
using sofa::defaulttype::Affine3fTypes;
using sofa::defaulttype::Quadratic3dTypes;
using sofa::defaulttype::Quadratic3fTypes;
using sofa::defaulttype::Rigid3dTypes;
using sofa::defaulttype::Rigid3fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, Vec3dTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, ExtVec3dTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient331dTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient332dTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, Vec3dTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, ExtVec3dTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, DeformationGradient331dTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, DeformationGradient332dTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, Vec3dTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, ExtVec3dTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, DeformationGradient331dTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, DeformationGradient332dTypes >;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
//extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, Vec3fTypes >;
//extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, ExtVec3fTypes >;
//extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, DeformationGradient332fTypes >;
#endif //SOFA_DOUBLE
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, Vec3fTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, ExtVec3fTypes >;
//extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, ExtVec3dTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, Vec3fTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, ExtVec3fTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, Vec3fTypes >;
extern template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, ExtVec3fTypes >;
#endif //SOFA_DOUBLE
#endif //SOFA_FLOAT
#endif //defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_CPP)


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
