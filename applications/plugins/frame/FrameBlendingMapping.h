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
#include "MappingTypes.h"
#include "NewMaterial.h"
#include "GridMaterial.h"
#include <sofa/core/Mapping.h>
#include <sofa/component/component.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/core/loader/PrimitiveGroup.h>

#include "AffineTypes.h"
#include "QuadraticTypes.h"
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include "DeformationGradientTypes.h"


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
using defaulttype::FrameData;
using defaulttype::SampleData;
using sofa::component::topology::PointData;


/** Linear blend skinning, from a variety of input types to a variety of output types.
 */
template <class TIn, class TOut>
class FrameBlendingMapping : public core::Mapping<TIn, TOut>, public FrameData<TIn,(defaulttype::OutDataTypesInfo<TOut>::primitive_order > 0)>, public SampleData<TOut,(defaulttype::OutDataTypesInfo<TOut>::primitive_order > 0)>
{
public:
    SOFA_CLASS3(SOFA_TEMPLATE2(FrameBlendingMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut), SOFA_TEMPLATE2(FrameData,TIn,(defaulttype::OutDataTypesInfo<TOut>::primitive_order > 0)), SOFA_TEMPLATE2(SampleData,TOut,(defaulttype::OutDataTypesInfo<TOut>::primitive_order > 0)));

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
    static const unsigned int num_spatial_dimensions=Out::spatial_dimensions;
    typedef typename Out::VecCoord VecOutCoord;
    typedef typename Out::VecDeriv VecOutDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real OutReal;

    // Material types
    typedef MaterialTypes<num_spatial_dimensions,InReal> materialType;
    typedef GridMaterial<materialType> GridMat;
    static const unsigned int num_material_dimensions = GridMat::num_material_dimensions;
    typedef typename GridMat::Coord  MaterialCoord;
    typedef vector<MaterialCoord>  VecMaterialCoord;
    typedef typename GridMat::SCoord  SpatialCoord;
    typedef typename GridMat::SGradient  MaterialDeriv;
    typedef typename GridMat::SHessian  MaterialMat;

    // Mass types
    enum {InVSize= defaulttype::InDataTypesInfo<In>::VSize};
    typedef FrameData<TIn,(defaulttype::OutDataTypesInfo<TOut>::primitive_order > 0)> FData;
    typedef typename FData::FrameMassType FrameMassType;
    typedef typename FData::VecMass VecMass;
    typedef typename FData::MassVector MassVector;
    typedef SampleData<TOut,(defaulttype::OutDataTypesInfo<TOut>::primitive_order > 0)> SData;

    // Conversion types
    static const unsigned int nbRef = GridMat::nbRef;
    typedef typename defaulttype::LinearBlendTypes<In,Out,GridMat,nbRef, defaulttype::OutDataTypesInfo<Out>::type > InOut;
    typedef typename defaulttype::DualQuatBlendTypes<In,Out,GridMat,nbRef, defaulttype::OutDataTypesInfo<Out>::type > DQInOut;
    typedef defaulttype::BaseFrameBlendingMapping<true> PhysicalMapping;

    typedef Vec<3,double> Vec3d;

public:
    FrameBlendingMapping (core::State<In>* from, core::State<Out>* to );
    virtual ~FrameBlendingMapping();

    virtual void init();

    virtual void draw(const core::visual::VisualParams* vparams);

    virtual void apply( InCoord& coord, const InCoord& restCoord);
    virtual void apply( typename SData::MaterialCoord& coord, const typename SData::MaterialCoord& restCoord);
    virtual void apply(typename Out::VecCoord& out, const typename In::VecCoord& in);
    virtual void applyJ(typename Out::VecDeriv& out, const typename In::VecDeriv& in);
    virtual void applyJT(typename In::VecDeriv& out, const typename Out::VecDeriv& in);
    virtual void applyJT(typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in);

    inline void findIndexInRepartition( bool& influenced, unsigned int& realIndex, const unsigned int& pointIndex, const unsigned int& frameIndex);

    // Adaptativity
    virtual void checkForChanges();
    virtual void handleTopologyChange(core::topology::Topology* t);
    virtual bool insertFrame( const Vec3d& pos);
    virtual void removeFrame( const unsigned int index); // Index is relative to addedFrameIndices

protected:
    bool inverseApply( InCoord& restCoord, InCoord& coord, const InCoord& targetCoord);
    inline void initFrames (const bool& setFramePos = true, const bool& updateFramePosFromOldOne = false);
    inline void initSamples();
    Data<bool> useLinearWeights;
    inline void updateWeights ();
    inline void normalizeWeights();
    virtual void LumpMassesToFrames (MassVector& f_mass0, MassVector& f_mass);

    // Adaptativity
    PhysicalMapping* physicalMapping;
    virtual void updateMapping(const bool& computeWeights = false);

    PointData<sofa::helper::vector<InOut> > inout;  ///< Data specific to the conversion between the types
    PointData<sofa::helper::vector<DQInOut> > dqinout;  ///< Data specific to the conversion between the types
    Data<bool> useDQ;  // use dual quat blending instead of linear blending
    Data<bool> useAdaptivity;  // use automatic adaptation of frames and samples


    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;

    PointData<sofa::helper::vector<OutCoord> > f_initPos;            // initial child coordinates in the world reference frame
    PointData<sofa::helper::vector<Vec<nbRef,unsigned int> > > f_index;   ///< The numChildren * numRefs column indices. index[i][j] is the index of the j-th parent influencing child i.
    PointData<sofa::helper::vector<unsigned int> > f_groups;            // child group for restricting interpolation (initialized from trianglegroupes)

    PointData<sofa::helper::vector<Vec<nbRef,InReal> > >       weight;
    PointData<sofa::helper::vector<Vec<nbRef,MaterialCoord> > > weightDeriv;
    PointData<sofa::helper::vector<Vec<nbRef,MaterialMat> > >   weightDeriv2;

    core::topology::BaseMeshTopology* to_topo; // Used to manage topological changes


public:
    Data<bool> showBlendedFrame;
    Data<unsigned int> showFromIndex;
    Data<bool> showWeights;
    Data<double> showGammaCorrection;
    Data<bool> showWeightsValues;
    Data<bool> showReps;
    Data<int> showValuesNbDecimals;
    Data<double> showTextScaleFactor;
    Data<bool> showGradients;
    Data<bool> showGradientsValues;
    Data<double> showGradientsScaleFactor;
    Data<bool> showStrain;
    Data<double> showStrainScaleFactor;
    Data<bool> showDetF;
    Data<double> showDetFScaleFactor;
    Data<bool> isAdaptive;

    helper::vector< helper::fixed_array <unsigned int,3> > triangles; ///< Topology of toModel (used for strain display)
    helper::vector< core::loader::PrimitiveGroup > trianglesGroups;  ///< triangle groups of toModel (used for restricting interpolation of a group to a label)

    GridMaterial< materialType>* gridMaterial;
    Data<unsigned int> targetFrameNumber;
    Data<bool> initializeFramesInRigidParts;  ///< Automatically initialize frames in rigid parts if stiffness>15E6
    Data<unsigned int> targetSampleNumber;
    Data<vector<int> > restrictInterpolationToLabel;  ///< restrict interpolation to a certain label in the gridmaterial
};

using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec3fTypes;
using sofa::defaulttype::ExtVec3dTypes;
using sofa::defaulttype::Affine3dTypes;
using sofa::defaulttype::Affine3fTypes;
using sofa::defaulttype::Quadratic3dTypes;
using sofa::defaulttype::Quadratic3fTypes;
using sofa::defaulttype::Rigid3dTypes;
using sofa::defaulttype::Rigid3fTypes;
using sofa::defaulttype::DeformationGradient331dTypes;
using sofa::defaulttype::DeformationGradient331fTypes;
using sofa::defaulttype::DeformationGradient332fTypes;
using sofa::defaulttype::DeformationGradient332dTypes;


//#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_CPP)
//#pragma warning(disable : 4231)
//#ifndef SOFA_FLOAT
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, Vec3dTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, ExtVec3dTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient331dTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient332dTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, Vec3dTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, ExtVec3dTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, DeformationGradient331dTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, DeformationGradient332dTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, Vec3dTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, ExtVec3dTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, DeformationGradient331dTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, DeformationGradient332dTypes >;
//#endif //SOFA_FLOAT
//#ifndef SOFA_DOUBLE
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, Vec3fTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, ExtVec3fTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3fTypes, DeformationGradient332fTypes >;
//#endif //SOFA_DOUBLE
//#ifndef SOFA_FLOAT
//#ifndef SOFA_DOUBLE
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, ExtVec3fTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, ExtVec3fTypes >;
//            extern template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, ExtVec3fTypes >;
//#endif //SOFA_DOUBLE
//#endif //SOFA_FLOAT
//#endif //defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_CPP)


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
