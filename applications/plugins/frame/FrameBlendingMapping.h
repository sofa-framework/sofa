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


/** Linear blend skinning, from a variety of input types to a variety of output types.
 */
template <class TIn, class TOut>
class FrameBlendingMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(FrameBlendingMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;

    // Input types
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real InReal;

    // Output types
    typedef TOut Out;
    typedef Out DataTypes;
    enum { N=DataTypes::spatial_dimensions };
    typedef typename Out::VecCoord VecOutCoord;
    typedef typename Out::VecDeriv VecOutDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real OutReal;

    // Material types
    typedef MaterialTypes<N,InReal> materialType;
    typedef GridMaterial<materialType> GridMat;
    typedef typename GridMat::MaterialCoord  MaterialCoord;
    typedef typename GridMat::SpatialCoord  SpatialCoord;
    static const unsigned num_material_dimensions = GridMat::num_material_dimensions;
//                typedef typename GridMat::VecVec3  VecMaterialCoord;
    typedef typename GridMat::Mat33  MaterialMat;
//                typedef typename GridMat::VMat33  VecMaterialMatrix;

    // Conversion types
    typedef typename defaulttype::LinearBlendTypes<In,Out,MaterialCoord,MaterialMat> InOut;
    typedef typename InOut::JacobianBlock JacobianBlock;


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
    inline void updateWeights ();
    inline void normalizeWeights();


    VecInCoord mm0;  ///< product of the current matrices with the inverse of the initial matrices

    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;

    Data<unsigned>       f_nbRefs;  ///< Number of parents influencing each child.
    Data< vector<unsigned> > f_index;   ///< The numChildren * numRefs column indices. index[nbRefs*i+j] is the index of the j-th parent influencing child i.

    Data< vector<OutReal> >       weight;
    Data< vector<MaterialCoord> > weightDeriv;
    Data< vector<MaterialMat> >   weightDeriv2;

    Data<VecOutCoord> f_initPos;            // initial child coordinates in the world reference frame
    Data<VecInCoord> f_initialInverseMatrices; // inverses of the initial parent matrices in the world reference frame

    vector<JacobianBlock> J;  ///< The tangent operator used in applyJ and applyJT


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

    //enum { InDerivDim=In::DataTypes::deriv_total_size };
    //enum { InDOFs=In::deriv_total_size };
    //enum { InAt=0 };
    //typedef defaulttype::Mat<N,N,InReal> Mat;
    //    typedef defaulttype::Mat<3,3,InReal> Mat33;
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
    //    typedef defaulttype::Vec<6,InReal> Vec6;
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
};


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
