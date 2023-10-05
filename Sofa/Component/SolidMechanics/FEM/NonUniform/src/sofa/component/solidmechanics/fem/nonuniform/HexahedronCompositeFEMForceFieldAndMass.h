/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/component/solidmechanics/fem/nonuniform/config.h>


#include <sofa/component/solidmechanics/fem/nonuniform/NonUniformHexahedronFEMForceFieldAndMass.h>



// for memory :
// HEXA :
//
// 	     7----6
//      /|   /|
// 	   3----2 |
//     | 4--|-5
//     |/   |/
//     0----1


namespace sofa::component::solidmechanics::fem::nonuniform
{

template<class DataTypes>
class HexahedronCompositeFEMForceFieldAndMass : public NonUniformHexahedronFEMForceFieldAndMass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(HexahedronCompositeFEMForceFieldAndMass,DataTypes), SOFA_TEMPLATE(NonUniformHexahedronFEMForceFieldAndMass,DataTypes));
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;


    typedef NonUniformHexahedronFEMForceFieldAndMass<DataTypes> NonUniformHexahedronFEMForceFieldAndMassT;
    typedef component::solidmechanics::fem::elastic::HexahedronFEMForceFieldAndMass<DataTypes> HexahedronFEMForceFieldAndMassT;
    typedef component::solidmechanics::fem::elastic::HexahedronFEMForceField<DataTypes> HexahedronFEMForceFieldT;

    typedef typename NonUniformHexahedronFEMForceFieldAndMassT::ElementStiffness ElementStiffness;
    typedef typename NonUniformHexahedronFEMForceFieldAndMassT::MaterialStiffness MaterialStiffness;
    typedef typename NonUniformHexahedronFEMForceFieldAndMassT::MassT MassT;
    typedef typename NonUniformHexahedronFEMForceFieldAndMassT::ElementMass ElementMass;


    typedef typename NonUniformHexahedronFEMForceFieldAndMassT::VecElement VecElement;

    typedef type::Mat<8*3, 8*3, Real> Weight;

    using Index = sofa::Index;



protected:

    HexahedronCompositeFEMForceFieldAndMass()
        : HexahedronFEMForceFieldAndMassT()
        , d_finestToCoarse(initData(&d_finestToCoarse,false,"finestToCoarse","Does the homogenization is done directly from the finest level to the coarse one?"))
        , d_homogenizationMethod(initData(&d_homogenizationMethod,0,"homogenizationMethod","0->static, 1->constrained static, 2->modal analysis"))
        , d_completeInterpolation(initData(&d_completeInterpolation,false,"completeInterpolation","Is the non-linear, complete interpolation used?"))
        , d_useRamification(initData(&d_useRamification,true,"useRamification","If SparseGridRamification, are ramifications taken into account?"))
        , d_drawType(initData(&d_drawType,0,"drawType",""))
        , d_drawColor(initData(&d_drawColor,0,"drawColor",""))
        , d_drawSize(initData(&d_drawSize,(float)-1.0,"drawSize",""))
    {
    }

public:

    void init() override;
    void reinit() override;
    void draw(const core::visual::VisualParams* vparams) override;


    Data<bool> d_finestToCoarse; ///< Does the homogenization is done directly from the finest level to the coarse one?
    Data<int> d_homogenizationMethod; ///< 0->static, 1->constrained static, 2->modal analysis
    Data<bool> d_completeInterpolation; ///< Is the non-linear, complete interpolation used?
    Data<bool> d_useRamification; ///< If SparseGridRamification, are ramifications taken into account?
    Data<int> d_drawType;
    Data<int> d_drawColor;
    Data<float> d_drawSize;


    void findFinestChildren( type::vector<Index>& finestChildren, const Index elementIndice,  int level=0);
    void computeMechanicalMatricesDirectlyFromTheFinestToCoarse( ElementStiffness &K, ElementMass &M, const Index elementIndice);
    void computeMechanicalMatricesRecursively( ElementStiffness &K, ElementMass &M, const Index elementIndice,  int level);
    void computeMechanicalMatricesRecursivelyWithRamifications( ElementStiffness &K, ElementMass &M, const Index elementIndice,  int level);

    /// multiply all weights for all levels and go to the finest level to obtain the final weights from the coarsest to the finest directly
    void computeFinalWeights( const Weight &W, const Index coarseElementIndice, const Index elementIndice,  int level);
    void computeFinalWeightsRamification( const Weight &W, const Index coarseElementIndice, const Index elementIndice,  int level);


    // surcharge NonUniformHexahedronFEMForceFieldAndMass::computeMechanicalMatricesByCondensation
    void computeMechanicalMatricesByCondensation( ) override;



    type::vector< type::vector<Weight> > _weights;
    type::vector< std::pair<Index, Weight> > _finalWeights; // for each fine element -> the coarse element idx and corresponding Weight

protected:


    static const int FineHexa_FineNode_IndiceForAssembling[8][8]; // give an assembled idx for each node or each fine element
    static const int FineHexa_FineNode_IndiceForCutAssembling_27[27];// give an cutted assembled idx for each node or each fine element, if constrained -> idx in Kg, if not constrained -> idx in Kf

    static const int CoarseToFine[8]; // from a coarse node idx -> give the idx of the same node in the fine pb

    static const bool IS_CONSTRAINED_27[27]; // is the ith assembled vertices constrained?

    static const int WEIGHT_MASK[27*3][8*3];
    static const int WEIGHT_MASK_CROSSED[27*3][8*3];
    static const int WEIGHT_MASK_CROSSED_DIFF[27*3][8*3];
    static  const float MIDDLE_INTERPOLATION[27][8];
    static  const int MIDDLE_AXES[27];
    static const int FINE_ELEM_IN_COARSE_IN_ASS_FRAME[8][8];

    static const float RIGID_STIFFNESS[8*3][8*3];

};

#if !defined(SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONCOMPOSITEFEMFORCEFIELDANDMASS_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_NONUNIFORM_API HexahedronCompositeFEMForceFieldAndMass<defaulttype::Vec3Types>;

#endif

} // namespace sofa::component::solidmechanics::fem::nonuniform
