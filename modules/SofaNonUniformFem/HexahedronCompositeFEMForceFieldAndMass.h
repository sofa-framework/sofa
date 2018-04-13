/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONCOMPOSITEFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONCOMPOSITEFEMFORCEFIELD_H
#include "config.h"


#include <SofaNonUniformFem/NonUniformHexahedronFEMForceFieldAndMass.h>



// for memory :
// SOFA_NEW_HEXA :
//
// 	     7----6
//      /|   /|
// 	   3----2 |
//     | 4--|-5
//     |/   |/
//     0----1


namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
class HexahedronCompositeFEMForceFieldAndMass : public sofa::component::forcefield::NonUniformHexahedronFEMForceFieldAndMass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(HexahedronCompositeFEMForceFieldAndMass,DataTypes), SOFA_TEMPLATE(sofa::component::forcefield::NonUniformHexahedronFEMForceFieldAndMass,DataTypes));
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;


    typedef sofa::component::forcefield::NonUniformHexahedronFEMForceFieldAndMass<DataTypes> NonUniformHexahedronFEMForceFieldAndMassT;
    typedef sofa::component::forcefield::HexahedronFEMForceFieldAndMass<DataTypes> HexahedronFEMForceFieldAndMassT;
    typedef sofa::component::forcefield::HexahedronFEMForceField<DataTypes> HexahedronFEMForceFieldT;

    typedef typename NonUniformHexahedronFEMForceFieldAndMassT::ElementStiffness ElementStiffness;
    typedef typename NonUniformHexahedronFEMForceFieldAndMassT::MaterialStiffness MaterialStiffness;
    typedef typename NonUniformHexahedronFEMForceFieldAndMassT::MassT MassT;
    typedef typename NonUniformHexahedronFEMForceFieldAndMassT::ElementMass ElementMass;


    typedef typename NonUniformHexahedronFEMForceFieldAndMassT::VecElement VecElement;

    typedef defaulttype::Mat<8*3, 8*3, Real> Weight;




protected:

    HexahedronCompositeFEMForceFieldAndMass()
        : HexahedronFEMForceFieldAndMassT()
        , _finestToCoarse(initData(&_finestToCoarse,false,"finestToCoarse","Does the homogenization is done directly from the finest level to the coarse one?"))
        , _homogenizationMethod(initData(&_homogenizationMethod,0,"homogenizationMethod","0->static, 1->constrained static, 2->modal analysis"))
        , _completeInterpolation(initData(&_completeInterpolation,false,"completeInterpolation","Is the non-linear, complete interpolation used?"))
        , _useRamification(initData(&_useRamification,true,"useRamification","If SparseGridRamification, are ramifications taken into account?"))
        , _drawType(initData(&_drawType,0,"drawType",""))
        , _drawColor(initData(&_drawColor,0,"drawColor",""))
        , _drawSize(initData(&_drawSize,(float)-1.0,"drawSize",""))
    {
    }

public:

    virtual void init() override;
    virtual void reinit() override
    {
        serr<<"WARNING : composite mechanical properties can't be updated, changes on mechanical properties (young, poisson, density) are not taken into account."<<sendl;
        if(_drawSize.getValue()==-1)
            _drawSize.setValue( (float)((this->_sparseGrid->getMax()[0]-this->_sparseGrid->getMin()[0]) * .004f) );
    }

    virtual void draw(const core::visual::VisualParams* vparams) override;


    Data<bool> _finestToCoarse; ///< Does the homogenization is done directly from the finest level to the coarse one?
    Data<int> _homogenizationMethod; ///< 0->static, 1->constrained static, 2->modal analysis
    Data<bool> _completeInterpolation; ///< Is the non-linear, complete interpolation used?
    Data<bool> _useRamification; ///< If SparseGridRamification, are ramifications taken into account?
    Data<int> _drawType;
    Data<int> _drawColor;
    Data<float> _drawSize;


//       protected:


    void findFinestChildren( helper::vector<int>& finestChildren, const int elementIndice,  int level=0);
    void computeMechanicalMatricesDirectlyFromTheFinestToCoarse( ElementStiffness &K, ElementMass &M, const int elementIndice);
    void computeMechanicalMatricesRecursively( ElementStiffness &K, ElementMass &M, const int elementIndice,  int level);
    void computeMechanicalMatricesRecursivelyWithRamifications( ElementStiffness &K, ElementMass &M, const int elementIndice,  int level);

    /// multiply all weights for all levels and go to the finest level to obtain the final weights from the coarsest to the finest directly
    void computeFinalWeights( const Weight &W, const int coarseElementIndice, const int elementIndice,  int level);
    void computeFinalWeightsRamification( const Weight &W, const int coarseElementIndice, const int elementIndice,  int level);


    // surcharge NonUniformHexahedronFEMForceFieldAndMass::computeMechanicalMatricesByCondensation
//         virtual void computeMechanicalMatricesByCondensation( ElementStiffness &K, ElementMass &M, const int elementIndice,  int level);
    virtual void computeMechanicalMatricesByCondensation( ) override;



    helper::vector< helper::vector<Weight> > _weights;
    helper::vector< std::pair<int, Weight> > _finalWeights; // for each fine element -> the coarse element idx and corresponding Weight

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

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONCOMPOSITEFEMFORCEFIELDANDMASS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_NON_UNIFORM_FEM_API HexahedronCompositeFEMForceFieldAndMass<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_NON_UNIFORM_FEM_API HexahedronCompositeFEMForceFieldAndMass<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
