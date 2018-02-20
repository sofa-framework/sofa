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
#ifndef SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRALFEMFORCEFIELDANDMASS_H
#define SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRALFEMFORCEFIELDANDMASS_H
#include "config.h"

#include <SofaGeneralSimpleFem/HexahedralFEMForceFieldAndMass.h>
#include <sofa/core/topology/TopologyChange.h>

namespace sofa
{

namespace component
{

namespace topology
{
class MultilevelHexahedronSetTopologyContainer;
class MultilevelModification;
}

namespace forcefield
{


/**

@InProceedings{NPF06,
author       = "Nesme, Matthieu and Payan, Yohan and Faure, Fran\c{c}ois",
title        = "Animating Shapes at Arbitrary Resolution with Non-Uniform Stiffness",
booktitle    = "Eurographics Workshop in Virtual Reality Interaction and Physical Simulation (VRIPHYS)",
month        = "nov",
year         = "2006",
organization = "Eurographics",
address      = "Madrid",
url          = "http://www-evasion.imag.fr/Publications/2006/NPF06"
}

indices ordering (same as in HexahedronSetTopology):

     Y  7---------6
     ^ /         /|
     |/    Z    / |
     3----^----2  |
     |   /     |  |
     |  4------|--5
     | /       | /
     |/        |/
     0---------1-->X

*/

template<class DataTypes>
class NonUniformHexahedralFEMForceFieldAndMass : virtual public HexahedralFEMForceFieldAndMass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(NonUniformHexahedralFEMForceFieldAndMass, DataTypes), SOFA_TEMPLATE(HexahedralFEMForceFieldAndMass, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef HexahedralFEMForceFieldAndMass<DataTypes> HexahedralFEMForceFieldAndMassT;
    typedef HexahedralFEMForceField<DataTypes> HexahedralFEMForceFieldT;

    typedef typename HexahedralFEMForceFieldAndMassT::VecElement VecElement;
    typedef typename HexahedralFEMForceFieldAndMassT::ElementStiffness ElementStiffness;
    typedef typename HexahedralFEMForceFieldAndMassT::MaterialStiffness MaterialStiffness;
    typedef typename HexahedralFEMForceFieldAndMassT::MassT MassT;
    typedef typename HexahedralFEMForceFieldAndMassT::ElementMass ElementMass;
    typedef typename HexahedralFEMForceFieldAndMassT::Element Element;

    typedef typename defaulttype::Mat<8, 8, Real> Mat88;
    typedef typename defaulttype::Vec<3, int> Vec3i;


protected:
    NonUniformHexahedralFEMForceFieldAndMass();
public:
    virtual void init() override;
    virtual void reinit() override;

    // handle topological changes
    virtual void handleTopologyChange(core::topology::Topology*) override;

protected:
    /// condensate matrice from finest level to the actual mechanical level
    virtual void computeMechanicalMatricesByCondensation(
        ElementStiffness &K,
        ElementMass &M,
        Real& totalMass,const int elementIndex);

//  virtual void computeCorrection(ElementMass& /*M*/) {};

    void initLarge(const int i);

    void initPolar(const int i);

private:

    void handleHexaAdded(const core::topology::HexahedraAdded&);
    void handleHexaRemoved(const core::topology::HexahedraRemoved&);
    void handleMultilevelModif(const component::topology::MultilevelModification&);


    void computeHtfineH(const Mat88& H, const ElementStiffness& fine, ElementStiffness& HtfineH ) const;
    void addHtfineHtoCoarse(const Mat88& H, const ElementStiffness& fine, ElementStiffness& coarse ) const;
    void subtractHtfineHfromCoarse(const Mat88& H, const ElementStiffness& fine, ElementStiffness& coarse ) const;

    void computeMechanicalMatricesByCondensation_IntervalAnalysis(
        ElementStiffness &K,
        ElementMass &M,
        Real& totalMass,
        const ElementStiffness &K_fine,
        const ElementMass &M_fine,
        const Real& mass_fine,
        const unsigned int level,
        const std::set<Vec3i>& voxels) const;

    void computeMechanicalMatricesByCondensation_Recursive(
        ElementStiffness &K,
        ElementMass &M,
        Real& totalMass,
        const ElementStiffness &K_fine,
        const ElementMass &M_fine,
        const Real& mass_fine,
        const unsigned int level,
        const unsigned int startIdx,
        const std::set<unsigned int>& fineChildren) const;

    void computeMechanicalMatricesByCondensation_Direct(
        ElementStiffness &K,
        ElementMass &M,
        Real& totalMass,
        const ElementStiffness &K_fine,
        const ElementMass &M_fine,
        const Real& mass_fine,
        const unsigned int level,
        const std::set<Vec3i>& voxels) const;


    int ijk2octree(const int i, const int j, const int k) const;
    void octree2ijk(const int octreeIdx, int &i, int &j, int &k) const;
    Vec3i octree2voxel(const int octreeIdx) const;

    // [level][childId][childNodeId][parentNodeId] -> weight
    helper::vector< helper::vector < Mat88 > > _H; ///< interpolation matrices from finer level to a coarser (to build stiffness and mass matrices)

    typedef struct
    {
        MaterialStiffness	C;	// Mat<6, 6, Real>
        ElementStiffness	K;	// Mat<24, 24, Real>
        ElementMass		M;	// Mat<24, 24, Real>
        Real			mass;
    } Material;

    Material _material; // TODO: enable combination of multiple materials

    component::topology::MultilevelHexahedronSetTopologyContainer*	_multilevelTopology;

    Data<bool>		_bRecursive; ///< Use recursive matrix computation

protected:

    // ---------------  Modified method: compute and re-use MBK
    typedef HexahedralFEMForceFieldAndMass<DataTypes> Inherited;
    typedef typename Inherited::HexahedronInformation HexahedronInformation;
    typedef typename Inherited::Mat33 Mat33;
    typedef typename Inherited::Displacement Displacement;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    Data<bool> useMBK; ///< if true, compute and use MBK matrix

    /** Matrix-vector product for implicit methods with iterative solvers.
        If the MBK matrix is ill-conditionned, recompute it, and correct it to avoid too small singular values.
    */
    virtual void addMBKdx(const core::MechanicalParams* mparams, core::MultiVecDerivId dfId) override;

    bool matrixIsDirty;                      ///< Matrix \f$ \alpha M + \beta B + \gamma C \f$ needs to be recomputed
    helper::vector< ElementMass > mbkMatrix; ///< Matrix \f$ \alpha M + \beta B + \gamma C \f$

protected:
    virtual void computeCorrection( ElementMass& ) {} ///< Limit the conditioning number of each mbkMatrix as defined by maxConditioning (in derived classes).
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRALFEMFORCEFIELDANDMASS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_NON_UNIFORM_FEM_API NonUniformHexahedralFEMForceFieldAndMass<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_NON_UNIFORM_FEM_API NonUniformHexahedralFEMForceFieldAndMass<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
