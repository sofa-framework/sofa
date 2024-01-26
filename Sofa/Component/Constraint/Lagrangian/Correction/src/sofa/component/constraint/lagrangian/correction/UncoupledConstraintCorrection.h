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
#include <sofa/component/constraint/lagrangian/correction/config.h>

#include <sofa/core/behavior/ConstraintCorrection.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/topology/TopologyData.h>

namespace sofa::component::constraint::lagrangian::correction
{

/**
 *  \brief Component computing constraint forces within a simulated body using the compliance method.
 */
template<class TDataTypes>
class UncoupledConstraintCorrection : public sofa::core::behavior::ConstraintCorrection< TDataTypes >
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(UncoupledConstraintCorrection,TDataTypes), SOFA_TEMPLATE(sofa::core::behavior::ConstraintCorrection, TDataTypes));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::MatrixDeriv::RowConstIterator MatrixDerivRowConstIterator;
    typedef typename DataTypes::MatrixDeriv::ColConstIterator MatrixDerivColConstIterator;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename DataTypes::MatrixDeriv::ColIterator MatrixDerivColIterator;
    typedef typename Coord::value_type Real;

    typedef type::vector<Real> VecReal;

    typedef sofa::core::behavior::ConstraintCorrection< TDataTypes > Inherit;
protected:
    UncoupledConstraintCorrection(sofa::core::behavior::MechanicalState<DataTypes> *mm = nullptr);

    virtual ~UncoupledConstraintCorrection();
public:
    void init() override;

    void reinit() override;

    void addComplianceInConstraintSpace(const sofa::core::ConstraintParams *cparams, sofa::linearalgebra::BaseMatrix *W) override;

    void getComplianceMatrix(linearalgebra::BaseMatrix* ) const override;

    // for multigrid approach => constraints are merged
    void getComplianceWithConstraintMerge(linearalgebra::BaseMatrix* Wmerged, std::vector<int> &constraint_merge) override;


    /// @name Correction API
    /// @{

    void computeMotionCorrection(const core::ConstraintParams* cparams, core::MultiVecDerivId dx, core::MultiVecDerivId f) override;

    void applyMotionCorrection(const sofa::core::ConstraintParams *cparams, Data< VecCoord > &x, Data< VecDeriv > &v, Data< VecDeriv > &dx, const Data< VecDeriv > &correction) override;

    void applyPositionCorrection(const sofa::core::ConstraintParams *cparams, Data< VecCoord > &x, Data<VecDeriv>& dx, const Data< VecDeriv > & correction) override;

    void applyVelocityCorrection(const sofa::core::ConstraintParams *cparams, Data< VecDeriv > &v, Data<VecDeriv>& dv, const Data< VecDeriv > & correction) override;

    /// @}


    /// @name Deprecated API
    /// @{

    void applyContactForce(const linearalgebra::BaseVector *f) override;

    void resetContactForce() override;

    /// @}


    /// @name Unbuilt constraint system during resolution
    /// @{

    bool hasConstraintNumber(int index) override;  // virtual ???

    void resetForUnbuiltResolution(SReal* f, std::list<unsigned int>& /*renumbering*/) override;

    void addConstraintDisplacement(SReal* d, int begin,int end) override;

    void setConstraintDForce(SReal* df, int begin, int end, bool update) override;

    void getBlockDiagonalCompliance(linearalgebra::BaseMatrix* W, int begin, int end) override;

    /// @}

    core::topology::PointData< VecReal > compliance; ///< Rigid compliance value: 1st value for translations, 6 others for upper-triangular part of symmetric 3x3 rotation compliance matrix

    Data< Real > defaultCompliance; ///< Default compliance value for new dof or if all should have the same (in which case compliance vector should be empty)

    Data<bool> f_verbose; ///< Dump the constraint matrix at each iteration

    Data< Real > d_correctionVelocityFactor; ///< Factor applied to the constraint forces when correcting the velocities
    Data< Real > d_correctionPositionFactor; ///< Factor applied to the constraint forces when correcting the positions

    Data < bool > d_useOdeSolverIntegrationFactors; ///< Use odeSolver integration factors instead of correctionVelocityFactor and correctionPositionFactor
                                                    
    /// Link to be set to the topology container in the component graph.
    SingleLink<UncoupledConstraintCorrection<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

private:
    // new :  for non building the constraint system during solving process //
    VecDeriv constraint_disp, constraint_force;
    std::list<int> constraint_dofs;		// list of indices of each point which is involve with constraint

protected:

    sofa::core::behavior::OdeSolver* m_pOdeSolver;

    /**
     * @brief Compute dx correction from motion space force vector.
     */
    void computeDx(const Data< VecDeriv > &f, VecDeriv& x);
};


template<>
void UncoupledConstraintCorrection< sofa::defaulttype::Rigid3Types >::init();

template<>
void UncoupledConstraintCorrection< sofa::defaulttype::Rigid3Types >::getComplianceMatrix(sofa::linearalgebra::BaseMatrix * /*m*/) const;


#if !defined(SOFA_COMPONENT_CONSTRAINTSET_UNCOUPLEDCONSTRAINTCORRECTION_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_CORRECTION_API UncoupledConstraintCorrection<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_CORRECTION_API UncoupledConstraintCorrection<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_CORRECTION_API UncoupledConstraintCorrection<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_CORRECTION_API UncoupledConstraintCorrection<defaulttype::Rigid3Types>;

#endif

} //namespace sofa::component::constraint::lagrangian::correction
