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
#include <sofa/component/constraint/lagrangian/model/config.h>

#include <sofa/core/behavior/PairInteractionConstraint.h>
#include <sofa/core/behavior/MechanicalState.h>

#include <sofa/type/Mat.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

#include <deque>

#include <sofa/component/constraint/lagrangian/model/BilateralConstraintResolution.h>

namespace sofa::component::constraint::lagrangian::model::bilateralinteractionconstraint
{

/// These 'using' are in a per-file namespace so they will not leak
/// and polluate the standard namespace.
using sofa::core::behavior::BaseConstraint ;
using sofa::core::behavior::ConstraintResolution ;
using sofa::core::behavior::PairInteractionConstraint ;
using sofa::core::objectmodel::Data ;
using sofa::core::ConstraintParams ;
using sofa::core::ConstVecCoordId;

using sofa::linearalgebra::BaseVector ;
using sofa::type::Vec3d;
using sofa::type::Quat ;

using sofa::defaulttype::Rigid3Types ;
using sofa::defaulttype::Vec3Types ;


template<class T>
class BilateralInteractionConstraintSpecialization {};


template<class DataTypes>
class BilateralInteractionConstraint : public PairInteractionConstraint<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BilateralInteractionConstraint,DataTypes),
               SOFA_TEMPLATE(PairInteractionConstraint,DataTypes));

    /// That any templates variation of BilateralInteractionConstraintSpecialization are friend.
    template<typename>
    friend class BilateralInteractionConstraintSpecialization ;

    typedef PairInteractionConstraint<DataTypes> Inherit;

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;
    typedef BaseConstraint::PersistentID PersistentID;

    typedef Data<VecCoord>		DataVecCoord;
    typedef Data<VecDeriv>		DataVecDeriv;
    typedef Data<MatrixDeriv>    DataMatrixDeriv;

protected:
    std::vector<Deriv> dfree;
    Quat<SReal> q;

    std::vector<unsigned int> cid;

    Data<type::vector<int> > m1; ///< index of the constraint on the first model
    Data<type::vector<int> > m2; ///< index of the constraint on the second model
    Data<VecDeriv> restVector; ///< Relative position to maintain between attached points (optional)
    VecCoord initialDifference;

    Data<double> d_numericalTolerance; ///< a real value specifying the tolerance during the constraint solving. (default=0.0001
    Data<int> activateAtIteration; ///< activate constraint at specified interation (0 = always enabled, -1=disabled)
    Data<bool> merge; ///< TEST: merge the bilateral constraints in a unique constraint
    Data<bool> derivative; ///< TEST: derivative
    Data<bool> keepOrientDiff; ///< keep the initial difference in orientation (only for rigids)
    std::vector<Vec3d> prevForces;

    // grouped square constraints
    bool squareXYZ[3];
    Deriv dfree_square_total;


    bool activated;
    int iteration;

    BilateralInteractionConstraint(MechanicalState* object1, MechanicalState* object2) ;
    BilateralInteractionConstraint(MechanicalState* object) ;
    BilateralInteractionConstraint();

    virtual ~BilateralInteractionConstraint(){}
public:
    void doBaseObjectInit() override;

    void bwdInit() override {}

    void reinit() override;

    void reset() override;

    void buildConstraintMatrix(const ConstraintParams* cParams,
                                       DataMatrixDeriv &c1, DataMatrixDeriv &c2,
                                       unsigned int &cIndex,
                                       const DataVecCoord &x1, const DataVecCoord &x2) override;

    void getConstraintViolation(const ConstraintParams* cParams,
                                        BaseVector *v,
                                        const DataVecCoord &x1, const DataVecCoord &x2,
                                        const DataVecDeriv &v1, const DataVecDeriv &v2) override;

    void getVelocityViolation(BaseVector *v,
                              const DataVecCoord &x1, const DataVecCoord &x2,
                              const DataVecDeriv &v1, const DataVecDeriv &v2);

    void getConstraintResolution(const ConstraintParams* cParams,
                                         std::vector<ConstraintResolution*>& resTab,
                                         unsigned int& offset) override;

    void handleEvent(sofa::core::objectmodel::Event *event) override;

    void draw(const core::visual::VisualParams* vparams) override;

    void clear(int reserve = 0) ;

    virtual void addContact(Deriv norm, Coord P, Coord Q, Real contactDistance,
                            int m1, int m2, Coord Pfree, Coord Qfree,
                            long id=0, PersistentID localid=0);

    void addContact(Deriv norm, Coord P, Coord Q, Real contactDistance,
                    int m1, int m2, long id=0, PersistentID localid=0) ;

    void addContact(Deriv norm, Real contactDistance, int m1, int m2,
                    long id=0, PersistentID localid=0) ;

private:
    void unspecializedInit() ;
};


#if !defined(SOFA_COMPONENT_CONSTRAINTSET_BILATERALINTERACTIONCONSTRAINT_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API BilateralInteractionConstraint< Vec3Types >;
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API BilateralInteractionConstraint< Rigid3Types >;
#endif

} // namespace sofa::component::constraint::lagrangian::model::bilateralinteractionconstraint

namespace sofa::component::constraint::lagrangian::model
{
/// Import the following into the constraintset namespace to preserve
/// compatibility with the existing sofa source code.
using bilateralinteractionconstraint::BilateralInteractionConstraint;
using bilateralinteractionconstraint::BilateralInteractionConstraintSpecialization;

} //namespace sofa::component::constraint::lagrangian::model
