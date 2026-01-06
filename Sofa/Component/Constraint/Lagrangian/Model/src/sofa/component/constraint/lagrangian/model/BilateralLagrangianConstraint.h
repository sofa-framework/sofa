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

#include <sofa/core/topology/TopologySubsetIndices.h>
#include <sofa/core/behavior/PairInteractionConstraint.h>
#include <sofa/core/behavior/MechanicalState.h>

#include <sofa/type/Mat.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

#include <deque>

#include <sofa/component/constraint/lagrangian/model/BilateralConstraintResolution.h>
#include <sofa/core/objectmodel/DataCallback.h>

namespace sofa::component::constraint::lagrangian::model
{

/// These 'using' are in a per-file namespace so they will not leak
/// and polluate the standard namespace.
using sofa::core::behavior::BaseLagrangianConstraint ;
using sofa::core::behavior::ConstraintResolution ;
using sofa::core::behavior::PairInteractionConstraint ;
using sofa::core::ConstraintParams ;
using sofa::core::ConstVecCoordId;

using sofa::linearalgebra::BaseVector ;
using sofa::type::Vec3;
using sofa::type::Quat ;

using sofa::defaulttype::Rigid3Types ;
using sofa::defaulttype::Vec3Types ;


template<class T>
class BilateralLagrangianConstraintSpecialization {};


template<class DataTypes>
class BilateralLagrangianConstraint : public PairInteractionConstraint<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BilateralLagrangianConstraint,DataTypes),
               SOFA_TEMPLATE(PairInteractionConstraint,DataTypes));

    /// That any templates variation of BilateralLagrangianConstraintSpecialization are friend.
    template<typename>
    friend class BilateralLagrangianConstraintSpecialization ;

    typedef PairInteractionConstraint<DataTypes> Inherit;

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;
    typedef BaseLagrangianConstraint::PersistentID PersistentID;

    typedef Data<VecCoord>		DataVecCoord;
    typedef Data<VecDeriv>		DataVecDeriv;
    typedef Data<MatrixDeriv>    DataMatrixDeriv;

    using SubsetIndices = type::vector<Index>;
    using DataSubsetIndices = sofa::core::topology::TopologySubsetIndices;

protected:
    sofa::type::vector<Deriv> m_violation;
    Quat<SReal> q;

    std::vector<unsigned int> cid;

    DataSubsetIndices d_m1; ///< index of the constraint on the first model
    DataSubsetIndices d_m2; ///< index of the constraint on the second model
    Data<VecDeriv> d_restVector; ///< Relative position to maintain between attached points (optional)
    VecCoord initialDifference;

    SOFA_ATTRIBUTE_DISABLED__BILATERALREMOVEUNUSEDTOLERANCE() DeprecatedAndRemoved d_numericalTolerance; ///< a real value specifying the tolerance during the constraint solving. (default=0.0001

    Data<bool> d_activate; ///< control constraint activation (true by default)
    Data<bool> d_keepOrientDiff; ///< keep the initial difference in orientation (only for rigids)
    Data<SReal> d_load; ///< Apply this factor to the constraint force to enable incremental loading. This value should be in the interval [0.0, 1.0].
    core::objectmodel::DataCallback c_loadCallback;



    SingleLink<BilateralLagrangianConstraint<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology1; ///< Link to be set to the first topology container in order to support topological changes
    SingleLink<BilateralLagrangianConstraint<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology2; ///< Link to be set to the second topology container in order to support topological changes

    std::vector<Vec3> prevForces;

    BilateralLagrangianConstraint(MechanicalState* object1, MechanicalState* object2) ;
    BilateralLagrangianConstraint(MechanicalState* object) ;
    BilateralLagrangianConstraint();

    ~BilateralLagrangianConstraint() override = default;

public:
    void init() override;

    void bwdInit() override {}

    void reinit() override;

    void buildConstraintMatrix(const ConstraintParams* cParams,
                               DataMatrixDeriv& c1, DataMatrixDeriv& c2,
                               unsigned int& cIndex,
                               const DataVecCoord& x1, const DataVecCoord& x2) override;

    void getConstraintViolation(const ConstraintParams* cParams,
                                BaseVector* v,
                                const DataVecCoord& x1, const DataVecCoord& x2,
                                const DataVecDeriv& v1, const DataVecDeriv& v2) override;

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

    /// Method to remove a contact using point @param indices and id of buffer: @sa m1 (resp. @sa 2m) if @param objectId is equal to 0 (resp. to 1)
    void removeContact(int objectId, SubsetIndices indices);

    virtual type::vector<std::string> getBilateralInteractionIdentifiers() {return {};}

    virtual type::vector<std::string> getPairInteractionIdentifiers() override final
    {
        type::vector<std::string> ids = getBilateralInteractionIdentifiers();
        ids.push_back("Bilateral");
        return ids;
    }

private:
    void unspecializedInit() ;

    /// Method to get the index position of a @param point Id inside @sa m1 or @sa m2) depending of the value passed in @param cIndices. Return InvalidID if not found.
    Index indexOfElemConstraint(const SubsetIndices& cIndices, Index Id);
};

template<> SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API
void BilateralLagrangianConstraint<Rigid3Types>::bwdInit();


#if !defined(SOFA_COMPONENT_CONSTRAINTSET_BILATERALLAGRANGIANCONSTRAINT_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API BilateralLagrangianConstraint< Vec3Types >;
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API BilateralLagrangianConstraint< Rigid3Types >;
#endif

} // namespace sofa::component::constraint::lagrangian::model
