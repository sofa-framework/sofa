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

#include <sofa/core/config.h>
#include <sofa/core/behavior/BaseInteractionConstraint.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/PairStateAccessor.h>


namespace sofa::core::behavior
{

/**
 *  \brief Component computing constraints between a pair of simulated body.
 *
 *  This class define the abstract API common to interaction constraints
 *  between a pair of bodies using a given type of DOFs.
 */
template<class TDataTypes>
class PairInteractionConstraint : public BaseInteractionConstraint, public PairStateAccessor<TDataTypes>
{
public:
    SOFA_ABSTRACT_CLASS2(SOFA_TEMPLATE(PairInteractionConstraint,TDataTypes), BaseInteractionConstraint, SOFA_TEMPLATE2(PairStateAccessor,TDataTypes,TDataTypes));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef core::objectmodel::Data<VecCoord>		DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv>		DataVecDeriv;
    typedef core::objectmodel::Data<MatrixDeriv>    DataMatrixDeriv;
protected:
    explicit PairInteractionConstraint(MechanicalState<DataTypes> *mm1 = nullptr, MechanicalState<DataTypes> *mm2 = nullptr);

    ~PairInteractionConstraint() override;
public:
    Data<SReal> endTime;  ///< Time when the constraint becomes inactive (-1 for infinitely active)
    virtual bool isActive() const; ///< if false, the constraint does nothing

    using BaseConstraintSet::getConstraintViolation;
    /// Construct the Constraint violations vector of each constraint
    ///
    /// \param v is the result vector that contains the whole constraints violations
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    void getConstraintViolation(const ConstraintParams* cParams, linearalgebra::BaseVector *v) override;

    /// Construct the Constraint violations vector of each constraint
    ///
    /// \param v is the result vector that contains the whole constraints violations
    /// \param x1 and x2 are the position vectors used to compute contraint position violation
    /// \param v1 and v2 are the velocity vectors used to compute contraint velocity violation
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    ///
    /// This is the method that should be implemented by the component
    virtual void getConstraintViolation(const ConstraintParams* cParams, linearalgebra::BaseVector *v, const DataVecCoord &x1, const DataVecCoord &x2
            , const DataVecDeriv &v1, const DataVecDeriv &v2) = 0;

    /// Construct the Jacobian Matrix
    ///
    /// \param cId is the result constraint sparse matrix Id
    /// \param cIndex is the index of the next constraint equation: when building the constraint matrix, you have to use this index, and then update it
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    void buildConstraintMatrix(const ConstraintParams* cParams, MultiMatrixDerivId cId, unsigned int &cIndex) override;

    /// Construct the Jacobian Matrix
    ///
    /// \param c1 and c2 are the results constraint sparse matrix
    /// \param cIndex is the index of the next constraint equation: when building the constraint matrix, you have to use this index, and then update it
    /// \param x1 and x2 are the position vectors used for contraint equation computation
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    ///
    /// This is the method that should be implemented by the component
    virtual void buildConstraintMatrix(const ConstraintParams* cParams, DataMatrixDeriv &c1, DataMatrixDeriv &c2, unsigned int &cIndex
            , const DataVecCoord &x1, const DataVecCoord &x2) = 0;


    void storeLambda(const ConstraintParams* cParams, MultiVecDerivId res, const sofa::linearalgebra::BaseVector* lambda) override;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        MechanicalState<DataTypes>* mstate1 = nullptr;
        MechanicalState<DataTypes>* mstate2 = nullptr;
        std::string object1 = arg->getAttribute("object1","@./");
        std::string object2 = arg->getAttribute("object2","@./");

        context->findLinkDest(mstate1, object1, nullptr);
        context->findLinkDest(mstate2, object2, nullptr);

        if (!mstate1) {
            arg->logError("Data attribute 'object1' does not point to a valid mechanical state of datatype '" + std::string(DataTypes::Name()) + "'.");
            return false;
        }
        if (!mstate2) {
            arg->logError("Data attribute 'object2' does not point to a valid mechanical state of datatype '" + std::string(DataTypes::Name()) + "'.");
            return false;
        }

        return BaseInteractionConstraint::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T* p0, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = core::behavior::BaseInteractionConstraint::create(p0, context, arg);

        if (arg)
        {
            const std::string object1 = arg->getAttribute("object1","");
            const std::string object2 = arg->getAttribute("object2","");
            if (!object1.empty())
            {
                arg->setAttribute("object1", object1);
            }
            if (!object2.empty())
            {
                arg->setAttribute("object2", object2);
            }
            obj->parse(arg);
        }

        return obj;
    }

    using Inherit2::getMechModel1;
    using Inherit2::getMechModel2;

protected:

     virtual type::vector<std::string> getInteractionIdentifiers() override final
     {
            type::vector<std::string> ids = getPairInteractionIdentifiers();
            ids.push_back("Pair");
            return ids;
     }

     virtual type::vector<std::string> getPairInteractionIdentifiers(){ return {}; }

    void storeLambda(const ConstraintParams* cParams, Data<VecDeriv>& res1, Data<VecDeriv>& res2, const Data<MatrixDeriv>& j1, const Data<MatrixDeriv>& j2,
                               const sofa::linearalgebra::BaseVector* lambda);
};

#if !defined(SOFA_CORE_BEHAVIOR_PAIRINTERACTIONCONSTRAINT_CPP)
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Vec3Types>;
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Vec2Types>;
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Vec1Types>;
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Rigid3Types>;
extern template class SOFA_CORE_API PairInteractionConstraint<defaulttype::Rigid2Types>;


#endif

} // namespace sofa::core::behavior
