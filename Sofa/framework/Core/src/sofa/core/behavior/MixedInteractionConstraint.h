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
template<class TDataTypes1, class TDataTypes2>
class MixedInteractionConstraint : public BaseInteractionConstraint, public PairStateAccessor<TDataTypes1, TDataTypes2>
{
public:
    SOFA_ABSTRACT_CLASS2(SOFA_TEMPLATE2(MixedInteractionConstraint,TDataTypes1,TDataTypes2), BaseInteractionConstraint, SOFA_TEMPLATE2(PairStateAccessor, TDataTypes1, TDataTypes2));

    typedef TDataTypes1 DataTypes1;
    typedef typename DataTypes1::VecCoord VecCoord1;
    typedef typename DataTypes1::VecDeriv VecDeriv1;
    typedef typename DataTypes1::MatrixDeriv MatrixDeriv1;
    typedef typename DataTypes1::Coord Coord1;
    typedef typename DataTypes1::Deriv Deriv1;
    typedef TDataTypes2 DataTypes2;
    typedef typename DataTypes2::VecCoord VecCoord2;
    typedef typename DataTypes2::VecDeriv VecDeriv2;
    typedef typename DataTypes2::MatrixDeriv MatrixDeriv2;
    typedef typename DataTypes2::Coord Coord2;
    typedef typename DataTypes2::Deriv Deriv2;

    typedef core::objectmodel::Data< VecCoord1 >		DataVecCoord1;
    typedef core::objectmodel::Data< VecDeriv1 >		DataVecDeriv1;
    typedef core::objectmodel::Data< MatrixDeriv1 >		DataMatrixDeriv1;

    typedef core::objectmodel::Data< VecCoord2 >		DataVecCoord2;
    typedef core::objectmodel::Data< VecDeriv2 >		DataVecDeriv2;
    typedef core::objectmodel::Data< MatrixDeriv2 >		DataMatrixDeriv2;
protected:
    MixedInteractionConstraint(MechanicalState<DataTypes1> *mm1 = nullptr, MechanicalState<DataTypes2> *mm2 = nullptr);
    ~MixedInteractionConstraint() override;

    virtual type::vector<std::string> getInteractionIdentifiers() override final
    {
        type::vector<std::string> ids = getMixedInteractionIdentifiers();
        ids.push_back("Mixed");
        return ids;
    }

    virtual type::vector<std::string> getMixedInteractionIdentifiers(){ return {}; }


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
    virtual void getConstraintViolation(const ConstraintParams* cParams, linearalgebra::BaseVector *v, const DataVecCoord1 &x1, const DataVecCoord2 &x2
            , const DataVecDeriv1 &v1, const DataVecDeriv2 &v2) = 0;

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
    virtual void buildConstraintMatrix(const ConstraintParams* cParams, DataMatrixDeriv1 &c1, DataMatrixDeriv2 &c2, unsigned int &cIndex
            , const DataVecCoord1 &x1, const DataVecCoord2 &x2) = 0;


    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T* p0, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = core::behavior::BaseInteractionConstraint::create(p0, context, arg);

        if (arg)
        {
            obj->parse(arg);
        }

        return obj;
    }
};

#if !defined(SOFA_CORE_BEHAVIOR_MIXEDINTERACTIONCONSTRAINT_CPP)
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec3Types, defaulttype::Vec3Types>;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec2Types, defaulttype::Vec2Types>;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec1Types, defaulttype::Vec1Types>;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Rigid3Types, defaulttype::Rigid3Types> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Rigid2Types, defaulttype::Rigid2Types> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec3Types, defaulttype::Rigid3Types> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Vec2Types, defaulttype::Rigid2Types> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Rigid3Types, defaulttype::Vec3Types> ;
extern template class SOFA_CORE_API MixedInteractionConstraint<defaulttype::Rigid2Types, defaulttype::Vec2Types> ;


#endif

} // namespace sofa::core::behavior
