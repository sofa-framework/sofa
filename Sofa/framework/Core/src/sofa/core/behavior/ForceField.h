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
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/core/behavior/SingleStateAccessor.h>

namespace sofa::core::behavior
{

/**
 *  \brief Component computing forces within a simulated body.
 *
 *  This class defines the abstract API common to force fields using a
 *  given type of DOFs.
 *  A force field computes forces applied to one simulated body
 *  given its current position and velocity.
 *
 *  For implicit integration schemes, it must also compute the derivative
 *  ( df, given a displacement dx ).
 */
template<class TDataTypes>
class ForceField : public BaseForceField, public SingleStateAccessor<TDataTypes>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(ForceField, TDataTypes), BaseForceField, SOFA_TEMPLATE(SingleStateAccessor, TDataTypes));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real             Real;
    typedef typename DataTypes::Coord            Coord;
    typedef typename DataTypes::Deriv            Deriv;
    typedef typename DataTypes::VecCoord         VecCoord;
    typedef typename DataTypes::VecDeriv         VecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
protected:
    explicit ForceField(MechanicalState<DataTypes> *mm = nullptr);

    ~ForceField() override;
public:

    /// @name Vector operations
    /// @{

    /// Given the current position and velocity states, update the current force
    /// vector by computing and adding the forces associated with this
    /// ForceField.
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// \f$ f += B v + K x \f$
    ///
    /// This method retrieves the force, x and v vector from the MechanicalState
    /// and call the internal addForce(const MechanicalParams*, DataVecDeriv&,const DataVecCoord&,const DataVecDeriv&)
    /// method implemented by the component.
    void addForce(const MechanicalParams* mparams, MultiVecDerivId fId ) override;

    /// Given the current position and velocity states, update the current force
    /// vector by computing and adding the forces associated with this
    /// ForceField.
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// \f$ f += B v + K x \f$
    ///
    /// This is the method that should be implemented by the component
    virtual void addForce(const MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) = 0;

    /// Compute the force derivative given a small displacement from the
    /// position and velocity used in the previous call to addForce().
    ///
    /// The derivative should be directly derived from the computations
    /// done by addForce. Any forces neglected in addDForce will be integrated
    /// explicitly (i.e. using its value at the beginning of the timestep).
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// \f$ df += kFactor K dx + bFactor B dx \f$
    ///
    /// This method retrieves the force and dx vector from the MechanicalState
    /// and call the internal addDForce(VecDeriv&,const VecDeriv&,SReal,SReal)
    /// method implemented by the component.
    void addDForce(const MechanicalParams* mparams, MultiVecDerivId dfId ) override;

    /// Internal addDForce
    /// Overloaded function, usually called from the generic addDForce version.
    /// This addDForce version directly gives access to df and dx vectors through its parameters.
    /// @param mparams
    /// @param df Output vector to fill, result of \f$ kFactor K dx + bFactor B dx \f$
    /// @param dx Input vector used to compute \f$ df = kFactor K dx + bFactor B dx \f$
    virtual void addDForce(const MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx ) = 0;

    /// Compute the product of the Compliance matrix C
    /// with the Lagrange multipliers lambda
    /// \f$ res += cFactor * C * lambda \f$
    /// used by the graph-scattered (unassembled API when the ForceField is handled as a constraint)
    void addClambda(const MechanicalParams* mparams, MultiVecDerivId resId, MultiVecDerivId lambdaId, SReal cFactor ) override;

    virtual void addClambda(const MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& lambda, SReal cFactor );

    /// Get the potential energy associated to this ForceField.
    ///
    /// Used to estimate the total energy of the system by some
    /// post-stabilization techniques.
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic ForceField::getPotentialEnergy(const MechanicalParams* mparams) method.
    SReal getPotentialEnergy(const MechanicalParams* mparams) const override;

    virtual SReal getPotentialEnergy(const MechanicalParams* /*mparams*/, const DataVecCoord& x) const = 0;


    /// @}

    /// @name Matrix operations
    /// @{

    void addKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix ) override;

    /// Internal addKToMatrix
    /// Overloaded function, usually called from the generic addKToMatrix version.
    /// This addKToMatrix version directly gives access to the matrix to fill, the stiffness factor and
    /// the offset used to identify where the force field must add its contributions in the matrix.
    /// @param matrix the global stiffness matrix in which the force field adds its contribution. The matrix is global,
    /// i.e. different objects can add their contribution into the same large matrix. Each object adds its contribution
    /// to a different section of the matrix. That is why, an offset is used to identify where in the matrix the force
    /// field must start adding its contribution.
    /// @param kFact stiffness factor that needs to be multiplied to each matrix entry.
    /// @param offset Starting index of the submatrix to fill in the global matrix.
    virtual void addKToMatrix(sofa::linearalgebra::BaseMatrix * matrix, SReal kFact, unsigned int &offset);

    void addBToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    /** Accumulate an element matrix to a global assembly matrix. This is a helper for addKToMatrix, to accumulate each (square) element matrix in the (square) assembled matrix.
    \param bm the global assembly matrix
    \param offset start index of the local DOFs within the global matrix
    \param nodeIndex indices of the nodes of the element within the local nodes, as stored in the topology
    \param em element matrix, typically a stiffness, damping, mass, or weighted sum thereof
    \param scale weight applied to the matrix, typically ±params->kfactor() for a stiffness matrix
    */
    template<class IndexArray, class ElementMat>
    void addToMatrix(sofa::linearalgebra::BaseMatrix* bm, unsigned offset, const IndexArray& nodeIndex, const ElementMat& em, SReal scale )
    {
        constexpr auto S = DataTypes::deriv_total_size; // size of node blocks
        for (unsigned n1=0; n1<nodeIndex.size(); n1++)
        {
            for(unsigned i=0; i<S; i++)
            {
                const unsigned ROW = offset + S*nodeIndex[n1] + i;  // i-th row associated with node n1 in BaseMatrix
                const unsigned row = S*n1+i;                        // i-th row associated with node n1 in the element matrix

                for (unsigned n2=0; n2<nodeIndex.size(); n2++)
                {
                    for (unsigned j=0; j<S; j++)
                    {
                        const unsigned COLUMN = offset + S*nodeIndex[n2] +j; // j-th column associated with node n2 in BaseMatrix
                        const unsigned column = S*n2+j;                      // j-th column associated with node n2 in the element matrix
                        bm->add( ROW,COLUMN, em[row][column]* scale );
                    }
                }
            }
        }
    }

    virtual void addBToMatrix(sofa::linearalgebra::BaseMatrix * matrix, SReal bFact, unsigned int &offset);
    /// @}

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        const std::string attributeName {"mstate"};
        std::string mstateLink = arg->getAttribute(attributeName,"");
        if (mstateLink.empty())
        {
            if (dynamic_cast<MechanicalState<DataTypes>*>(context->getMechanicalState()) == nullptr)
            {
                arg->logError("Since the attribute '" + attributeName + "' has not been specified, a mechanical state "
                    "with the datatype '" + DataTypes::Name() + "' has been searched in the current context, but not found.");
                return false;
            }
        }
        else
        {
            MechanicalState<DataTypes>* mstate = nullptr;
            context->findLinkDest(mstate, mstateLink, nullptr);
            if (!mstate)
            {
                arg->logError("Data attribute '" + attributeName + "' does not point to a valid mechanical state of datatype '" + std::string(DataTypes::Name()) + "'.");
                return false;
            }
        }
        return BaseObject::canCreate(obj, context, arg);
    }

    template<class T>
    static std::string shortName(const T* ptr = nullptr, objectmodel::BaseObjectDescription* arg = nullptr)
    {
        std::string name = Inherit1::shortName(ptr, arg);
        sofa::helper::replaceAll(name, "ForceField", "FF");
        return name;
    }

};

#if !defined(SOFA_CORE_BEHAVIOR_FORCEFIELD_CPP)
extern template class SOFA_CORE_API ForceField<defaulttype::Vec3Types>;
extern template class SOFA_CORE_API ForceField<defaulttype::Vec2Types>;
extern template class SOFA_CORE_API ForceField<defaulttype::Vec1Types>;
extern template class SOFA_CORE_API ForceField<defaulttype::Vec6Types>;
extern template class SOFA_CORE_API ForceField<defaulttype::Rigid3Types>;
extern template class SOFA_CORE_API ForceField<defaulttype::Rigid2Types>;


#endif

} // namespace sofa::core::behavior
