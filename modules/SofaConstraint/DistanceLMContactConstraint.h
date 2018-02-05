/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_CONSTRAINT_DistanceLMContactConstraint_H
#define SOFA_COMPONENT_CONSTRAINT_DistanceLMContactConstraint_H
#include "config.h"

#include <sofa/core/VecId.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/LMConstraint.h>
#include <SofaConstraint/ContactDescription.h>
#include <sofa/simulation/Node.h>


namespace sofa
{

namespace component
{

namespace constraintset
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template <class DataTypes>
class DistanceLMContactConstraintInternalData
{
};




/** Keep two particules at an initial distance
*/
template <class DataTypes>
class DistanceLMContactConstraint :  public core::behavior::LMConstraint<DataTypes,DataTypes>, public ContactDescriptionHandler
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DistanceLMContactConstraint,DataTypes),SOFA_TEMPLATE2(sofa::core::behavior::LMConstraint, DataTypes, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;

    typedef typename core::behavior::MechanicalState<DataTypes> MechanicalState;
    typedef typename sofa::core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef typename sofa::core::topology::BaseMeshTopology::Edge Edge;
    typedef core::ConstraintParams::ConstOrder ConstOrder;
    typedef core::behavior::ConstraintGroup ConstraintGroup;


protected:
    DistanceLMContactConstraint( MechanicalState *dof)
        : core::behavior::LMConstraint<DataTypes,DataTypes>(dof,dof)
        , pointPairs(sofa::core::objectmodel::Base::initData(&pointPairs, "pointPairs", "List of the edges to constrain"))
        , contactFriction(sofa::core::objectmodel::Base::initData(&contactFriction, "contactFriction", "Coulomb friction coefficient (same for all)"))
        , intersection(0)
    {
        initColorContactState();
    }

    DistanceLMContactConstraint( MechanicalState *dof1, MechanicalState * dof2)
        : core::behavior::LMConstraint<DataTypes,DataTypes>(dof1,dof2)
        , pointPairs(sofa::core::objectmodel::Base::initData(&pointPairs, "pointPairs", "List of the edges to constrain"))
        , contactFriction(sofa::core::objectmodel::Base::initData(&contactFriction, "contactFriction", "Coulomb friction coefficient (same for all)"))
        , intersection(0)
    {
        initColorContactState();
    }

    DistanceLMContactConstraint()
        : pointPairs(sofa::core::objectmodel::Base::initData(&pointPairs, "pointPairs", "List of the edges to constrain"))
        , contactFriction(sofa::core::objectmodel::Base::initData(&contactFriction, "contactFriction", "Coulomb friction coefficient (same for all)"))
        , intersection(0)
    {
        initColorContactState();
    }

    ~DistanceLMContactConstraint() override {}
public:
    // -- LMConstraint interface
    void buildConstraintMatrix(const core::ConstraintParams* cParams, core::MultiMatrixDerivId cId, unsigned int &cIndex) override;

    void writeConstraintEquations(unsigned int& lineNumber, core::MultiVecId id, ConstOrder order) override;
    void LagrangeMultiplierEvaluation(const SReal* Wptr, const SReal* cptr, SReal* LambdaInitptr,
            core::behavior::ConstraintGroup * group) override;

    bool isCorrectionComputedWithSimulatedDOF(ConstOrder order) const override;
    //
    void clear();
    /// register a new contact
    void addContact(unsigned m1, unsigned m2);
    virtual void draw(const core::visual::VisualParams* vparams) override;

    std::string getTemplateName() const override
    {
        return templateName(this);
    }
    static std::string templateName(const DistanceLMContactConstraint<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }


protected :

    /// Each scalar constraint (up to three per contact) has an associated index
    helper::vector<  unsigned int > scalarConstraintsIndices;

public:
    /// Contacts are represented by pairs of point indices
    Data< SeqEdges > pointPairs;

    /// Friction coefficients (same for all contacts)
    Data< SReal > contactFriction;

protected:
    ///Compute the length of an edge given the vector of coordinates corresponding
    double lengthEdge(const Edge &e, const VecCoord &x1,const VecCoord &x2) const;
    /// Contact normal
    Deriv computeNormal(const Edge &e, const VecCoord &x1, const VecCoord &x2) const;
    /// Contact tangent vectors
    void computeTangentVectors( Deriv& T1, Deriv& T2, const Deriv& N );

    struct Contact
    {
        //Constrained Axis
        Deriv n,t1,t2;
        Contact() {}
        Contact( Deriv norm, Deriv tgt1, Deriv tgt2 ):n(norm),t1(tgt1),t2(tgt2),contactForce(Deriv()) {}
        Deriv contactForce;
    };

    std::map< Edge, Contact > edgeToContact;
    std::map< ConstraintGroup*, Contact* > constraintGroupToContact;
    core::collision::Intersection* intersection;
protected:
    DistanceLMContactConstraintInternalData<DataTypes> data;
    friend class DistanceLMContactConstraintInternalData<DataTypes>;


    void initColorContactState()
    {
        colorsContactState.clear();
        //Vanishing
        colorsContactState.push_back(defaulttype::Vec<4,float>(0.0f,1.0f,0.0f,1.0));
        //Sticking
        colorsContactState.push_back(defaulttype::Vec<4,float>(1.0f,0.0f,0.0f,1.0));
        //Sliding
        colorsContactState.push_back(defaulttype::Vec<4,float>(1.0f,1.0f,0.0f,1.0));
        //Sliding Direction
        colorsContactState.push_back(defaulttype::Vec<4,float>(1.0f,0.0f,1.0f,1.0));
    }

    helper::vector< defaulttype::Vec<4,float> > colorsContactState;
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_CONSTRAINTSET_DistanceLMContactConstraint_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_CONSTRAINT_API DistanceLMContactConstraint<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_CONSTRAINT_API DistanceLMContactConstraint<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
