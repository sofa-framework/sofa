/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_CONSTRAINT_ROTATIONLMCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_ROTATIONLMCONSTRAINT_INL

#include <sofa/component/constraint/RotationLMConstraint.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/template.h>





namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::helper;


// Define TestNewPointFunction
template< class DataTypes>
bool RotationLMConstraint<DataTypes>::FCTestNewPointFunction(int /*nbPoints*/, void* param, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& )
{
    RotationLMConstraint<DataTypes> *fc= (RotationLMConstraint<DataTypes> *)param;
    if (fc)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// Define RemovalFunction
template< class DataTypes>
void RotationLMConstraint<DataTypes>::FCRemovalFunction(int pointIndex, void* param)
{
    RotationLMConstraint<DataTypes> *fc= (RotationLMConstraint<DataTypes> *)param;
    if (fc)
    {
        fc->removeConstraint((unsigned int) pointIndex);
    }
    return;
}

template <class DataTypes>
void RotationLMConstraint<DataTypes>::clearConstraints()
{
    f_indices.beginEdit()->clear();
    f_indices.endEdit();
}

template <class DataTypes>
void RotationLMConstraint<DataTypes>::addConstraint(unsigned int index)
{
    f_indices.beginEdit()->push_back(index);
    f_indices.endEdit();
}

template <class DataTypes>
void RotationLMConstraint<DataTypes>::removeConstraint(unsigned int index)
{
    removeValue(*f_indices.beginEdit(),index);
    f_indices.endEdit();
}


template <class DataTypes>
void RotationLMConstraint<DataTypes>::init()
{
    core::componentmodel::behavior::LMConstraint<DataTypes,DataTypes>::init();

    topology = this->getContext()->getMeshTopology();

    // Initialize functions and parameters
    topology::PointSubset my_subset = f_indices.getValue();

    my_subset.setTestFunction(FCTestNewPointFunction);
    my_subset.setRemovalFunction(FCRemovalFunction);

    my_subset.setTestParameter( (void *) this );
    my_subset.setRemovalParameter( (void *) this );
}

// Handle topological changes
template <class DataTypes> void RotationLMConstraint<DataTypes>::handleTopologyChange()
{
    std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd =topology->lastChange();

    f_indices.beginEdit()->handleTopologyEvents(itBegin,itEnd,this->constrainedObject1->getSize());
}


template<class DataTypes>
void RotationLMConstraint<DataTypes>::buildJacobian()
{
    idxEquations.clear();
    const SetIndexArray &indices = f_indices.getValue().getArray();
    const helper::vector<Deriv> &axis=rotationAxis.getValue();

    idxEquations.resize(indices.size());
    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        const unsigned int index=*it;

        for (unsigned int i=0; i<axis.size(); ++i)
        {
            SparseVecDeriv V; V.add(index,axis[i]);
            idxEquations[index].push_back(registerEquationInJ1(V));
        }
        this->constrainedObject1->forceMask.insertEntry(index);
    }
}


template<class DataTypes>
void RotationLMConstraint<DataTypes>::writeConstraintEquations(ConstOrder Order)
{

    //We don't constrain the Position, only the velocities and accelerations
    if (Order==core::componentmodel::behavior::BaseLMConstraint::POS) return;


    const SetIndexArray & indices = f_indices.getValue().getArray();
    const helper::vector<SReal> &factor=factorAxis.getValue();

    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        const unsigned int index = *it;

        core::componentmodel::behavior::BaseLMConstraint::ConstraintGroup *constraint = this->addGroupConstraint(Order);
        for (unsigned int i=0; i<idxEquations[index].size(); ++i)
        {
            SReal correction=0;
            switch(Order)
            {
            case core::componentmodel::behavior::BaseLMConstraint::ACC :
            {
                correction = this->constrainedObject1->getConstraintJacobianTimesVecDeriv(idxEquations[index][i],core::componentmodel::behavior::BaseMechanicalState::VecId::dx());

                break;
            }
            case core::componentmodel::behavior::BaseLMConstraint::VEL :
            {
                correction = this->constrainedObject1->getConstraintJacobianTimesVecDeriv(idxEquations[index][i],core::componentmodel::behavior::BaseMechanicalState::VecId::velocity());
                break;
            }
            default: break;
            };
            if (!factor.empty())
            {
                if (i < factor.size()) correction*=factor[i];
                else                   correction*=factor.back();
            }
            constraint->addConstraint( idxEquations[index][i], -correction, core::componentmodel::behavior::BaseLMConstraint::BILATERAL);
        }

    }
}



template <class DataTypes>
void RotationLMConstraint<DataTypes>::draw()
{
    if (!this->getContext()->getShowForceFields()) return;
    const VecCoord& x = *this->constrainedObject1->getX();

    const SetIndexArray & indices = f_indices.getValue().getArray();

    for (SetIndexArray::const_iterator it = indices.begin();
            it != indices.end();
            ++it)
    {
        unsigned int index=(*it);
        Coord pos=x[index];
        const defaulttype::Vector3 &c=pos.getCenter();
        glColor3f(1,1,0);
        const helper::vector<Deriv>& axis=rotationAxis.getValue();
        for (unsigned int i=0; i<axis.size(); ++i)
        {
            helper::gl::Axis::draw(c,c+axis[i].getVOrientation()*showSizeAxis.getValue(),
                    showSizeAxis.getValue()*0.03);
        }

    }

}





} // namespace constraint

} // namespace component

} // namespace sofa

#endif


