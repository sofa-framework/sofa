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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/componentmodel/behavior/BaseLMConstraint.h>
namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

BaseLMConstraint::BaseLMConstraint():
    pathObject1( initData(&pathObject1,  "object1","First Object to constrain") ),
    pathObject2( initData(&pathObject2,  "object2","Second Object to constrain") )
{
};

unsigned int BaseLMConstraint::getNumConstraint(ConstOrder Order)
{
    unsigned int result=0;
    helper::vector< ConstraintGroup* > &vec = constraintOrder[Order];
    for (unsigned int i=0; i<vec.size(); ++i)
    {
        result+=vec[i]->getNumConstraint();
    }
    return result;
}

BaseLMConstraint::ConstraintGroup* BaseLMConstraint::addGroupConstraint( ConstOrder id)
{
    ConstraintGroup *c=new ConstraintGroup(id);
    constraintOrder[id].push_back(c);
    return c;
}

void BaseLMConstraint::constraintTransmissionJ1(unsigned int entry)
{
    for (std::map< unsigned int,unsigned int >::iterator it=linesInSimulatedObject1.begin(); it!=linesInSimulatedObject1.end(); ++it)
    {
        it->second += entry;
    }
}

void BaseLMConstraint::constraintTransmissionJ2(unsigned int entry)
{
    for (std::map< unsigned int,unsigned int >::iterator it=linesInSimulatedObject2.begin(); it!=linesInSimulatedObject2.end(); ++it)
    {
        it->second += entry;
    }
}

void BaseLMConstraint::getIndicesUsed1(ConstOrder Order, helper::vector< unsigned int > &used0)
{
    const helper::vector< BaseLMConstraint::ConstraintGroup* > &constraints=constraintOrder[Order];

    for (unsigned int idxGroupConstraint=0; idxGroupConstraint<constraints.size(); ++idxGroupConstraint)
    {
        ConstraintGroup *group=constraints[idxGroupConstraint];
        std::pair< ConstraintGroup::EquationIterator, ConstraintGroup::EquationIterator > range=group->data();

        for (ConstraintGroup::EquationIterator equation=range.first; equation!=range.second; ++equation)
        {
            if (equation->idxInConstrainedDOF1 >= 0)
                used0.push_back(linesInSimulatedObject1[equation->idxInConstrainedDOF1]);
        }
    }
}

void BaseLMConstraint::getIndicesUsed2(ConstOrder Order, helper::vector< unsigned int > &used1)
{
    const helper::vector< BaseLMConstraint::ConstraintGroup* > &constraints=constraintOrder[Order];

    for (unsigned int idxGroupConstraint=0; idxGroupConstraint<constraints.size(); ++idxGroupConstraint)
    {
        ConstraintGroup *group=constraints[idxGroupConstraint];
        std::pair< ConstraintGroup::EquationIterator, ConstraintGroup::EquationIterator > range=group->data();

        for (ConstraintGroup::EquationIterator equation=range.first; equation!=range.second; ++equation)
        {
            if (equation->idxInConstrainedDOF2 >= 0)
                used1.push_back(linesInSimulatedObject2[equation->idxInConstrainedDOF2]);
        }
    }
}


void BaseLMConstraint::getCorrections(ConstOrder Order, helper::vector<SReal>& c)
{
    const helper::vector< BaseLMConstraint::ConstraintGroup* > &constraints=constraintOrder[Order];

    for (unsigned int idxGroupConstraint=0; idxGroupConstraint<constraints.size(); ++idxGroupConstraint)
    {
        ConstraintGroup *group=constraints[idxGroupConstraint];
        std::pair< ConstraintGroup::EquationIterator, ConstraintGroup::EquationIterator > range=group->data();

        for (ConstraintGroup::EquationIterator equation=range.first; equation!=range.second; ++equation)
        {
            c.push_back(equation->correction);
        }
    }
}


void BaseLMConstraint::resetConstraint()
{
    std::map< ConstOrder, helper::vector< ConstraintGroup*> >::iterator it;
    for (it=constraintOrder.begin(); it!=constraintOrder.end(); it++)
    {
        helper::vector< ConstraintGroup* > &v=it->second;
        for (unsigned int i=0; i<v.size(); ++i)
        {
            delete v[i];
        }
    }
    constraintOrder.clear();
    linesInSimulatedObject1.clear();
    linesInSimulatedObject2.clear();
}

}
}
}
}
