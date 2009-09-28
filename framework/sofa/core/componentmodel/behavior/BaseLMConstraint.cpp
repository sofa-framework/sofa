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
    helper::vector< constraintGroup* > &vec = constraintOrder[Order];
    for (unsigned int i=0; i<vec.size(); ++i)
    {
        result+=vec[i]->getNumConstraint();
    }
    return result;
}

BaseLMConstraint::constraintGroup* BaseLMConstraint::addGroupConstraint( ConstOrder id)
{
    constraintGroup *c=new constraintGroup(id);
    constraintOrder[id].push_back(c);
    return c;
}

void BaseLMConstraint::constraintTransmission(ConstOrder order, BaseMechanicalState* state, unsigned int entry)
{
    helper::vector< constraintGroup* > &vec = constraintOrder[order];
    for (unsigned int i=0; i<vec.size(); ++i)
    {
        constraintGroup *group=vec[i];

        if (state==getMechModel1())
        {
            helper::vector< unsigned int > &lines=group->getIndicesUsed0();
            for (unsigned int index=0; index<lines.size(); ++index)
            {
                lines[index]+=entry;
            }
        }
        if (state==getMechModel2())
        {
            helper::vector< unsigned int > &lines=group->getIndicesUsed1();
            for (unsigned int index=0; index<lines.size(); ++index)
            {
                lines[index]+=entry;
            }
        }


    }
}


void BaseLMConstraint::getIndicesUsed(ConstOrder Order, helper::vector< unsigned int > &used0,helper::vector< unsigned int > &used1)
{
    const helper::vector< BaseLMConstraint::constraintGroup* > &constraints=constraintOrder[Order];
    for (unsigned int idxGroupConstraint=0; idxGroupConstraint<constraints.size(); ++idxGroupConstraint)
    {
        const helper::vector< unsigned int > &iUsed0= constraints[idxGroupConstraint]->getIndicesUsed0();
        used0.insert(used0.end(),iUsed0.begin(), iUsed0.end());
        const helper::vector< unsigned int > &iUsed1= constraints[idxGroupConstraint]->getIndicesUsed1();
        used1.insert(used1.end(),iUsed1.begin(), iUsed1.end());
    }
}
void BaseLMConstraint::getCorrections(ConstOrder Order, helper::vector<SReal>& c)
{
    const helper::vector< BaseLMConstraint::constraintGroup* > &constraints=constraintOrder[Order];
    for (unsigned int idxGroupConstraint=0; idxGroupConstraint<constraints.size(); ++idxGroupConstraint)
    {
        const helper::vector<SReal>& correction=constraints[idxGroupConstraint]->getCorrections();
        c.insert(c.end(), correction.begin(), correction.end());
    }
}

void BaseLMConstraint::clear()
{
    std::map< ConstOrder, helper::vector< constraintGroup*> >::iterator it;
    for (it=constraintOrder.begin(); it!=constraintOrder.end(); it++)
    {
        helper::vector< constraintGroup* > &v=it->second;
        for (unsigned int i=0; i<v.size(); ++i)
        {
            delete v[i];
        }
    }
    constraintOrder.clear();
}

}
}
}
}
