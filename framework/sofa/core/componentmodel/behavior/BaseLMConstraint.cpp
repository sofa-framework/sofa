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

unsigned int BaseLMConstraint::getNumConstraint(ConstId Id)
{
    unsigned int result=0;
    std::vector< constraintGroup > &vec = constraintId[Id];
    for (unsigned int i=0; i<vec.size(); ++i)
    {
        result+=vec[i].getNumConstraint();
    }
    return result;
}

BaseLMConstraint::constraintGroup* BaseLMConstraint::addGroupConstraint( ConstId Id)
{
    constraintId[Id].push_back(constraintGroup(Id));
    return &(constraintId[Id][constraintId[Id].size()-1]);
}

void BaseLMConstraint::getIndicesUsed(ConstId Id, std::vector< unsigned int > &used0,std::vector< unsigned int > &used1)
{
    const std::vector< BaseLMConstraint::constraintGroup > &constraints=constraintId[Id];
    for (unsigned int idxGroupConstraint=0; idxGroupConstraint<constraints.size(); ++idxGroupConstraint)
    {
        const std::vector< unsigned int > &iUsed0= constraints[idxGroupConstraint].getIndicesUsed0();
        used0.insert(used0.end(),iUsed0.begin(), iUsed0.end());
        const std::vector< unsigned int > &iUsed1= constraints[idxGroupConstraint].getIndicesUsed1();
        used1.insert(used1.end(),iUsed1.begin(), iUsed1.end());
    }
}
void BaseLMConstraint::getCorrections(ConstId Id, std::vector<SReal>& c)
{
    const std::vector< BaseLMConstraint::constraintGroup > &constraints=constraintId[Id];
    for (unsigned int idxGroupConstraint=0; idxGroupConstraint<constraints.size(); ++idxGroupConstraint)
    {
        const std::vector<SReal>& correction=constraints[idxGroupConstraint].getCorrections();
        c.insert(c.end(), correction.begin(), correction.end());
    }
}

}
}
}
}
