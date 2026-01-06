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

#include <sofa/core/objectmodel/BaseObject.h>


namespace sofa::core::behavior
{

class BaseRotationFinder : public virtual sofa::core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseRotationFinder, sofa::core::objectmodel::BaseObject);

     /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doGetRotations" internally,
     * which is the method to override from now on.
     *
     **/
    virtual void getRotations(linearalgebra::BaseMatrix * m, int offset = 0) final
    {
        //TODO (SPRINT SED 2025): Component state mechamism
        doGetRotations(m,offset);
    }

protected:
    virtual void doGetRotations(linearalgebra::BaseMatrix * m, int offset = 0) = 0;
};

} // namespace sofa::core::behavior
