/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/******************************************************************************
*  Contributors:                                                              *
*  - damien.marchal@univ-lille1.fr                                            *
******************************************************************************/
#ifndef SOFAPYTHON_TEMPLATE_H
#define SOFAPYTHON_TEMPLATE_H
#include <sofa/core/objectmodel/BaseContext.h>
using sofa::core::objectmodel::BaseObject ;

#include <SofaPython/PythonMacros.h>
SP_DECLARE_CLASS_TYPE(Template)


namespace sofa
{

namespace component
{

namespace _template_
{

class Template : public BaseObject
{

public:
    SOFA_CLASS(Template, BaseObject);

    Template() ;
    virtual ~Template() ;

    PyObject* m_rawTemplate { nullptr };
};


} // namespace _template_

using _template_::Template ;

} // namespace component

} // namespace sofa

#endif /// SOFAPYTHON_TEMPLATE_H

