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
#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::Base ;

#include <sofa/core/objectmodel/BaseContext.h>
using sofa::core::objectmodel::BaseObject ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;
using sofa::core::RegisterObject ;

#include "PythonMacros.h"
#include "Binding_BaseObject.h"

#include "PythonMacros.h"

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

    PyObject* m_rawTemplate ;
};

Template::Template() : BaseObject()
{
}

Template::~Template(){}

SOFA_DECL_CLASS(Template)
int TemplateClass = core::RegisterObject("An object template encoded as parsed hson-py object.")
        .add< Template >();



} // namespace _baseprefab_

} // namespace component

} // namespace sofa


using sofa::component::_template_::Template ;

static PyObject * Template_setTemplate(PyObject *self, PyObject * args)
{
    Template* obj= ((PySPtr<Template>*)self)->object.get() ;

    obj->m_rawTemplate = nullptr ;
    if (!PyArg_ParseTuple(args, "O", &(obj->m_rawTemplate))) {
        return NULL;
    }

    return obj->m_rawTemplate ;
}

static PyObject * Template_getTemplate(PyObject *self, PyObject * args)
{
    Template* obj= ((PySPtr<Template>*)self)->object.get() ;

    return obj->m_rawTemplate ;
}


SP_CLASS_METHODS_BEGIN(Template)
SP_CLASS_METHOD(Template, setTemplate)
SP_CLASS_METHOD(Template, getTemplate)
SP_CLASS_METHODS_END


SP_CLASS_TYPE_SPTR(Template,Template,BaseObject)


