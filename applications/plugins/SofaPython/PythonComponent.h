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
#ifndef SOFA_PYTHONCOMPONENT_H
#define SOFA_PYTHONCOMPONENT_H
#include <sofa/core/objectmodel/BaseContext.h>
using sofa::core::objectmodel::BaseObject ;

#include <sofa/helper/vector.h>

#include <sofa/core/DataTracker.h>

#include <SofaPython/PythonMacros.h>
SP_DECLARE_CLASS_TYPE(Python)

namespace sofa {
    namespace core {
        namespace objectmodel {
            class BaseData ;
        }
    }
}

namespace sofa
{

namespace component
{

namespace _pythoncomponent_
{
using sofa::core::objectmodel::BaseData;
using sofa::core::objectmodel::Event;
using sofa::core::DataTracker;
using sofa::helper::vector ;

class PythonComponent : public BaseObject
{
public:
    SOFA_CLASS(PythonComponent, BaseObject);

    PythonComponent() ;
    virtual ~PythonComponent() ;

    PyObject* m_rawPython { nullptr };
    Data<std::string> m_source  ;

    void addDataToTrack(BaseData*) ;
};


} // namespace _pythoncomponent_

namespace python {
    using _pythoncomponent_::PythonComponent ;
}

} // namespace component

} // namespace sofa

#endif /// SOFA_PYTHONCOMPONENT_H

