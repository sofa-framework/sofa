/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
/******************************************************************************
* Contributors:                                                               *
*     - damien.marchal@univ-lille1.fr                                         *
******************************************************************************/
#ifndef SOFA_INFOCOMPONENT_H
#define SOFA_INFOCOMPONENT_H

#include <sofa/core/objectmodel/BaseObject.h>
#include "config.h"
#include <string>

namespace sofa
{
namespace component
{

/// I use a per-file namespace so that I can employ the 'using' keywords without
/// fearing it will leack names into the global namespace. When closing this namespace
/// selected object from this per-file namespace are then imported into their parent namespace.
/// for ease of use
namespace infocomponent
{
using sofa::core::objectmodel::BaseObject ;

/// Despite this component does absolutely nothin... it is very usefull as it can be used to
/// retain information scene graph.
class SOFA_COMPONENT_BASE_API InfoComponent : public BaseObject
{
public:
    SOFA_CLASS(InfoComponent, BaseObject);

    InfoComponent() {}
    virtual ~InfoComponent(){}
};

}

/// Import the component from the per-file namespace.
using infocomponent::InfoComponent ;

}
}
#endif // SOFA_INFOCOMPONENT_H
