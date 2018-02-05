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
/******************************************************************************
*  Contributors:                                                              *
*  - damien.marchal@univ-lille1.fr                                            *
******************************************************************************/
#ifndef SOFA_APIVERSION_H
#define SOFA_APIVERSION_H
#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include "config.h"

namespace sofa
{

namespace component
{

namespace _apiversion_
{

class SOFA_GRAPH_COMPONENT_API APIVersion : public BaseObject
{

public:
    SOFA_CLASS(APIVersion, BaseObject);

    const std::string& getApiLevel() ;
    virtual void init() override ;

protected:
    APIVersion() ;
    virtual ~APIVersion() ;
    void checkInputData() ;
private:
    Data<std::string>  d_level ;
};

} // namespace _apiversion_

using _apiversion_::APIVersion ;

} // namespace component

} // namespace sofa

#endif /// SOFA_APIVERSION_H
