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
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa::core::loader
{

bool SOFA_CORE_API canLoad(const char* filename);

class SOFA_CORE_API BaseLoader : public objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseLoader, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseLoader)

    virtual bool load() = 0;
    virtual bool canLoad() ;

    void parse(objectmodel::BaseObjectDescription *arg) override ;

    void setFilename(std::string f)  ;
    const std::string &getFilename() ;

    objectmodel::DataFileName d_filename;

protected:
    BaseLoader() ;
    ~BaseLoader() override ;

    static void skipToEOL(FILE* f) ;
    static bool readLine(char* buf, int size, FILE* f) ;
};

} /// namespace sofa::core::loader

