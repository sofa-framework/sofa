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
#ifndef SOFA_HELPER_IO_TRIANGLELOADER_H
#define SOFA_HELPER_IO_TRIANGLELOADER_H

#include <cstdio>
#include <sofa/helper/config.h>


namespace sofa::helper::io
{

class SOFA_HELPER_API TriangleLoader
{
public:
    virtual ~TriangleLoader() {}
    bool load(const char *filename);
    virtual void addVertices (SReal /*x*/, SReal /*y*/, SReal /*z*/) {}
    virtual void addTriangle (int /* idp1 */, int /*idp2*/, int /*idp3*/) {}

private:
    void loadTriangles(FILE *file);
};

} // namespace sofa::helper::io


#endif
