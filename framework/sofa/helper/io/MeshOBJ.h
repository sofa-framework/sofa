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
#ifndef SOFA_HELPER_IO_MESHOBJ_H
#define SOFA_HELPER_IO_MESHOBJ_H

#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/helper.h>
#include <fstream>

namespace sofa
{

namespace helper
{

namespace io
{

using namespace sofa::defaulttype;

class SOFA_HELPER_API MeshOBJ : public Mesh
{
private:

    std::vector<Material> materials;

    void readOBJ (std::ifstream &file, const std::string &filename);
    void readMTL (const char *filename);

public:

    MeshOBJ(const std::string& filename)
    {
        init (filename);
    }

    void init (std::string filename);
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
