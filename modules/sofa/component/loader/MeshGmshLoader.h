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
#ifndef SOFA_COMPONENT_LOADER_MESHGMSHLOADER_H
#define SOFA_COMPONENT_LOADER_MESHGMSHLOADER_H

#include <sofa/core/componentmodel/loader/MeshLoader.h>
//#include <sofa/helper/helper.h>
#include <sofa/component/component.h>

namespace sofa
{

namespace component
{

namespace loader
{

class SOFA_COMPONENT_LOADER_API MeshGmshLoader : public sofa::core::componentmodel::loader::MeshLoader
{
public:

    virtual bool load();

    //    virtual bool canLoad(); // To Be replace by cancreate

    //    virtual bool load (const char* filename){}


    template <class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        std::cout << "MeshGmshLoader::cancreate()" << std::endl;

        //      std::cout << BaseLoader::m_filename << " is not an Gmsh file." << std::endl;

        return BaseLoader::canCreate (obj, context, arg);
    }


protected:

    bool readGmsh(FILE *file, const unsigned int gmshFormat);


public:
    // Add specific Data here:


};




} // namespace loader

} // namespace component

} // namespace sofa

#endif
