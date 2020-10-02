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
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LOADER_MESHTRIANLOADER_H
#define SOFA_COMPONENT_LOADER_MESHTRIANLOADER_H
#include "config.h"

#include <sofa/core/loader/MeshLoader.h>

namespace sofa
{

namespace component
{

namespace loader
{

/// Cette classe permet la fabrication d'un visuel pour un fichier de type trian
/// ces fichiers se presentent de la maniere suivante
/// nombre de sommets
///liste des coordonnees des sommets ex 1.45 1.25 6.85
/// nombre de faces
///liste de toutes les faces ex 1 2 3 0 0 0 les 3 derniers chiffres ne sont pas utilises pour le moment

class SOFA_GENERAL_LOADER_API MeshTrianLoader : public sofa::core::loader::MeshLoader
{
public:
    SOFA_CLASS(MeshTrianLoader,sofa::core::loader::MeshLoader);
protected:
    MeshTrianLoader();
public:
    virtual bool load();

    template <class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        return BaseLoader::canCreate (obj, context, arg);
    }

protected:

    bool readTrian(const char* filename);

    bool readTrian2(const char* filename);

public:
    //Add specific Data here:
    Data <bool> p_trian2;
    Data <helper::vector < helper::fixed_array <int,3> > > neighborTable;
    Data <helper::vector < helper::vector <unsigned int> > > edgesOnBorder;
    Data <helper::vector <unsigned int> > trianglesOnBorderList;

};


} // namespace loader

} // namespace component

} // namespace sofa

#endif
