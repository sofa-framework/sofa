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
#ifndef SOFA_HELPER_IO_MESHTRIAN_H
#define SOFA_HELPER_IO_MESHTRIAN_H

#include <sofa/helper/io/Mesh.h>

namespace sofa
{

namespace helper
{

namespace io
{

/// Cette classe permet la fabrication d'un visuel pour un fichier de type trian
/// ces fichiers se presentent de la maniere suivante
/// nombre de sommets
///liste des coordonnees des sommets ex 1.45 1.25 6.85
/// nombre de faces
///liste de toutes les faces ex 1 2 3 0 0 0 les 3 derniers chiffres ne sont pas utilises pour le moment

class SOFA_HELPER_API MeshTrian : public Mesh
{
private:

    void readTrian(FILE *file);

public:

    MeshTrian(const std::string& filename)
    {
        init(filename);
    }

    void init(std::string filename);
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
