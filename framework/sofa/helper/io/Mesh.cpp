/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace helper
{

namespace io
{

// commented by Sylvere
// template class Factory<std::string, Mesh, std::string>;

Mesh::Material::Material()
{
    ambient =  Vec4f( 0.2f,0.2f,0.2f,1.0f);
    diffuse =  Vec4f( 0.75f,0.75f,0.75f,1.0f);
    specular =  Vec4f( 1.0f,1.0f,1.0f,1.0f);
    emissive =  Vec4f( 0.0f,0.0f,0.0f,0.0f);

    shininess =  45.0f;
    name = "Default";
    useAmbient =  true;
    useDiffuse =  true;
    useSpecular =  false;
    useEmissive =  false;
    useShininess =  false;
    activated = false;
}

void Mesh::Material::setColor(float r, float g, float b, float a)
{
    float f[4] = { r, g, b, a };
    for (int i=0; i<4; i++)
    {
        ambient = Vec4f(f[0]*0.2f,f[1]*0.2f,f[2]*0.2f,f[3]);
        diffuse = Vec4f(f[0],f[1],f[2],f[3]);
        specular = Vec4f(f[0],f[1],f[2],f[3]);
        emissive = Vec4f(f[0],f[1],f[2],f[3]);
    }
}

Mesh* Mesh::Create(std::string filename)
{
    std::string loader="default";
    std::string::size_type p = filename.rfind('.');
    if (p!=std::string::npos)
        loader = std::string(filename, p+1);
    return FactoryMesh::CreateObject(loader, filename);
}

} // namespace io

} // namespace helper

} // namespace sofa

