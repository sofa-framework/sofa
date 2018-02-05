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
#include <sofa/helper/types/Material.h>

namespace sofa
{

namespace helper
{

namespace types
{

    void Material::setColor(float r, float g, float b, float a)
    {
        ambient = defaulttype::RGBAColor(r*0.2f,g*0.2f,b*0.2f,a);
        diffuse = defaulttype::RGBAColor(r,g,b,a);
        specular = defaulttype::RGBAColor(r,g,b,a);
        emissive = defaulttype::RGBAColor(r,g,b,a);
    }

    SOFA_HELPER_API std::ostream&  operator << (std::ostream& out, const Material& m )
    {
        out   << m.name         << " ";
        out  << "Diffuse"       << " " <<  m.useDiffuse   << " " <<  m.diffuse      << " ";
        out  << "Ambient"       << " " <<  m.useAmbient   << " " <<  m.ambient      << " ";
        out  << "Specular"      << " " <<  m.useSpecular  << " " <<  m.specular     << " ";
        out  << "Emissive"      << " " <<  m.useEmissive  << " " <<  m.emissive     << " ";
        out  << "Shininess"     << " " <<  m.useShininess << " " <<  m.shininess   << " ";
        return out;
    }

    SOFA_HELPER_API std::istream&  operator >> (std::istream& in, Material &m )
    {
        std::string element;
        in  >>  m.name ;
        for (unsigned int i=0; i<5; ++i)
        {
            in  >>  element;
            if      (element == std::string("Diffuse")   || element == std::string("diffuse")   ) { in  >>  m.useDiffuse   ; in >> m.diffuse;   }
            else if (element == std::string("Ambient")   || element == std::string("ambient")   ) { in  >>  m.useAmbient   ; in >> m.ambient;   }
            else if (element == std::string("Specular")  || element == std::string("specular")  ) { in  >>  m.useSpecular  ; in >> m.specular;  }
            else if (element == std::string("Emissive")  || element == std::string("emissive")  ) { in  >>  m.useEmissive  ; in >> m.emissive;  }
            else if (element == std::string("Shininess") || element == std::string("shininess") ) { in  >>  m.useShininess ; in >> m.shininess; }
        }
        return in;
    }

     Material::Material()
{
    ambient =  defaulttype::RGBAColor( 0.2f,0.2f,0.2f,1.0f);
    diffuse =  defaulttype::RGBAColor( 0.75f,0.75f,0.75f,1.0f);
    specular =  defaulttype::RGBAColor( 1.0f,1.0f,1.0f,1.0f);
    emissive =  defaulttype::RGBAColor( 0.0f,0.0f,0.0f,0.0f);

    shininess =  45.0f;
    name = "Default";
    useAmbient =  true;
    useDiffuse =  true;
    useSpecular =  false;
    useEmissive =  false;
    useShininess =  false;
    activated = false;

    useTexture = false;
    textureFilename ="DEFAULT";

    useBumpMapping = false;
    bumpTextureFilename ="DEFAULT";
}

Material::Material(const Material& mat)
{
    ambient =  mat.ambient;
    diffuse =  mat.diffuse;
    specular =  mat.specular;
    emissive =  mat.emissive;

    shininess =  mat.shininess;
    name = mat.name;
    useAmbient =  mat.useAmbient;
    useDiffuse =  mat.useDiffuse ;
    useSpecular =  mat.useSpecular ;
    useEmissive =  mat.useEmissive;
    useShininess =  mat.useShininess ;
    activated = mat.activated;

    useTexture = mat.useTexture;
    textureFilename = mat.textureFilename;

    useBumpMapping = mat.useBumpMapping;
    bumpTextureFilename = mat.bumpTextureFilename;
}

} // namespace loader

} // namespace core

} // namespace sofa

