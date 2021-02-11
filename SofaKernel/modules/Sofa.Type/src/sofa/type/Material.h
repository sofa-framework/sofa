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

#include <sofa/type/config.h>
#include <sofa/type/RGBAColor.h>

namespace sofa::type
{

class SOFA_TYPE_API Material
{
public:
    std::string 	name;		        /* name of material */
    RGBAColor  diffuse ;	/* diffuse component */
    RGBAColor  ambient ;	/* ambient component */
    RGBAColor  specular;	/* specular component */
    RGBAColor  emissive;	/* emmissive component */
    float  shininess;	                /* specular exponent */
    bool   useDiffuse;
    bool   useSpecular;
    bool   useAmbient;
    bool   useEmissive;
    bool   useShininess;
    bool   useTexture;
    bool   useBumpMapping;
    bool   activated;
    std::string   textureFilename; // path to the texture linked to the material
    std::string   bumpTextureFilename; // path to the bump texture linked to the material

    friend SOFA_TYPE_API std::ostream& operator << (std::ostream& out, const Material& m ) ;
    friend SOFA_TYPE_API std::istream& operator >> (std::istream& in, Material &m ) ;

    void setColor(float r, float g, float b, float a) ;

    Material() ;
    Material(const Material& mat) ;
    Material & operator= (const Material& other);
};

} /// namespace sofa::type
