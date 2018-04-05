/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#ifndef SOFA_HELPER_TYPES_MATERIAL_H_
#define SOFA_HELPER_TYPES_MATERIAL_H_

#include <sofa/core/core.h>
#include <sofa/defaulttype/RGBAColor.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa
{

namespace helper
{

namespace types
{

class SOFA_HELPER_API Material
{
public:
    std::string 	name;		        /* name of material */
    defaulttype::RGBAColor  diffuse ;	/* diffuse component */
    defaulttype::RGBAColor  ambient ;	/* ambient component */
    defaulttype::RGBAColor  specular;	/* specular component */
    defaulttype::RGBAColor  emissive;	/* emmissive component */
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

    void setColor(float r, float g, float b, float a) ;

    friend SOFA_HELPER_API std::ostream& operator << (std::ostream& out, const Material& m ) ;
    friend SOFA_HELPER_API std::istream& operator >> (std::istream& in, Material &m ) ;
    Material() ;
    Material(const Material& mat) ;
};

} // namespace types

} // namespace helper

} // namespace sofa

#endif /* SOFA_HELPER_TYPES_MATERIAL_H_ */
