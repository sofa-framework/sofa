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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LOADER_MESHOBJLOADER_H
#define SOFA_COMPONENT_LOADER_MESHOBJLOADER_H

#include <sofa/core/componentmodel/loader/MeshLoader.h>
#include <sofa/component/component.h>

namespace sofa
{

namespace component
{

namespace loader
{

//  using namespace sofa::defaulttype;
//  using namespace sofa::helper::io;
using sofa::defaulttype::Vec4f;


class SOFA_COMPONENT_LOADER_API MeshObjLoader : public sofa::core::componentmodel::loader::MeshLoader
{
public:

    class SOFA_COMPONENT_LOADER_API Material
    {
    public:
        std::string 	name;		/* name of material */
        Vec4f  diffuse ;	/* diffuse component */
        Vec4f  ambient ;	/* ambient component */
        Vec4f  specular;	/* specular component */
        Vec4f  emissive;	/* emmissive component */
        float  shininess;	/* specular exponent */
        bool   useDiffuse;
        bool   useSpecular;
        bool   useAmbient;
        bool   useEmissive;
        bool   useShininess;
        bool   activated;

        void setColor(float r, float g, float b, float a);

        inline friend std::ostream& operator << (std::ostream& out, const Material& m )
        {
            out   << m.name         << " ";
            out  << "Diffuse"       << " " <<  m.useDiffuse   << " " <<  m.diffuse      << " ";
            out  << "Ambient"       << " " <<  m.useAmbient   << " " <<  m.ambient      << " ";
            out  << "Specular"      << " " <<  m.useSpecular  << " " <<  m.specular     << " ";
            out  << "Emissive"      << " " <<  m.useEmissive  << " " <<  m.emissive     << " ";
            out  << "Shininess"     << " " <<  m.useShininess << " " <<  m.shininess ;
            return out;
        }
        inline friend std::istream& operator >> (std::istream& in, Material &m )
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

        Material();
    };


    MeshObjLoader();

    virtual bool load();

    template <class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        //std::cout << "MeshObjLoader::cancreate()" << std::endl;

        //      std::cout << BaseLoader::m_filename << " is not an Gmsh file." << std::endl;

        return BaseLoader::canCreate (obj, context, arg);
    }


protected:

    bool readOBJ (FILE *file, const char* filename);

    bool readMTL (const char* filename, helper::vector <Material>& materials);

    Material material;
    std::string textureName;

public:

    Data <helper::vector <Material> > materials;

    Data <helper::vector <helper::vector <int> > > texturesList;
    Data< helper::vector<sofa::defaulttype::Vector2> > texCoords;

    Data <helper::vector <helper::vector <int> > > normalsList; //TODO: check if it not the sames daata as normals and texCoords
    Data< helper::vector<sofa::defaulttype::Vector3> > normals;




    virtual std::string type()                 { return "The format of this mesh is OBJ."; }

    //    vector<Material>& getMaterialsArray() { return m_materials; }

};




} // namespace loader

} // namespace component

} // namespace sofa

#endif
