#ifndef SOFA_COMPONENTS_COMMON_MESH_H
#define SOFA_COMPONENTS_COMMON_MESH_H

#include <vector>
#include "Vec.h"
#include "Factory.h"

namespace Sofa
{

namespace Components
{

namespace Common
{

class Mesh
{
public:

    struct Material
    {
        std::string name;
        bool useDiffuse;
        double diffuse[4];
        bool useAmbient;
        double ambient[4];
        bool useSpecular;
        double specular[4];
        bool useShininess;
        double shininess;
        bool activated;

        Material();
    };

protected:
    std::vector<Vector3> vertices;
    std::vector<Vector3> texCoords; // for the moment, we suppose that texCoords is order 2 (2 texCoords for a vertex)
    std::vector<Vector3> normals;
    std::vector< std::vector < std::vector <int> > > facets;
    Material material;

    std::string textureName;
public:

    std::vector<Vector3> & getVertices()
    {
        //std::cout << "vertices size : " << vertices.size() << std::endl;
        return vertices;
    };
    std::vector<Vector3> & getTexCoords() {return texCoords;};
    std::vector<Vector3> & getNormals() {return normals;};
    std::vector< std::vector < std::vector <int> > > & getFacets()
    {
        //std::cout << "facets size : " << facets.size() << std::endl;
        return facets;
    };
    Material& getMaterial() {return material;};

    std::string& getTextureName()
    {
        return textureName;
    };

    typedef Factory<std::string, Mesh, std::string> Factory;

    static Mesh* Create(std::string filename);
};

} // namespace Common

} // namespace Components

} // namespace Sofa

#endif
