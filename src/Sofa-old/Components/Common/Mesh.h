#ifndef SOFA_COMPONENTS_COMMON_MESH_H
#define SOFA_COMPONENTS_COMMON_MESH_H

#include <Sofa/Components/Common/vector.h>
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
    vector<Vector3> vertices;
    vector<Vector3> texCoords; // for the moment, we suppose that texCoords is order 2 (2 texCoords for a vertex)
    vector<Vector3> normals;
    vector< vector < vector <int> > > facets;
    Material material;

    std::string textureName;
public:

    vector<Vector3> & getVertices()
    {
        //std::cout << "vertices size : " << vertices.size() << std::endl;
        return vertices;
    };
    vector<Vector3> & getTexCoords() {return texCoords;};
    vector<Vector3> & getNormals() {return normals;};
    vector< vector < vector <int> > > & getFacets()
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
