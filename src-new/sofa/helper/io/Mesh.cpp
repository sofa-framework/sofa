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
    activated = false;
    useDiffuse = false;
    useAmbient = false;
    useSpecular = false;
    useShininess = false;
    for (int i = 0; i < 3; i++)
    {
        diffuse[i] = 0.0;
        ambient[i] = 0.0;
        specular[i] = 0.0;
    }
    diffuse[3] = 1.0;
    ambient[3] = 1.0;
    specular[3] = 1.0;
    shininess = 0.0;
}

Mesh* Mesh::Create(std::string filename)
{
    std::string loader="default";
    std::string::size_type p = filename.rfind('.');
    if (p!=std::string::npos)
        loader = std::string(filename, p+1);
    return Factory::CreateObject(loader, filename);
}

} // namespace io

} // namespace helper

} // namespace sofa

