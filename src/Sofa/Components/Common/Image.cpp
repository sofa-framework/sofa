#include "Image.h"
#include "Factory.inl"

namespace Sofa
{

namespace Components
{

namespace Common
{

template class Factory<std::string, Image, std::string>;

Image::Image()
    : data(NULL)
{
}

Image::~Image()
{
    if (data) free(data);
}

Image* Image::Create(std::string filename)
{
    std::string loader="default";
    std::string::size_type p = filename.rfind('.');
    if (p!=std::string::npos)
        loader = std::string(filename, p+1);
    return Factory::CreateObject(loader, filename);
}

} // namespace Common

} // namespace Components

} // namespace Sofa
