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
    : width(0), height(0), nbBits(0), data(NULL)
{
}

Image::~Image()
{
    if (data) free(data);
}

void Image::init(int w, int h, int nbb)
{
    clear();
    width = w;
    height = h;
    nbBits = nbb;
    data = (unsigned char*) malloc(((nbb+7)/8)*width*height);
}

void Image::clear()
{
    if (data!=NULL) free(data);
    width = 0;
    height = 0;
    nbBits = 0;
    data = NULL;
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
