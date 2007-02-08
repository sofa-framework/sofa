#include <sofa/helper/io/Image.h>
#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace helper
{

template class Factory<std::string, sofa::helper::io::Image, std::string>;

namespace io
{



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

} // namespace io

} // namespace helper

} // namespace sofa

