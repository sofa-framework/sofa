#ifndef SOFA_HELPER_IO_IMAGEBMP_H
#define SOFA_HELPER_IO_IMAGEBMP_H

#include <sofa/helper/io/Image.h>
#include <string>
#include <assert.h>

namespace sofa
{

namespace helper
{

namespace io
{

//using namespace sofa::defaulttype;

class ImageBMP : public Image
{
public:
    ImageBMP ()
    {
    }

    ImageBMP (const std::string &filename)
    {
        load(filename);
    }

    bool load(const std::string &filename);
    bool save(const std::string &filename);
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
