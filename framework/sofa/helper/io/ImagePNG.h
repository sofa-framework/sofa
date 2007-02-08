#ifndef SOFA_HELPER_IO_IMAGEPNG_H
#define SOFA_HELPER_IO_IMAGEPNG_H

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

#ifdef SOFA_HAVE_PNG

class ImagePNG : public Image
{
public:
    ImagePNG ()
    {
    }

    ImagePNG (const std::string &filename)
    {
        load(filename);
    }

    bool load(const std::string &filename);
    bool save(const std::string &filename);
};

#else
// #warning not supported by MSVC
//#warning PNG not supported. Define SOFA_HAVE_PNG in sofa.cfg to activate.
#endif

} // namespace io

} // namespace helper

} // namespace sofa

#endif
