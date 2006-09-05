#ifndef SOFA_COMPONENTS_IMAGEPNG_H
#define SOFA_COMPONENTS_IMAGEPNG_H

#include "Common/Image.h"
#include <string>
#include <assert.h>

namespace Sofa
{

namespace Components
{

using namespace Common;

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
#warning PNG not supported. Define SOFA_HAVE_PNG in sofa.cfg to activate.
#endif

} // namespace Components

} // namespace Sofa

#endif
