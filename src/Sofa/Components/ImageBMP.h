#ifndef SOFA_COMPONENTS_IMAGEBMP_H
#define SOFA_COMPONENTS_IMAGEBMP_H

#include "Common/Image.h"
#include <string>
#include <assert.h>

namespace Sofa
{

namespace Components
{

using namespace Common;

class ImageBMP : public Image
{
public:
    ImageBMP (const std::string &filename)
    {
        init (filename);
    }

    void init (const std::string &filename);
};

} // namespace Components

} // namespace Sofa

#endif
