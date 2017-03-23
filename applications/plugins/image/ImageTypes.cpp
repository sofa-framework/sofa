#include <sofa/defaulttype/TemplatesAliases.h>

#include "ImageTypes.h"

namespace sofa
{
namespace defaulttype
{


#ifndef SOFA_FLOAT
RegisterTemplateAlias ImageRAlias("ImageR", "ImageD");
#else
RegisterTemplateAlias ImageRAlias("ImageR", "ImageF");
#endif

}
}
