#define DEPTHIMAGETOOLBOX_CPP

#include <sofa/core/ObjectFactory.h>

#include "depthimagetoolbox.h"
#include <image/ImageTypes.h>




namespace sofa
{

namespace component
{

namespace engine
{

int DepthImageToolBox_Class = core::RegisterObject("DepthImageToolBox")
.add< DepthImageToolBox >()
.addLicense("LGPL")
.addAuthor("Vincent Majorczyk");

}}}



