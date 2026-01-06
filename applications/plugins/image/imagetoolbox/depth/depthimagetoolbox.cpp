#define DEPTHIMAGETOOLBOX_CPP

#include "depthimagetoolbox.h"
#include <sofa/core/ObjectFactory.h>


#include <image/ImageTypes.h>




namespace sofa
{

namespace component
{

namespace engine
{

void registerDepthImageToolBox(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("DepthImageToolBox")
    .add< DepthImageToolBox >()
    .addLicense("LGPL")
    .addAuthor("Vincent Majorczyk")
    );
}

}}}



