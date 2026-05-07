

#include "labelboximagetoolbox.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

void registerLabelBoxImageToolBox(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("LabelBoxImageToolBox")
    .add< LabelBoxImageToolBox >()
    .addLicense("LGPL")
    .addAuthor("Vincent Majorczyk")
    );
}

}}}

