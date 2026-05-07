

#include "labelpointimagetoolbox.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

void registerLabelPointImageToolBox(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("LabelPointImageToolBox")
    .add< LabelPointImageToolBox >()
    .addLicense("LGPL")
    .addAuthor("Vincent Majorczyk")
    );
}

}}}



