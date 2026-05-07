

#include "labelpointsbysectionimagetoolbox.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

void registerLabelPointsBySectionImageToolBox(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("LabelPointsBySectionImageToolBox")
    .add< LabelPointsBySectionImageToolBox >()
    .addLicense("LGPL")
    .addAuthor("Vincent Majorczyk")
    );
}

}}}


