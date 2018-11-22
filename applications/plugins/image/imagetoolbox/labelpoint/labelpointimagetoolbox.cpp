

#include "labelpointimagetoolbox.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{


int LabelPointImageToolBox_Class = core::RegisterObject("LabelPointImageToolBox")
.add< LabelPointImageToolBox >()
.addLicense("LGPL")
.addAuthor("Vincent Majorczyk");

}}}



