#include "labelpointimagetoolbox.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(LabelPointImageToolBox)

int LabelPointImageToolBox_Class = core::RegisterObject("LabelPointImageToolBox")
.add< LabelPointImageToolBox >()
.addLicense("LGPL")
.addAuthor("Vincent Majorczyk");

}}}
