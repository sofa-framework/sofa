#ifndef SOFA_COMPONENTS_XML_XML_H
#define SOFA_COMPONENTS_XML_XML_H

#include "Node.h"

namespace Sofa
{

namespace Components
{

namespace XML
{

BaseNode* load(const char *filename);

bool save(const char *filename, BaseNode* root);

} // namespace XML

} // namespace Components

} // namespace Sofa

#endif
