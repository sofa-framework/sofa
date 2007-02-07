#ifndef SOFA_SIMULATION_TREE_XML_XML_H
#define SOFA_SIMULATION_TREE_XML_XML_H

#include <sofa/simulation/tree/xml/Element.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

namespace xml
{

BaseElement* load(const char *filename);

bool save(const char *filename, BaseElement* root);

} // namespace xml

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
