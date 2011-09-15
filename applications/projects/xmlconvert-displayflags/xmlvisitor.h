#ifndef SOFA_XMLCONVERT_XMLVISITOR
#define SOFA_XMLCONVERT_XMLVISITOR

#include <tinyxml.h>
#include <map>
#include <vector>
#include <sofa/core/visual/DisplayFlags.h>

namespace sofa
{
namespace xml
{

class DiscoverNodes : public TiXmlVisitor
{
public:

    virtual bool VisitEnter( const TiXmlElement&, const TiXmlAttribute* );

    TiXmlElement* rootNode()
    {
        if( nodes.empty() ) return NULL;
        else return nodes[0];
    }

    std::vector<TiXmlElement* > leaves;
    std::vector<TiXmlElement* > nodes;

};

class DiscoverDisplayFlagsVisitor : public TiXmlVisitor
{
public:

    virtual bool VisitEnter( const TiXmlElement&, const TiXmlAttribute* );
    ~DiscoverDisplayFlagsVisitor();
    std::map<const TiXmlElement*,sofa::core::visual::DisplayFlags*> map_displayFlags;

};

void createVisualStyleVisitor(TiXmlElement* leaf, const std::map<const TiXmlElement*,sofa::core::visual::DisplayFlags*>& map_displayFlags);

void removeShowAttributes(TiXmlElement* node);

void convert_false_to_neutral(sofa::core::visual::DisplayFlags& flags);


}
}


#endif
