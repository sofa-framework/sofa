#include <sofa/core/visual/DisplayFlags.h>

namespace sofa
{
namespace core
{
namespace visual
{

FlagTreeItem::FlagTreeItem(const std::string& showName, const std::string& hideName, const tristate& state, FlagTreeItem* parent):
    m_showName(showName),
    m_hideName(hideName),
    m_state(state),
    m_parent(parent)
{
    if( m_parent ) m_parent->m_child.push_back(this);
    propagateStateUp(this);
}


void FlagTreeItem::setValue(const tristate &state)
{
    this->m_state = state;
    propagateStateDown(this);
    propagateStateUp(this);
}

void FlagTreeItem::propagateStateDown(FlagTreeItem* origin)
{
    ChildIterator iter;
    for( iter = origin->m_child.begin(); iter != origin->m_child.end(); ++iter)
    {
        (*iter)->m_state = origin->m_state;
        propagateStateDown(*iter);
    }
}

void FlagTreeItem::propagateStateUp(FlagTreeItem* origin)
{
    FlagTreeItem* parent = origin->m_parent;
    if(!parent) return;

    ChildIterator iter;
    tristate flag = parent->m_child[0]->m_state;
    for( iter = parent->m_child.begin(); iter != parent->m_child.end(); ++iter )
    {
        flag = fusion_tristate((*iter)->m_state,flag);
    }
    parent->m_state=flag;
    propagateStateUp(parent);
}


std::ostream& FlagTreeItem::write(std::ostream &os) const
{
    std::string s;
    write_recursive(this,s);
    s.erase(s.find_last_not_of(" \n\r\t")+1);
    os << s;
    return os;
}

std::istream& FlagTreeItem::read(std::istream &in)
{
    std::map<std::string, bool> parse_map;
    create_parse_map(this,parse_map);
    std::string token;
    while(!in.eof())
    {
        in >> token;
        if( parse_map.find(token) != parse_map.end() )
        {
            parse_map[token] = true;
        }
    }

    read_recursive(this,parse_map);
    return in;
}



/*static*/
void FlagTreeItem::create_parse_map(FlagTreeItem *root,std::map<std::string,bool>& map)
{
    map[root->m_showName] = false;
    map[root->m_hideName] = false;
    ChildIterator iter;
    for( iter = root->m_child.begin(); iter != root->m_child.end(); ++iter)
    {
        create_parse_map(*iter,map);
    }
}

void FlagTreeItem::read_recursive(FlagTreeItem *root,const std::map<std::string,bool>& parse_map)
{
    ChildIterator iter;
    std::map<std::string,bool>::const_iterator iter_show;
    std::map<std::string,bool>::const_iterator iter_hide;
    for( iter = root->m_child.begin(); iter != root->m_child.end(); ++iter)
    {
        iter_show = parse_map.find((*iter)->m_showName);
        iter_hide = parse_map.find((*iter)->m_hideName);
        if( iter_show != parse_map.end() && iter_hide != parse_map.end() )
        {
            bool show  = iter_show->second;
            bool hide  = iter_hide->second;
            if(show || hide)
            {
                tristate merge_showhide;
                if (show) merge_showhide = tristate::true_value;
                if (hide) merge_showhide = tristate::false_value;
                (*iter)->setValue(merge_showhide);
            }
            else
            {
                (*iter)->m_state = tristate::neutral_value;
                read_recursive(*iter,parse_map);
            }
        }
    }
}

void FlagTreeItem::write_recursive(const FlagTreeItem* root, std::string& str )
{
    ChildConstIterator iter;
    for( iter = root->m_child.begin(); iter != root->m_child.end(); ++iter )
    {
        switch( (*iter)->m_state.state )
        {
        case tristate::true_value:
            str.append((*iter)->m_showName);
            str.append(" ");
            break;
        case tristate::false_value:
            str.append((*iter)->m_hideName);
            str.append(" ");
            break;
        case tristate::neutral_value:
            write_recursive(*iter,str);
        }
    }
}

DisplayFlags::DisplayFlags():
    m_root(FlagTreeItem("showRoot","hideRoot",true,NULL)),
    m_showAll(FlagTreeItem("showAll","hideAll",true,&m_root)),
    m_showVisual(FlagTreeItem("showVisual","hideVisual",true,&m_showAll)),
    m_showVisualModels(FlagTreeItem("showVisualModels","hideVisualModels",true,&m_showVisual)),
    m_showBehavior(FlagTreeItem("showBehavior","hideBehavior",tristate::neutral_value,&m_showAll)),
    m_showBehaviorModels(FlagTreeItem("showBehaviorModels","hideBehaviorModels",tristate::neutral_value,&m_showBehavior)),
    m_showForceFields(FlagTreeItem("showForceFields","hideForceFields",tristate::neutral_value,&m_showBehavior)),
    m_showInteractionForceFields(FlagTreeItem("showInteractionForceFields","hideInteractionForceFields",tristate::neutral_value,&m_showBehavior)),
    m_showCollision(FlagTreeItem("showCollision","hideCollision",tristate::neutral_value,&m_showAll)),
    m_showCollisionModels(FlagTreeItem("showCollisionModels","hideCollisionModels",tristate::neutral_value,&m_showCollision)),
    m_showBoundingCollisionModels(FlagTreeItem("showBoundingCollisionModels","hideBoundingCollisionModels",tristate::neutral_value,&m_showCollision)),
    m_showMapping(FlagTreeItem("showMapping","hideMapping",tristate::neutral_value,&m_showAll)),
    m_showVisualMappings(FlagTreeItem("showMappings","hideMappings",tristate::neutral_value,&m_showMapping)),
    m_showMechanicalMappings(FlagTreeItem("showMechanicalMappings","",tristate::neutral_value,&m_showMapping)),
    m_showOptions(FlagTreeItem("showOptions","hideOptions",tristate::neutral_value,&m_root)),
    m_showWireframe(FlagTreeItem("showWireframe","hideWireframe",tristate::neutral_value,&m_showOptions)),
    m_showNormals(FlagTreeItem("showNormals","hideNormals",tristate::neutral_value,&m_showOptions))
#ifdef SOFA_SMP
    m_showProcessorColor(FlagTreeItem("showProcessorColor","hideProcessorColor",tristate::neutral_value,&m_showOptions))
#endif
{
}

DisplayFlags merge_displayFlags(const DisplayFlags &previous, const DisplayFlags &current)
{
    DisplayFlags merge;
    merge.m_showVisualModels.setValue( merge_tristate(previous.m_showVisualModels.state(),current.m_showVisualModels.state()) );
    merge.m_showBehaviorModels.setValue( merge_tristate(previous.m_showBehaviorModels.state(),current.m_showBehaviorModels.state()) );
    merge.m_showForceFields.setValue( merge_tristate(previous.m_showForceFields.state(),current.m_showForceFields.state()) );
    merge.m_showInteractionForceFields.setValue( merge_tristate(previous.m_showInteractionForceFields.state(),current.m_showInteractionForceFields.state()) );
    merge.m_showCollisionModels.setValue( merge_tristate(previous.m_showCollisionModels.state(),current.m_showCollisionModels.state()) );
    merge.m_showBoundingCollisionModels.setValue( merge_tristate(previous.m_showBoundingCollisionModels.state(),current.m_showBoundingCollisionModels.state()) );
    merge.m_showVisualMappings.setValue( merge_tristate(previous.m_showVisualMappings.state(),current.m_showVisualMappings.state()) );
    merge.m_showMechanicalMappings.setValue( merge_tristate(previous.m_showMechanicalMappings.state(),current.m_showMechanicalMappings.state()) );
    merge.m_showWireframe.setValue( merge_tristate(previous.m_showWireframe.state(),current.m_showWireframe.state()) );
    merge.m_showNormals.setValue( merge_tristate(previous.m_showNormals.state(),current.m_showNormals.state()) );
#ifdef SOFA_SMP
    merge.m_showProcessorColor.setValue( merge_tristate(previous.m_showProcessorColor.state(),current.m_showProcessorColor.state()) )
#endif
    return merge;
}

}

}

}
