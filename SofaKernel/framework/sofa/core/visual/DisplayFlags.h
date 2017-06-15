/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_VISUAL_DISPLAYFLAGS_H
#define SOFA_CORE_VISUAL_DISPLAYFLAGS_H

#include <sofa/core/core.h>
#include <sofa/helper/vector.h>
#include <sstream>
#include <map>


namespace sofa
{
namespace core
{
namespace visual
{

struct SOFA_CORE_API tristate
{
    enum state_t { false_value, true_value, neutral_value } state;
    tristate(bool b):state(b==true ? true_value : false_value )
    {
    }
    tristate():state(true_value)
    {
    }
    tristate(state_t state):state(state) {}

    operator bool() const
    {
        return state == true_value ? true : false;
    }

    bool operator==(const tristate& t) const
    {
        return state == t.state;
    }

    bool operator!=(const tristate& t) const
    {
        return state != t.state;
    }

    bool operator==(const state_t& s) const
    {
        return state == s;
    }

    bool operator!=(const state_t& s) const
    {
        return state != s;
    }

    friend inline tristate fusion_tristate(const tristate& lhs, const tristate& rhs);
    friend inline tristate merge_tristate(const tristate& previous, const tristate& current);
    friend inline tristate difference_tristate(const tristate& previous, const tristate& current);
};

inline tristate fusion_tristate(const tristate &lhs, const tristate &rhs)
{
    if( lhs.state == rhs.state ) return lhs;
    return tristate(tristate::neutral_value);
}

inline tristate merge_tristate(const tristate& previous, const tristate& current)
{
    if(current.state == tristate::neutral_value ) return previous;
    return current;
}
inline tristate difference_tristate(const tristate& previous, const tristate& current)
{
    if( current.state == tristate::neutral_value || current.state == previous.state )
        return tristate(tristate::neutral_value);
    return current;
}

class SOFA_CORE_API FlagTreeItem
{
protected:
    std::string m_showName;
    std::string m_hideName;
    tristate m_state;

    FlagTreeItem* m_parent;
    sofa::helper::vector<FlagTreeItem*> m_child;

    typedef helper::vector<FlagTreeItem*>::iterator ChildIterator;
    typedef helper::vector<FlagTreeItem*>::const_iterator ChildConstIterator;

public:
    FlagTreeItem(const std::string& showName, const std::string& hideName, FlagTreeItem* parent = NULL);

    const tristate& state( ) const {return m_state;}
    tristate& state() {return m_state;}

    friend std::ostream& operator<< ( std::ostream& os, const FlagTreeItem& root )
    {
        return root.write(os);
    }
    friend std::istream& operator>> ( std::istream& in, FlagTreeItem& root )
    {
        return root.read(in);
    }
    std::ostream& write(std::ostream& os) const;
    std::istream& read(std::istream& in);

    void setValue(const tristate& state);

protected:
    void propagateStateDown(FlagTreeItem* origin);
    void propagateStateUp(FlagTreeItem* origin);
    static std::map<std::string,bool> create_flagmap(FlagTreeItem* root);
    static void create_parse_map(FlagTreeItem* root, std::map<std::string,bool>& map);
    static void read_recursive(FlagTreeItem* root, const std::map<std::string,bool>& map);
    static void write_recursive(const FlagTreeItem* root,  std::string& str);
};

/** \brief Class which describes the display of components in a hierarchical fashion
* DisplayFlags are conveyed by the VisualParams, and therefore are accessible in a
* read only fashion inside a Component draw method.
* A component can tell if it should draw something on the display by looking at the current state
* of the displayFlags through the VisualParams parameter.
* DisplayFlags are embeddable inside a Data and can therefore read/write their state
* from a stream.
*
* root
* |--all
* |  |--visual
* |  |  |--visualmodels
* |  |--behavior
* |  |  |--behaviormodels
* |  |  |--forcefields
* |  |  |--interactionforcefields
* |  |--collision
* |  |  |--collisionmodels
* |  |  |--boundingcollisionmodels
* |  |--mapping
* |  |  |--visualmappings
* |  |  |--mechanicalmappings
* |--options
* |  |--advancedrendering
* |  |--wireframe
* |  |--normals
*/
class SOFA_CORE_API DisplayFlags
{
public:
    DisplayFlags();
    DisplayFlags(const DisplayFlags& );
    DisplayFlags& operator=(const DisplayFlags& );
    tristate getShowAll() const { return m_showAll.state(); }
    tristate getShowVisual() const { return m_showVisual.state(); }
    tristate getShowVisualModels() const { return m_showVisualModels.state(); }
    tristate getShowBehavior() const { return m_showBehavior.state(); }
    tristate getShowBehaviorModels() const { return m_showBehaviorModels.state(); }
    tristate getShowForceFields() const { return m_showForceFields.state(); }
    tristate getShowInteractionForceFields() const { return m_showInteractionForceFields.state(); }
    tristate getShowCollision() const { return m_showCollision.state(); }
    tristate getShowCollisionModels() const { return m_showCollisionModels.state(); }
    tristate getShowBoundingCollisionModels() const { return m_showBoundingCollisionModels.state(); }
    tristate getShowMapping() const { return m_showMapping.state(); }
    tristate getShowMappings() const { return m_showVisualMappings.state(); }
    tristate getShowMechanicalMappings() const { return m_showMechanicalMappings.state(); }
    tristate getShowOptions() const { return m_showOptions.state(); }
    tristate getShowRendering() const { return m_showRendering.state(); }
    tristate getShowWireFrame() const { return m_showWireframe.state(); }
    tristate getShowNormals() const { return m_showNormals.state(); }
#ifdef SOFA_SMP
    tristate getShowProcessorColor() const { return m_showProcessorColor.state(); }
#endif
    DisplayFlags& setShowAll(tristate v=true) { m_showAll.setValue(v); return (*this); }
    DisplayFlags& setShowVisual(tristate v=true ) { m_showVisual.setValue(v); return (*this); }
    DisplayFlags& setShowVisualModels(tristate v=true)  { m_showVisualModels.setValue(v); return (*this); }
    DisplayFlags& setShowBehavior(tristate v=true) { m_showBehavior.setValue(v); return (*this); }
    DisplayFlags& setShowBehaviorModels(tristate v=true)  { m_showBehaviorModels.setValue(v); return (*this); }
    DisplayFlags& setShowForceFields(tristate v=true)  { m_showForceFields.setValue(v); return (*this); }
    DisplayFlags& setShowInteractionForceFields(tristate v=true) { m_showInteractionForceFields.setValue(v); return (*this); }
    DisplayFlags& setShowCollision(tristate v=true ) { m_showCollisionModels.setValue(v); return (*this); }
    DisplayFlags& setShowCollisionModels(tristate v=true) { m_showCollisionModels.setValue(v); return (*this); }
    DisplayFlags& setShowBoundingCollisionModels(tristate v=true) { m_showBoundingCollisionModels.setValue(v); return (*this); }
    DisplayFlags& setShowMapping(tristate v=true) { m_showMapping.setValue(v); return (*this); }
    DisplayFlags& setShowMappings(tristate v=true) { m_showVisualMappings.setValue(v); return (*this); }
    DisplayFlags& setShowMechanicalMappings(tristate v=true) { m_showMechanicalMappings.setValue(v); return (*this); }
    DisplayFlags& setShowOptions(tristate v=true) { m_showOptions.setValue(v); return (*this); }
    DisplayFlags& setShowRendering(tristate v=true) { m_showRendering.setValue(v); return (*this); }
    DisplayFlags& setShowWireFrame(tristate v=true) { m_showWireframe.setValue(v); return (*this); }
    DisplayFlags& setShowNormals(tristate v=true) { m_showNormals.setValue(v); return (*this); }
#ifdef SOFA_SMP
    DisplayFlags& setShowProcessorColor(tristate v) const { return m_showProcessorColor.setValue(v); return (*this); }
#endif
    friend std::ostream& operator<< ( std::ostream& os, const DisplayFlags& flags )
    {
        return flags.m_root.write(os);
    }
    friend std::istream& operator>> ( std::istream& in, DisplayFlags& flags )
    {
        return flags.m_root.read(in);
    }

    bool isNeutral() const;

    friend SOFA_CORE_API DisplayFlags merge_displayFlags(const DisplayFlags& previous, const DisplayFlags& current);
    friend SOFA_CORE_API DisplayFlags difference_displayFlags(const DisplayFlags& parent, const DisplayFlags& child);
protected:
    FlagTreeItem m_root;

    FlagTreeItem m_showAll;

    FlagTreeItem m_showVisual;
    FlagTreeItem m_showVisualModels;

    FlagTreeItem m_showBehavior;
    FlagTreeItem m_showBehaviorModels;
    FlagTreeItem m_showForceFields;
    FlagTreeItem m_showInteractionForceFields;

    FlagTreeItem m_showCollision;
    FlagTreeItem m_showCollisionModels;
    FlagTreeItem m_showBoundingCollisionModels;

    FlagTreeItem m_showMapping;
    FlagTreeItem m_showVisualMappings;
    FlagTreeItem m_showMechanicalMappings;

    FlagTreeItem m_showOptions;

    FlagTreeItem m_showRendering;
    FlagTreeItem m_showWireframe;
    FlagTreeItem m_showNormals;
#ifdef SOFA_SMP
    FlagTreeItem m_showProcessorColor;
#endif
};

SOFA_CORE_API DisplayFlags merge_displayFlags(const DisplayFlags& previous, const DisplayFlags& current);
SOFA_CORE_API DisplayFlags difference_displayFlags(const DisplayFlags& parent, const DisplayFlags& child);

}

}

}

#endif // SOFA_CORE_VISUAL_DISPLAYFLAGS_H
