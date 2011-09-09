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

class FlagTreeItem;

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
* |  |--wireframe
* |  |--normals
*/
class SOFA_CORE_API DisplayFlags
{
public:
    DisplayFlags();
    bool getShowVisualModels() const { return m_showVisualModels.state(); }
    bool getShowBehaviorModels() const { return m_showBehaviorModels.state(); }
    bool getShowForceFields() const { return m_showForceFields.state(); }
    bool getShowInteractionForceFields() const { return m_showInteractionForceFields.state(); }
    bool getShowCollisionModels() const { return m_showCollisionModels.state(); }
    bool getShowBoundingCollisionModels() const { return m_showBoundingCollisionModels.state(); }
    bool getShowMappings() const { return m_showVisualMappings.state(); }
    bool getShowMechanicalMappings() const { return m_showMechanicalMappings.state(); }
    bool getShowWireFrame() const { return m_showWireframe.state(); }
    bool getShowNormals() const { return m_showNormals.state(); }
#ifdef SOFA_SMP
    bool getShowProcessorColor() const { return m_showProcessorColor.state(); }
#endif
    void setShowVisualModels(tristate v)  { m_showVisualModels.setValue(v) ; }
    void setShowBehaviorModels(tristate v)  { m_showBehaviorModels.setValue(v) ; }
    void setShowForceFields(tristate v)  { m_showForceFields.setValue(v) ; }
    void setShowInteractionForceFields(tristate v) { m_showInteractionForceFields.setValue(v) ; }
    void setShowCollisionModels(tristate v) { m_showCollisionModels.setValue(v) ; }
    void setShowBoundingCollisionModels(tristate v) { m_showBoundingCollisionModels.setValue(v) ; }
    void setShowMappings(tristate v) { m_showVisualMappings.setValue(v) ; }
    void setShowMechanicalMappings(tristate v) { m_showMechanicalMappings.setValue(v) ; }
    void setShowWireFrame(tristate v) { m_showWireframe.setValue(v) ; }
    void setShowNormals(tristate v) { m_showNormals.setValue(v) ; }
#ifdef SOFA_SMP
    void setShowProcessorColor(tristate v) const { return m_showProcessorColor.setValue(v); }
#endif
    friend std::ostream& operator<< ( std::ostream& os, const DisplayFlags& flags )
    {
        return flags.m_root.write(os);
    }
    friend std::istream& operator>> ( std::istream& in, DisplayFlags& flags )
    {
        return flags.m_root.read(in);
    }

    friend DisplayFlags merge_displayFlags(const DisplayFlags& previous, const DisplayFlags& current);

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

    FlagTreeItem m_showWireframe;
    FlagTreeItem m_showNormals;
#ifdef SOFA_SMP
    FlagTreeItem m_showProcessorColor;
#endif
};

DisplayFlags SOFA_CORE_API merge_displayFlags(const DisplayFlags& previous, const DisplayFlags& current);

struct tristate
{
    enum state_t { false_value, true_value, neutral_value } state;
    tristate(bool b):state(b ? state = true_value : state = false_value )
    {
    }
    tristate():state(true_value)
    {
    }
    tristate(state_t state):state(state) {}
    bool operator==(const tristate& other) const
    {
        return (this->state == other.state);
    }
    bool operator!=(const tristate& other) const
    {
        return !(*this==other);
    }
    operator bool() const
    {
        return state == true_value ? true : false;
    }
    friend inline tristate fusion_tristate(const tristate& lhs, const tristate& rhs);
    friend inline tristate merge_tristate(const tristate& previous, const tristate& current);
};

inline tristate fusion_tristate(const tristate &lhs, const tristate &rhs)
{
    if( lhs == rhs  ) return lhs;
    else return tristate(tristate::neutral_value);
}

inline tristate merge_tristate(const tristate& previous, const tristate& current)
{
    if(current.state == tristate::neutral_value ) return previous;
    else return current;
}

class FlagTreeItem
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
    FlagTreeItem(const std::string& showName, const std::string& hideName, const tristate& state, FlagTreeItem* parent = NULL);

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

}

}

}

#endif // SOFA_CORE_VISUAL_DISPLAYFLAGS_H
