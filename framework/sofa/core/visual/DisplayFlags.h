#ifndef SOFA_CORE_VISUAL_DISPLAYFLAGS_H
#define SOFA_CORE_VISUAL_DISPLAYFLAGS_H
#include <sofa/core/core.h>
#include <sstream>
#include <map>

namespace sofa
{
namespace core
{
namespace visual
{

class SOFA_CORE_API DisplayFlags
{
public:

    DisplayFlags();
//VISUAL
    void setShowVisualModels(bool val) { m_showVisualModels = val;  }
    bool getShowVisualModels() const
    {
        return m_showAll
                || m_showVisual
                || m_showVisualModels;
    }
//BEHAVIOR
    void setShowBehaviorModels(bool val) { m_showBehaviorModels = val; }
    bool getShowBehaviorModels() const
    {
        return m_showAll
                || m_showBehavior
                || m_showBehaviorModels;
    }

    void setShowForceFields(bool val) { m_showForceFields = val; }
    bool getShowForceFields() const
    {
        return m_showAll
                || m_showBehavior
                || m_showForceFields;
    }

    void setShowInteractionForceFields(bool val) { m_showInteractionForceFields = val; }
    bool getShowInteractionForceFields() const
    {
        return m_showAll
                || m_showBehavior
                || m_showInteractionForceFields;
    }
//COLLISION
    void setShowCollisionModels(bool val) { m_showCollisionModels = val; }
    bool getShowCollisionModels() const
    {
        return m_showAll
                || m_showCollision
                || m_showCollisionModels;
    }

    void setShowBoundingCollisionModels(bool val) { m_showBoundingCollisionModels = val; }
    bool getShowBoundingCollisionModels() const
    {
        return m_showAll
                || m_showCollision
                || m_showBoundingCollisionModels;
    }
//MAPPINGS
    void setShowMappings(bool val) { m_showMappings = val; }
    bool getShowMappings() const
    {
        return m_showAll
                || m_showMapping
                || m_showMappings;
    }

    void setShowMechanicalMappings(bool val) { m_showMechanicalMappings= val; }
    bool getShowMechanicalMappings() const
    {
        return m_showAll
                || m_showMapping
                || m_showMechanicalMappings;
    }
//WIREFRAME
    void setShowWireFrame(bool val) {m_showWireFrame = val; }
    bool getShowWireFrame() const { return m_showWireFrame; }
//NORMALS
    void setShowNormals(bool val) {m_showNormals = val; }
    bool getShowNormals() const { return m_showAll || m_showNormals; }

#ifdef SOFA_SMP
    void setShowProcessorColor(bool val) {  m_showProcessorColor = val; }
    bool getShowProcessorColor() const { return m_showProcessorColor; }
#endif

    inline friend std::ostream& operator<< ( std::ostream& os, const DisplayFlags& flags )
    {
        return flags.write( os );
    }
    inline friend std::istream& operator>> ( std::istream& in, DisplayFlags& flags )
    {
        return flags.read( in );
    }

protected:

    inline std::ostream& write( std::ostream& os ) const
    {
        std::string s;
        s.clear();
        if(m_showWireFrame) s.append("showWireFrame ");
        if(m_showNormals) s.append("showNormals ");
#ifdef SOFA_SMP
        if(m_showProcessorColor) s.append("showProcessorColor ");
#endif

        if(m_showAll)
        {
            s.append("showAll ");
            s.erase(s.find_last_not_of(" \n\r\t")+1);
            return os << s;
        }

        if(m_showVisual)
        {
            s.append("showVisual ");
        }
        else
        {
            if(m_showVisualModels) s.append("showVisualModels ");
        }
        if(m_showBehavior)
        {
            s.append("showBehavior ");
        }
        else
        {
            if(m_showBehaviorModels) s.append("showBehaviorModels ");
            if(m_showForceFields)    s.append("showForceFields ");
            if(m_showInteractionForceFields) s.append("showInteractionForceFields ");
        }
        if(m_showCollision)
        {
            s.append("showCollision ");
        }
        else
        {
            if(m_showCollisionModels) s.append("showCollisionModels ");
            if(m_showBoundingCollisionModels) s.append("showBoundingCollisionModels ");
        }
        if(m_showMapping)
        {
            s.append("showMapping ");
        }
        else
        {
            if(m_showMechanicalMappings) s.append("showMechanicalMappings ");
            if(m_showMappings) s.append("showMappings ");
        }
        //remove trailing whitespace
        s.erase(s.find_last_not_of(" \n\r\t")+1);

        return os;
    }

    inline std::istream& read ( std::istream& in )
    {
        std::string token;
        static std::map<std::string,bool> parse_map;
        parse_map["showVisualModels"] = false;
        parse_map["showVisual"]       = false;
        parse_map["showAll"]      = false;
        parse_map["showBehavior"] = false;
        parse_map["showBehaviorModels"] = false;
        parse_map["showForceFields"] = false;
        parse_map["showInteractionForceFields"] = false;
        parse_map["showCollision"] = false;
        parse_map["showCollisionModels"] = false;
        parse_map["showBoundingCollisionModels"] = false;
        parse_map["showMapping"] = false;
        parse_map["showMechanicalMappings"] = false;
        parse_map["showMappings"] = false;
        parse_map["showNormals"] = false;
        parse_map["showWireFrame"] = false;
#ifdef SOFA_SMP
        parse_map["showProcessorColor"] = false;
#endif
        while(!in.eof())
        {
            in >> token;
            if( parse_map.find(token) != parse_map.end() )
            {
                parse_map[token] = true;
            }
        }

        m_showVisualModels = parse_map["showVisualModels"];
        m_showVisual       = parse_map["showVisual"];
        m_showAll          = parse_map["showAll"];
        m_showBehavior     = parse_map["showBehavior"];
        m_showBehaviorModels = parse_map["showBehaviorModels"];
        m_showForceFields    = parse_map["showForceFields"];
        m_showInteractionForceFields  = parse_map["showInteractionForceFields"];
        m_showCollision               = parse_map["showCollision"];
        m_showCollisionModels         = parse_map["showCollisionModels"];
        m_showBoundingCollisionModels = parse_map["showBoundingCollisionModels"];
        m_showMapping                 = parse_map["showMapping"];
        m_showMechanicalMappings      = parse_map["showMechanicalMappings"];
        m_showMappings                = parse_map["showMappings"];
        m_showNormals                 = parse_map["showNormals"];
        m_showWireFrame               = parse_map["showWireFrame"];
#ifdef SOFA_SMP
        m_showProcessorColor          = parse_map["showProcessorColor"];
#endif

        return in;

    }
protected:
    bool m_showAll;
    bool m_showVisual;
    bool m_showVisualModels;
    bool m_showBehavior;
    bool m_showBehaviorModels;
    bool m_showForceFields;
    bool m_showInteractionForceFields;
    bool m_showCollision;
    bool m_showCollisionModels;
    bool m_showBoundingCollisionModels;
    bool m_showMapping;
    bool m_showMappings;
    bool m_showMechanicalMappings;
    bool m_showNormals;
    bool m_showWireFrame;
#ifdef SOFA_SMP
    bool m_showProcessorColor;
#endif
//  static const size_t      num_entries;
//  static const std::string valid_entries[];


};


}

}

}


#endif // SOFA_CORE_VISUAL_DISPLAYFLAGS_H
