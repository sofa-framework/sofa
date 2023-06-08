/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/gl/component/rendering2d/config.h>

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/ColorMap.h>
#include <sofa/type/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/gl/template.h>
#include <sofa/type/Vec.h>
#include <string>


namespace sofa::gl::component::rendering2d
{

class SOFA_GL_COMPONENT_RENDERING2D_API OglColorMap : public sofa::core::visual::VisualModel
{
public:
    SOFA_CLASS(OglColorMap, sofa::core::visual::VisualModel);

    typedef type::Vec3f Color3;  // Color tripplet
    typedef sofa::type::RGBAColor Color;   // ... with alpha value
    typedef sofa::type::vector<Color> VecColor;
    
protected:
    OglColorMap();
    ~OglColorMap() override;

public:
    Data<unsigned int> d_paletteSize; ///< How many colors to use
    Data<sofa::helper::OptionsGroup> d_colorScheme; ///< Color scheme to use

    Data<bool> d_showLegend; ///< Activate rendering of color scale legend on the side
    Data<type::Vec2f> d_legendOffset; ///< Draw the legend on screen with an x,y offset
    Data<std::string> d_legendTitle; ///< Add a title to the legend
    Data<unsigned int> d_legendSize; ///< Font size of the legend (if any)
    Data<float> d_min; ///< min value for drawing the legend without the need to actually use the range with getEvaluator method wich sets the min
    Data<float> d_max; ///< max value for drawing the legend without the need to actually use the range with getEvaluator method wich sets the max
    Data<float> d_legendRangeScale; ///< to convert unit

    sofa::helper::ColorMap m_colorMap;
    GLuint texture;

    void init() override;
    void reinit() override;

    //void initVisual() { initTextures(); }
    //void clearVisual() { }
    //void initTextures() {}
    void doDrawVisual(const core::visual::VisualParams* vparams) override;
    //void drawTransparent(const VisualParams* /*vparams*/)
    //void updateVisual();



    unsigned int getNbColors() { return m_colorMap.getNbColors(); }

    Color getColor(unsigned int i) 
    {
        return m_colorMap.getColor(i);
    }

    static OglColorMap* getDefault();

    template<class Real>
    helper::ColorMap::evaluator<Real> getEvaluator(Real vmin, Real vmax)
    {
        return m_colorMap.getEvaluator(vmin, vmax);
    }

    inline friend std::ostream& operator << (std::ostream& out, const OglColorMap& m )
    {
        if (m.getName().empty()) out << "\"\"";
        else out << m.getName();
        out << " ";
        out << m.m_colorMap;
        return out;
    }

    inline friend std::istream& operator >> (std::istream& in, OglColorMap& m )
    {
        std::string name;
        in >> name;
        if (name == "\"\"") m.setName("");
        else m.setName(name);
        in >> m.m_colorMap;
        return in;
    }
};

} // namespace sofa::gl::component::rendering2d
