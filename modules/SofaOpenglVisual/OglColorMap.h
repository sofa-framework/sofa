/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_VISUALMODEL_OGLCOLORMAP_H
#define SOFA_COMPONENT_VISUALMODEL_OGLCOLORMAP_H
#include "config.h"

#ifndef SOFA_NO_OPENGL

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/ColorMap.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/Vec.h>
#include <string>


namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_OPENGL_VISUAL_API OglColorMap : public sofa::core::visual::VisualModel
{
public:
    SOFA_CLASS(OglColorMap, sofa::core::visual::VisualModel);

    typedef defaulttype::Vec3f Color3;  // Color tripplet
    typedef defaulttype::Vec4f Color;   // ... with alpha value
    typedef sofa::helper::vector<Color> VecColor;
    
protected:
    OglColorMap();
    virtual ~OglColorMap();

public:
    Data<unsigned int> f_paletteSize; ///< How many colors to use
    Data<sofa::helper::OptionsGroup> f_colorScheme; ///< Color scheme to use

    Data<bool> f_showLegend; ///< Activate rendering of color scale legend on the side
    Data<defaulttype::Vec2f> f_legendOffset; ///< Draw the legend on screen with an x,y offset
    Data<std::string> f_legendTitle; ///< Add a title to the legend
    Data<float> d_min; ///< min value for drawing the legend without the need to actually use the range with getEvaluator method wich sets the min
    Data<float> d_max; ///< max value for drawing the legend without the need to actually use the range with getEvaluator method wich sets the max
    Data<float> d_legendRangeScale; ///< to convert unit

    sofa::helper::ColorMap m_colorMap;
    GLuint texture;

    void initOld(const std::string &data);

    void init() override;
    void reinit() override;

    //void initVisual() { initTextures(); }
    //void clearVisual() { }
    //void initTextures() {}
    void drawVisual(const core::visual::VisualParams* vparams) override;
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

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif /* SOFA_NO_OPENGL */

#endif
