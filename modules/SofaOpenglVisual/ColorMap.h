/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_VISUALMODEL_COLORMAP_H
#define SOFA_COMPONENT_VISUALMODEL_COLORMAP_H
#include "config.h"

#ifndef SOFA_NO_OPENGL

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/helper/OptionsGroup.h>
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

class SOFA_OPENGL_VISUAL_API ColorMap : public sofa::core::visual::VisualModel
{
public:
    SOFA_CLASS(ColorMap, sofa::core::visual::VisualModel);

    typedef defaulttype::Vec3f Color3;  // Color tripplet
    typedef defaulttype::Vec4f Color;   // ... with alpha value
    typedef sofa::helper::vector<Color> VecColor;
    
protected:
    ColorMap();
    virtual ~ColorMap();

public:
    template<class Real>
    class evaluator
    {
    public:
        evaluator()
            : map(NULL), vmin(0), vmax(0), vscale(0)
        {}

        evaluator(const ColorMap* map, Real vmin, Real vmax)
            : map(map), vmin(vmin), vmax(vmax), vscale((vmax == vmin) ? (Real)0 : (map->entries.size()-1)/(vmax-vmin)) {}

        Color operator()(Real r) const
        {
            Real e = (r-vmin)*vscale;
            if (e<0) return map->entries.front();

            unsigned int i = (unsigned int)(e);
            if (i>=map->entries.size()-1) return map->entries.back();

            Color c1 = map->entries[i];
            Color c2 = map->entries[i+1];
            return c1+(c2-c1)*(e-i);
        }
    protected:
        const ColorMap* map;
        Real vmin;
        Real vmax;
        Real vscale;
    };

    Data<unsigned int> f_paletteSize;
    Data<sofa::helper::OptionsGroup> f_colorScheme;

    Data<bool> f_showLegend;
    Data<defaulttype::Vec2f> f_legendOffset;
    Data<std::string> f_legendTitle;
    Data<float> d_min, d_max;
    Data<float> d_legendRangeScale; ///< to convert unit

    VecColor entries;
    GLuint texture;

    void initOld(const std::string &data);

    void init();
    void reinit();

    //void initVisual() { initTextures(); }
    //void clearVisual() { }
    //void initTextures() {}
    void drawVisual(const core::visual::VisualParams* vparams);
    //void drawTransparent(const VisualParams* /*vparams*/)
    //void updateVisual();



    unsigned int getNbColors() { return (unsigned int) entries.size(); }
    Color getColor(unsigned int i) {
        if (i < entries.size()) return entries[i];
        return Color(0.0, 0.0, 0.0, 0.0);
    }

    static ColorMap* getDefault();

    template<class Real>
    evaluator<Real> getEvaluator(Real vmin, Real vmax)
    {
        if (!entries.empty()) {
            return evaluator<Real>(this, vmin, vmax);
        } else {
            return evaluator<Real>(getDefault(), vmin, vmax);
        }
    }

    Color3 hsv2rgb(const Color3 &hsv);

    inline friend std::ostream& operator << (std::ostream& out, const ColorMap& m )
    {
        if (m.getName().empty()) out << "\"\"";
        else out << m.getName();
        out << " ";
        out << m.entries;
        return out;
    }

    inline friend std::istream& operator >> (std::istream& in, ColorMap& m )
    {
        std::string name;
        in >> name;
        if (name == "\"\"") m.setName("");
        else m.setName(name);
        in >> m.entries;
        return in;
    }
};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif /* SOFA_NO_OPENGL */

#endif
