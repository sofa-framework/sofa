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
#ifndef SOFA_HELPER_COLORMAP_H
#define SOFA_HELPER_COLORMAP_H

#include <sofa/helper/helper.h>

#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/defaulttype/Vec.h>
#include <string>
//#include <sofa/helper/OptionsGroup.h>


namespace sofa
{

namespace helper
{
    
class SOFA_HELPER_API ColorMap 
{
public:

    typedef defaulttype::Vec3f Color3;  // Color tripplet
    typedef defaulttype::Vec4f Color;   // ... with alpha value
    typedef sofa::helper::vector<Color> VecColor;
    
    ColorMap(unsigned int paletteSize = 256, const std::string& colorScheme = "HSV");
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

    unsigned int m_paletteSize;
    std::string m_colorScheme;
    
    VecColor entries;
    
    void init();
    void reinit();
    
    unsigned int getPaletteSize() { return m_paletteSize;  }
    void setPaletteSize(unsigned int paletteSize) { m_paletteSize = paletteSize; }

    const std::string& getColorScheme() { return m_colorScheme;  }
    void setColorScheme(const std::string& colorScheme) { m_colorScheme = colorScheme; }

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
        out << m.entries;
        return out;
    }

    inline friend std::istream& operator >> (std::istream& in, ColorMap& m )
    {
        in >> m.entries;
        return in;
    }
};


} // namespace helper

} // namespace sofa


#endif
