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
#ifndef SOFA_HELPER_COLORMAP_H
#define SOFA_HELPER_COLORMAP_H

#include <sofa/helper/config.h>

#include <sofa/type/vector.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/type/Vec.h>
#include <string>


namespace sofa::helper
{

class SOFA_HELPER_API ColorMap 
{
public:
    typedef sofa::type::vector<type::RGBAColor> VecColor;

    enum class ColorPreset : char
   {
       RED_TO_BLUE,
       BLUE_TO_RED,

       YELLOW_TO_CYAN,
       CYAN_TO_YELLOW,

       RED_TO_YELLOW,
       YELLOW_TO_RED,

       YELLOW_TO_GREEN,
       GREEN_TO_YELLOW,

       GREEN_TO_CYAN,
       CYAN_TO_GREEN,

       CYAN_TO_BLUE,
       BLUE_TO_CYAN,

       RED,
       RED_INV,

       GREEN,
       GREEN_INV,

       BLUE,
       BLUE_INV,

       HSV,

       UNKNOWN,
   };
    static const std::unordered_map<ColorPreset, std::string> colorPresetNamesMap;
    
    explicit ColorMap(unsigned int paletteSize = 256, const ColorPreset = ColorPreset::HSV);
    ColorMap(unsigned int paletteSize, const std::string& colorScheme);
    ColorMap(const sofa::type::RGBAColor& c1, const sofa::type::RGBAColor& c2);
    explicit ColorMap(const sofa::type::RGBAColor& color);

    bool buildFromColorScheme(unsigned int paletteSize = 256, const ColorPreset colorScheme = ColorPreset::HSV);

    template<class Real>
    class evaluator
    {
    public:
        evaluator()
            : map(nullptr), vmin(0), vmax(0), vscale(0)
        {}

        evaluator(const ColorMap* map, Real vmin, Real vmax)
            : map(map), vmin(vmin), vmax(vmax), vscale((vmax == vmin) ? (Real)0 : (map->m_entries.size()-1)/(vmax-vmin)) {}

        auto operator()(Real r) const
        {
            Real e = (r-vmin)*vscale;
            if (e<0) return map->m_entries.front();

            unsigned int i = (unsigned int)(e);
            if (i>=map->m_entries.size()-1) return map->m_entries.back();

            const auto& c1 = map->m_entries[i];
            const auto& c2 = map->m_entries[i+1];
            return c1+(c2-c1)*(e-i);
        }
    protected:
        const ColorMap* map;
        Real vmin;
        Real vmax;
        Real vscale;
    };

    unsigned int getNbColors() const { return (unsigned int) m_entries.size(); }
    type::RGBAColor getColor(unsigned int i) {
        if (i < m_entries.size()) return m_entries[i];
        return type::RGBAColor(0.f, 0.f, 0.f, 0.f);
    }

    static ColorMap* getDefault();

    template<class Real>
    evaluator<Real> getEvaluator(Real vmin, Real vmax) const
    {
        if (!m_entries.empty()) {
            return evaluator<Real>(this, vmin, vmax);
        } else {
            return evaluator<Real>(getDefault(), vmin, vmax);
        }
    }

    friend SOFA_HELPER_API std::ostream& operator<<(std::ostream& out, const ColorMap& m);
    friend SOFA_HELPER_API std::istream& operator>>(std::istream& in, ColorMap& m);

    void init() = delete;
    void reinit() = delete;

    unsigned int getPaletteSize() const = delete;
    void setPaletteSize(unsigned int paletteSize) = delete;

    const std::string& getColorScheme() const = delete;
    void setColorScheme(const std::string& colorScheme) = delete;

private:
    VecColor m_entries;
};

SOFA_HELPER_API std::ostream& operator<<(std::ostream& out, const ColorMap& m);
SOFA_HELPER_API std::istream& operator>>(std::istream& in, ColorMap& m);

} // namespace sofa::helper


#endif
