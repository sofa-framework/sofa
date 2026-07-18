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

#include <sofa/helper/ColorMap.h>
#include <sofa/helper/logging/Messaging.h>

#include <iostream>
#include <optional>
#include <string>

#include "StringUtils.h"

namespace sofa::helper
{

const std::unordered_map<ColorMap::ColorPreset, std::string> ColorMap::colorPresetNamesMap {
    {ColorMap::ColorPreset::RED_TO_BLUE, "Red to Blue"},
    {ColorMap::ColorPreset::BLUE_TO_RED, "Blue to Red"},
    {ColorMap::ColorPreset::YELLOW_TO_CYAN, "Yellow to Cyan"},
    {ColorMap::ColorPreset::CYAN_TO_YELLOW, "Cyan to Yellow"},
    {ColorMap::ColorPreset::RED_TO_YELLOW, "Red to Yello"},
    {ColorMap::ColorPreset::YELLOW_TO_RED, "Yellow to Red"},
    {ColorMap::ColorPreset::YELLOW_TO_GREEN, "Yellow to Green"},
    {ColorMap::ColorPreset::GREEN_TO_YELLOW, "Green to Yellow"},
    {ColorMap::ColorPreset::GREEN_TO_CYAN, "Green to Cyan"},
    {ColorMap::ColorPreset::CYAN_TO_GREEN, "Cyan to Green"},
    {ColorMap::ColorPreset::CYAN_TO_BLUE, "Cyan to Blue"},
    {ColorMap::ColorPreset::BLUE_TO_CYAN, "Blue to Cyan"},
    {ColorMap::ColorPreset::RED, "Red"},
    {ColorMap::ColorPreset::RED_INV, "RedInv"},
    {ColorMap::ColorPreset::GREEN, "Green"},
    {ColorMap::ColorPreset::GREEN_INV, "GreenInv"},
    {ColorMap::ColorPreset::BLUE, "Blue"},
    {ColorMap::ColorPreset::BLUE_INV, "BlueInv"},
    {ColorMap::ColorPreset::HSV, "HSV"}
};

namespace
{

std::optional<ColorMap::ColorPreset> stringToColorScheme(const std::string& input)
{
    const auto it = std::find_if(ColorMap::colorPresetNamesMap.begin(), ColorMap::colorPresetNamesMap.end(),
        [&input](const auto& entry)
        {
            return entry.second == input;
        });

    if (it != ColorMap::colorPresetNamesMap.end())
        return it->first;
    return {};
}

}

ColorMap* ColorMap::getDefault()
{
    static std::unique_ptr<ColorMap> defaultColorMap { nullptr };
    if (defaultColorMap == nullptr)
    {
        defaultColorMap = std::make_unique<ColorMap>();
    }
    return defaultColorMap.get();
}

ColorMap::ColorMap(const sofa::type::RGBAColor& c1, const sofa::type::RGBAColor& c2)
{
    m_entries = VecColor{ c1, c2 };
}

ColorMap::ColorMap(const sofa::type::RGBAColor& color) : ColorMap(color, color)
{}

ColorMap::ColorMap(unsigned int paletteSize, const ColorPreset colorScheme)
{
    buildFromColorScheme(paletteSize, colorScheme);
}

ColorMap::ColorMap(unsigned int paletteSize, const std::string& colorScheme)
{
    auto opt = stringToColorScheme(colorScheme);
    buildFromColorScheme(paletteSize, opt.value_or(ColorPreset::HSV));
}

bool ColorMap::buildFromColorScheme(unsigned int paletteSize, const ColorPreset colorScheme)
{
    m_entries.clear();

    if (paletteSize < 2)
    {
        dmsg_warning("ColorMap") << "Palette size must be greater than or equal to 2." << msgendl
                                << "Using the default value of '256'. ";
        paletteSize = 256;
    }

    m_entries.reserve(paletteSize);

    if (colorScheme == ColorPreset::RED_TO_BLUE)
    {
        // List the colors
        const float step = (2.0f / 3.0f) / (paletteSize - 1);
        for (unsigned int i = 0; i < paletteSize; i++)
        {
            const auto h = i * step * 360;
            m_entries.push_back(type::RGBAColor::fromHSVA(h, 1.0f, 1.0f, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::BLUE_TO_RED)
    {
        // List the colors
        const float step = (2.0f / 3.0f) / (paletteSize - 1);
        for (unsigned int i = 0; i < paletteSize; i++)
        {
            const auto h = (2.0f / 3.0f - i * step) * 360;
            m_entries.push_back(type::RGBAColor::fromHSVA(h, 1.0f, 1.0f, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::YELLOW_TO_CYAN)
    {
        // List the colors
        const float step = (0.5f - 1.0f / 6.0f) / (paletteSize - 1);
        for (unsigned int i = 0; i < paletteSize; i++)
        {
            const auto h = (1.0f / 6.0f + i * step) * 360;
            m_entries.push_back(type::RGBAColor::fromHSVA(h, 1.0f, 1.0f, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::CYAN_TO_YELLOW)
    {
        // List the colors
        const float step = (0.5f - 1.0f / 6.0f) / (paletteSize - 1);
        for (unsigned int i = 0; i < paletteSize; i++)
        {
            const auto h = (0.5f - i * step) * 360;
            m_entries.push_back(type::RGBAColor::fromHSVA(h, 1.0f, 1.0f, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::RED_TO_YELLOW)
    {
        const float step = 1.0f / (paletteSize);
        for (unsigned int i = 0; i < paletteSize; i++)
        {
            m_entries.push_back(type::RGBAColor(1.0f, i * step, 0.0f, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::YELLOW_TO_RED)
    {
        const float step = 1.0f / (paletteSize);
        for (unsigned int i = 0; i < paletteSize; i++)
        {
            m_entries.push_back(type::RGBAColor(1.0f, 1.0f - i * step, 0.0f, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::YELLOW_TO_GREEN)
    {
        const float step = 1.0f / (paletteSize);
        for (unsigned int i = 0; i < paletteSize; i++)
        {
            m_entries.push_back(type::RGBAColor(1.0f - i * step, 1.0f, 0.0f, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::GREEN_TO_YELLOW)
    {
        const float step = 1.0f / (paletteSize);
        for (unsigned int i = 0; i < paletteSize; i++)
        {
            m_entries.push_back(type::RGBAColor(i * step, 1.0f, 0.0f, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::GREEN_TO_CYAN)
    {
        const float step = 1.0f / (paletteSize);
        for (unsigned int i = 0; i < paletteSize; i++)
        {
            m_entries.push_back(type::RGBAColor(0.0f, 1.0f, i * step, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::CYAN_TO_GREEN)
    {
        const float step = 1.0f / (paletteSize);
        for (unsigned int i = 0; i < paletteSize; i++)
        {
            m_entries.push_back(type::RGBAColor(0.0f, 1.0f, 1.0f - i * step, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::CYAN_TO_BLUE)
    {
        const float step = 1.0f / (paletteSize);
        for (unsigned int i = 0; i < paletteSize; i++)
        {
            m_entries.push_back(type::RGBAColor(0.0f, 1.0f - i * step, 1.0f, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::BLUE_TO_CYAN)
    {
        const float step = 1.0f / (paletteSize);
        for (unsigned int i = 0; i < paletteSize; i++)
        {
            m_entries.push_back(type::RGBAColor(0.0f, i * step, 1.0f, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::RED)
    {
        const float step = 1.4f / (paletteSize);
        for (unsigned int i = 0; i < paletteSize / 2; i++)
        {
            m_entries.push_back(type::RGBAColor(0.3f + i * step, 0.0f, 0.0f, 1.0f));
        }
        for (unsigned int i = 0; i < (paletteSize - paletteSize / 2); i++)
        {
            m_entries.push_back(type::RGBAColor(1.0f, i * step, i * step, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::RED_INV)
    {
        const float step = 1.4f / (paletteSize);
        for (unsigned int i = 0; i < (paletteSize - paletteSize / 2); i++)
        {
            m_entries.push_back(type::RGBAColor(1.0f, 0.7f - i * step, 0.7f - i * step, 1.0f));
        }
        for (unsigned int i = 0; i < paletteSize / 2; i++)
        {
            m_entries.push_back(type::RGBAColor(1.0f - i * step, 0.0f, 0.0f, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::GREEN)
    {
        const float step = 1.4f / (paletteSize);
        for (unsigned int i = 0; i < paletteSize / 2; i++)
        {
            m_entries.push_back(type::RGBAColor(0.0f, 0.3f + i * step, 0.0f, 1.0f));
        }
        for (unsigned int i = 0; i < (paletteSize - paletteSize / 2); i++)
        {
            m_entries.push_back(type::RGBAColor(i * step, 1.0f, i * step, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::GREEN_INV)
    {
        const float step = 1.4f / (paletteSize);
        for (unsigned int i = 0; i < (paletteSize - paletteSize / 2); i++)
        {
            m_entries.push_back(type::RGBAColor(0.7f - i * step, 1.0f, 0.7f - i * step, 1.0f));
        }
        for (unsigned int i = 0; i < paletteSize / 2; i++)
        {
            m_entries.push_back(type::RGBAColor(0.0f, 1.0f - i * step, 0.0f, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::BLUE)
    {
        const float step = 1.4f / (paletteSize);
        for (unsigned int i = 0; i < paletteSize / 2; i++)
        {
            m_entries.push_back(type::RGBAColor(0.0f, 0.0f, 0.3f + i * step, 1.0f));
        }
        for (unsigned int i = 0; i < (paletteSize - paletteSize / 2); i++)
        {
            m_entries.push_back(type::RGBAColor(i * step, i * step, 1.0f, 1.0f));
        }
    }
    else if (colorScheme == ColorPreset::BLUE_INV)
    {
        const float step = 1.4f / (paletteSize);
        for (unsigned int i = 0; i < (paletteSize - paletteSize / 2); i++)
        {
            m_entries.push_back(type::RGBAColor(0.7f - i * step, 0.7f - i * step, 1.0f, 1.0f));
        }
        for (unsigned int i = 0; i < paletteSize / 2; i++)
        {
            m_entries.push_back(type::RGBAColor(0.0f, 0.0f, 1.0f - i * step, 1.0f));
        }
    } else if (colorScheme == ColorPreset::HSV)
    {
        // List the colors
        const float step = 1.0f/(paletteSize-1);
        for (unsigned int i=0; i<paletteSize; i++)
        {
            const auto h = i * step * 360;
            m_entries.emplace_back(type::RGBAColor::fromHSVA(h,1.f,1.f, 1.0f));
        }
    }
    else
    {
        return false;
    }
    return true;
}

std::ostream& operator<<(std::ostream& out, const ColorMap& m)
{
    out << sofa::helper::join(m.m_entries.begin(), m.m_entries.end(), [](const auto& color){ return color.toHexadecimal(); }, ' ');
    return out;
}

std::istream& operator>>(std::istream& in, ColorMap& m)
{
    in >> std::ws;

    std::string colorMap;
    {
        std::string line;
        while (in && std::getline(in, line))
        {
            colorMap += line + " ";
        }
        in.clear();
    }

    colorMap = sofa::helper::removeTrailingCharacter(colorMap, ' ');

    if (colorMap.empty())
        return in;

    const auto colorScheme = stringToColorScheme(colorMap);

    if (colorScheme.has_value())
    {
        m.buildFromColorScheme(256, colorScheme.value());
        return in;
    }

    std::istringstream lineStream(colorMap);
    m.m_entries.read(lineStream);
    return in;
}

} // namespace sofa::helper
