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
#include <SofaImGui/ImGuiDataWidget.h>
#include <imgui.h>
#include <GL/glew.h>
#include <SofaMatrix/BaseMatrixImageProxy.h>

#include <optional>

namespace sofamatrix::imgui
{

std::optional<std::vector<unsigned char>> imageData(sofa::linearalgebra::BaseMatrix* ptr)
{
    static std::map<sofa::linearalgebra::BaseMatrix*, std::vector<unsigned char>> map;
    auto mapIt = map.find(ptr);
    if (mapIt == map.end())
    {
        const auto insertResult = map.emplace(ptr, std::vector<unsigned char>());
        if (insertResult.second)
        {
            mapIt = insertResult.first;
        }
        else
        {
            return {};
        }
    }
    return mapIt->second;
}

std::optional<GLuint> textureID(sofa::linearalgebra::BaseMatrix* ptr)
{
    static std::map<sofa::linearalgebra::BaseMatrix*, GLuint> map;
    auto mapIt = map.find(ptr);
    if (mapIt == map.end())
    {
        const auto insertResult = map.emplace(ptr, GLuint());
        if (insertResult.second)
        {
            mapIt = insertResult.first;

            // generate a new texture ID
            GLuint& textureID = mapIt->second;
            glGenTextures(1, &textureID);

            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        }
        else
        {
            return {};
        }
    }
    return mapIt->second;
}

void updateTexture(sofa::linearalgebra::BaseMatrix* matrix, unsigned& textureIDRef,
                   std::vector<unsigned char>& imageDataRef)
{
    const auto width = matrix->rows();
    const auto height = matrix->cols();

    imageDataRef.resize(width * height, 255);

    // write the data
    for (sofa::SignedIndex y = 0; y < height; ++y)
    {
        for (sofa::SignedIndex x = 0; x < width; ++x)
        {
            imageDataRef[y * width + x] = !static_cast<bool>(matrix->element(x, y)) * std::numeric_limits<unsigned char>::max();
        }
    }

    // Update the texture
    glBindTexture(GL_TEXTURE_2D, textureIDRef);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,
                 imageDataRef.data());
}

void drawTexture(sofa::linearalgebra::BaseMatrix* matrix, bool updateTexture)
{
    if (matrix == nullptr)
        return;

    auto textureID = sofamatrix::imgui::textureID(matrix);
    if (textureID.has_value() == false)
        return;

    auto imageData = sofamatrix::imgui::imageData(matrix);
    if (imageData.has_value() == false)
        return;

    auto& textureIDRef = textureID.value();
    auto& imageDataRef = imageData.value();

    if (updateTexture)
    {
        sofamatrix::imgui::updateTexture(matrix, textureIDRef, imageDataRef);
    }

    const auto width = matrix->rows();
    const auto height = matrix->cols();

    ImVec2 imageSize(width, height);

    // Get the available width
    imageSize.x = ImGui::GetContentRegionAvail().x;

    // Calculate the aspect ratio
    float aspectRatio = (float)width / (float)height;

    // Calculate the height based on the available width and aspect ratio
    imageSize.y = imageSize.x / aspectRatio;

    // Draw the texture using imgui
    ImGui::Image((ImTextureID)(intptr_t)textureIDRef, imageSize);
}

static constexpr int unknownCounter = -1;
int* counter(sofa::core::objectmodel::BaseData& data)
{
    static std::map<sofa::core::objectmodel::BaseData*, std::unique_ptr<int>> map;
    auto mapIt = map.find(&data);
    if (mapIt == map.end())
    {
        const auto insertResult = map.emplace(&data, std::make_unique<int>(unknownCounter));
        if (insertResult.second)
        {
            mapIt = insertResult.first;
        }
        else
        {
            return nullptr;
        }
    }
    return mapIt->second.get();
}

}

namespace sofaimgui
{

template<>
void DataWidget<sofa::type::BaseMatrixImageProxy>::showWidget(MyData& data)
{
    const auto& matrixProxy = data.getValue();

    if (auto* matrix = matrixProxy.getMatrix())
    {
        std::stringstream ss;
        ss << std::to_string(matrix->rows()) << "x" << std::to_string(matrix->cols());
        ImGui::Text(ss.str().c_str());

        const int counter = data.getCounter();
        int* lastCounter = sofamatrix::imgui::counter(data);

        const bool updateTexture = lastCounter == nullptr || *lastCounter != counter;
        sofamatrix::imgui::drawTexture(matrix, updateTexture);
        if (lastCounter)
        {
            *lastCounter = counter;
        }
    }
    else
    {
        ImGui::Text("Invalid matrix");
    }
}
}

const bool dw_matrixproxy = sofaimgui::DataWidgetFactory::Add<sofa::type::BaseMatrixImageProxy>();
