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
#include <sofa/http/VersionChecker.h>
#include <httplib.h>
#include <json.h>
#include <sofa/helper/logging/Messaging.h>


namespace sofa::http
{

std::optional<std::string> getLatestSOFARelease()
{
    constexpr std::string_view githubURL = "https://api.github.com";
    constexpr std::string_view repoPath = "/repos/sofa-framework/sofa/releases/latest";

    httplib::Client cli{std::string(githubURL)};

    auto res = cli.Get(std::string(repoPath));
    if (res && res->status == 200)
    {
        sofa::helper::json j = sofa::helper::json::parse(res->body);
        return j["tag_name"];
    }

    return {};
}

void checkLatestSOFARelease()
{
    const auto removeVPrefix = [](std::string version)
    {
        if (!version.empty() && version[0] == 'v')
        {
            version = version.substr(1);
        }
        return version;
    };

    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto latestVersion = getLatestSOFARelease();
    const auto t2 = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::milli> ms = t2 - t1;

    if (latestVersion.value_or("").empty())
    {
        msg_warning("VersionChecker") << "Cannot get latest release version {" << ms.count() << " ms}";
        return;
    }

    const std::string latestVersionNumber = removeVPrefix(latestVersion.value());
    const std::string currentVersionNumber = removeVPrefix(std::string(MODULE_VERSION));

    // Compare versions and print the result
    if (latestVersionNumber > currentVersionNumber)
    {
        msg_info("VersionChecker") << "A newer version of SOFA (" << latestVersionNumber
                << ") is available! Current version is " << currentVersionNumber
                << ". {" << ms.count() << " ms}";
    }
    else
    {
        msg_info("VersionChecker") << "You are using the latest version ("
            << currentVersionNumber << ")" << ". {" << ms.count() << " ms}";
    }
}


}
