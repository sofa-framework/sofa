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
#ifndef SOFA_HELPER_IO_SPHERELOADER_H
#define SOFA_HELPER_IO_SPHERELOADER_H

#include <sofa/helper/config.h>

#include <sofa/type/Vec.h>


namespace sofa::helper::io
{

/// @brief Inherit this class to load data from sphere description.
///
/// To connect client-code data structure with the XspLoader you need to
/// Inherit from this class and override the virtual methods to you fill your
/// structures from the SphereLoader events.
/// @see SphereLoader for an example of use.
class SOFA_HELPER_API SphereLoaderDataHook
{
public:
    virtual ~SphereLoaderDataHook(){}

    /// @brief Called by the XspLoader when the loading is done.
    ///
    /// This method is called by the XspLoader when the loading is done.
    /// Overriding this method allows client-code to implement post-loading checking.
    /// @param isOk is set to false this means that the loading code detected a
    /// problem and that the loaded informations are invalid and should be removed from
    /// the container.
    virtual void finalizeLoading(const bool isOk){ SOFA_UNUSED(isOk); }

    /// @brief Called by the XspLoader to specify before loading the number of spheres.
    /// @param n number of sphere.
    virtual void setNumSpheres(const int n) { SOFA_UNUSED(n); }

    /// @brief Called by the Loader to specify the number of Spheres before actual loading.
    /// @param px, py, pz 3D position of the center.
    /// @param r the radius of the sphere.
    virtual void addSphere(const SReal /*px*/, const SReal /*py*/, const SReal /*pz*/, const SReal /*r*/) {}
};

class SOFA_HELPER_API SphereLoader
{
public:
    /// @brief Call this method to load a Sphere files.
    /// @param filename the name of the file in the RessourceRepository to read data from.
    /// @param data pass a object of this type (or inherit one) to load the file in caller's data
    ///        structures
    /// @return wheter the loading succeded.
    /// @example
    /// class MySphereData : public SphereLoaderDataHook
    /// {
    ///    std::vector<double> mx;
    /// public:
    ///     void addSpere(SReal px, SReal py, SReal pz, SReal r) override
    ///     {
    ///         mx.push_back(px);
    ///     }
    ///     void finalizeLoading(bool isOk) override
    ///     {
    ///         if(!isOk)
    ///             mx.clear();
    ///     }
    /// };
    ///
    /// MySphereData loadedData;
    /// SphereLoader::Load("myfile.sphere", loadedData);
    static bool Load(const std::string& filename, SphereLoaderDataHook& data);
};

} // namespace sofa::helper::io


#endif
