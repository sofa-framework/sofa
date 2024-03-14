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
#ifndef SOFA_HELPER_IO_XSPLOADER_H
#define SOFA_HELPER_IO_XSPLOADER_H

#include <cstddef>                  /// For size_t
#include <string>                   /// For std::string
#include <sofa/helper/config.h>     /// For SOFA_HELPER_API


namespace sofa::helper::io
{

/// @brief Inherit this class to load data from a Xsp file.
///
/// To connect client-code data structure with the XspLoader you need to
/// Inherit from this class and override the virtual methods to you fill your
/// structures from the XspLoader events.
///
/// Each overridable method is connected to the reading of a given "token"
/// in the Xsp file format.
///
/// @see XspLoader for an example of use.
class SOFA_HELPER_API XspLoaderDataHook
{
public:
    /// @brief Destructor, does nothing special
    virtual ~XspLoaderDataHook();

    /// @brief Called by the XspLoader when the loading is done.
    ///
    /// This method is called by the XspLoader when the loading is done.
    /// Overriding this method allows client-code to implement post-loading checking.
    /// @param isOk is set to false this means that the loading code detected a
    /// problem and that the loaded informations are invalid and should be removed from
    /// the container.
    virtual void finalizeLoading(bool isOk) { SOFA_UNUSED(isOk); }

    /// @brief Called by the XspLoader to specify before loading the number of masses.
    /// @param n number of massses.
    virtual void setNumMasses(size_t /*n*/) {}

    /// @brief Called by the XspLoader to specify before loading the number of springs.
    /// @param n number of springs.
    virtual void setNumSprings(size_t /*n*/) {}

    /// @brief Called by the XspLoader to specify the directional gravity.
    /// @param gx, gy, gz the three component of the gravity.
    virtual void setGravity(SReal /*gx*/, SReal /*gy*/, SReal /*gz*/) {}

    /// @brief Called by the XspLoader to specify the viscosity
    /// @param gx, gy, gz the three component of the gravity.
    virtual void setViscosity(SReal /*visc*/) {}

    /// @brief Add a new mass.
    /// @param px,py,pz 3D position.
    /// @param vx,vz,vz 3D velocity.
    /// @param mass.
    /// @param elastic property.
    /// @param fixed boolean indicates that the mass is "static".
    /// @param surface indicates that the mass is on the surface.
    virtual void addMass(SReal /*px*/, SReal /*py*/, SReal /*pz*/, SReal /*vx*/, SReal /*vy*/, SReal /*vz*/, SReal /*mass*/, SReal /*elastic*/, bool /*fixed*/, bool /*surface*/) {}

    /// @brief Add a new spring.
    virtual void addSpring(size_t /*m1*/, size_t /*m2*/, SReal /*ks*/, SReal /*kd*/, SReal /*initpos*/) {}

    /// @brief Add an extended spring.
    virtual void addVectorSpring(size_t m1, size_t m2, SReal ks, SReal kd, SReal initpos, SReal /*restx*/, SReal /*resty*/, SReal /*restz*/) { addSpring(m1, m2, ks, kd, initpos); }
};

class SOFA_HELPER_API XspLoader
{
public:
    /// @brief Call this method to load an XspFile.
    /// @param filename the name of the file in the RessourceRepository to read data from.
    /// @param data pass a object of this type (or inherit one) to load the file in caller's data
    ///        structures
    /// @return wheter the loading succeded.
    /// @example
    /// class MyXspLoader : public XspLoaderDataHook
    /// {
    /// std::vector<double> mx;
    /// public:
    ///     void addMass(SReal px, SReal py, SReal pz, SReal, SReal, SReal, SReal, SReal, bool, bool) override
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
    /// MyXspLoader loadedData;
    /// XspLoader::Load("myfile.xs3", loadedData);
    static bool Load(const std::string& filename,
                     XspLoaderDataHook& data);

private:
    static bool ReadXspContent(std::ifstream &file,
                               bool hasVectorSpring,
                               XspLoaderDataHook& data);

};

} // namespace sofa::helper::io


#endif
