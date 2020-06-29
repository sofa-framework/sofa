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
#ifndef SOFA_ASSIMPLOADER_H
#define SOFA_ASSIMPLOADER_H

#include <sofa/core/loader/MeshLoader.h>
#include <SofaAssimp/config.h>

struct aiScene;

namespace sofa
{

namespace component
{

namespace loader
{

/**
 * AssimpLoader class interfaces Assimp library reader with SOFA loader components.
 * For more information about the class API see doc: http://assimp.sourceforge.net/lib_html/usage.html
 *
 *  Created on: February 28th 2018
 *      Author: epernod 
 */
class SOFA_ASSIMP_API AssimpLoader : public sofa::core::loader::MeshLoader
{
public:
    SOFA_CLASS(AssimpLoader, sofa::core::loader::MeshLoader);
protected:
    /// Default constructor of the component
    AssimpLoader();
    virtual ~AssimpLoader();

public:
    /// Main Load method inherites from \sa sofa::core::loader::MeshLoader::load()
    virtual bool doLoad() override;
    virtual void doClearBuffers() override {}


    int getpPreprocessing() {
        return pp_Loadsteps;
    }
    void  setpPreprocessing(int p) {
        pp_Loadsteps = p;
    }

    aiScene* getScene() { return m_assimpScene; }
  
protected:
    /// Main internal method, implement the loading of OpenCTM mesh file.
    bool convertAssimpScene();
    
private:
    aiScene* m_assimpScene;
    unsigned int pp_Loadsteps;
};


} // namespace loader

} // namespace component

} // namespace sofa

#endif //SOFA_ASSIMPLOADER_H
