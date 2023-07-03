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
#include <sofa/component/io/mesh/SphereLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/Locale.h>
#include <sofa/type/Quat.h>
#include <sofa/type/Mat.h>
#include <sofa/core/ObjectFactory.h>

using namespace sofa::core::loader;
using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace sofa::type;

namespace sofa::component::io::mesh
{

int SphereLoaderClass = core::RegisterObject("Loader for sphere model description files")
        .add<SphereLoader>();

SphereLoader::SphereLoader()
    :BaseLoader(),
     d_positions(initData(&d_positions,"position","Sphere centers")),
     d_radius(initData(&d_radius,"listRadius","Radius of each sphere")),
     d_scale(initData(&d_scale, type::Vec3(1.0, 1.0, 1.0), "scale","Scale applied to sphere positions & radius")),
     d_rotation(initData(&d_rotation, type::Vec3(), "rotation", "Rotation of the DOFs")),
     d_translation(initData(&d_translation, type::Vec3(),"translation","Translation applied to sphere positions"))
{
    addAlias(&d_positions,"sphere_centers");
    addAlias(&d_scale, "scale3d");

    addUpdateCallback("updateFileNameAndTransformPosition", { &d_filename, &d_translation, &d_rotation, &d_scale}, [this](const core::DataTracker& tracker)
    {
        if(tracker.hasChanged(d_filename))
        {
            if (load()) {
                clearLoggedMessages();
                applyTransform();
                return sofa::core::objectmodel::ComponentState::Valid;
            }
        }
        else
        {
            applyTransform();
            return sofa::core::objectmodel::ComponentState::Valid;
        }

        return sofa::core::objectmodel::ComponentState::Invalid;
    }, { &d_positions, &d_radius });

    d_positions.setReadOnly(true);
    d_radius.setReadOnly(true);
}


void SphereLoader::applyTransform()
{
    const auto& scale = d_scale.getValue();
    const auto& rotation = d_rotation.getValue();
    const auto& translation = d_translation.getValue();

    if (scale != type::Vec3(1.0, 1.0, 1.0) || rotation != type::Vec3(0.0, 0.0, 0.0) || translation != type::Vec3(0.0, 0.0, 0.0))
    {
        if(scale != type::Vec3(1.0, 1.0, 1.0)) {
            if(scale[0] == 0.0 || scale[1] == 0.0 || scale[2] == 0.0) {
                msg_warning() << "Data scale should not be set to zero";
            }
        }
        const Matrix4 transformation = Matrix4::transformTranslation(translation) *
            Matrix4::transformRotation(type::Quat< SReal >::createQuaterFromEuler(rotation * M_PI / 180.0)) *
            Matrix4::transformScale(scale);

        auto my_positions = getWriteOnlyAccessor(d_positions);

        if(my_positions.size() != m_savedPositions.size()) {
            msg_error() << "Position size mismatch";
        }

        for (size_t i = 0; i < my_positions.size(); i++) {
            my_positions[i] = transformation.transform(m_savedPositions[i]);
        }
    }
}


bool SphereLoader::load()
{
    auto my_radius = getWriteOnlyAccessor(d_radius);
    auto my_positions = getWriteOnlyAccessor(d_positions);
    my_radius.clear();
    my_positions.clear();

    // Make sure that fscanf() uses a dot '.' as the decimal separator.
    helper::system::TemporaryLocale locale(LC_NUMERIC, "C");

    const char* filename = d_filename.getFullPath().c_str();
    std::string fname = std::string(filename);

    if (!sofa::helper::system::DataRepository.findFile(fname))
        return false;

    char cmd[64];
    FILE* file;

    static const char* SPH_FORMAT = "sph 1.0";

    if ((file = fopen(fname.c_str(), "r")) == nullptr)
    {
        msg_error("SphereLoader") << "cannot read file '" << filename << "'. ";
        return false;
    }

    // Check first line
    if (fgets(cmd, 7, file) == nullptr || !strcmp(cmd,SPH_FORMAT))
    {
        fclose(file);
        return false;
    }
    skipToEOL(file);

    std::ostringstream cmdScanFormat;
    cmdScanFormat << "%" << (sizeof(cmd) - 1) << "s";
    while (fscanf(file, cmdScanFormat.str().c_str(), cmd ) != EOF)
    {
        if (!strcmp(cmd,"nums"))
        {
            int total;
            if (fscanf(file, "%d", &total) == EOF)
                msg_error("SphereLoader") << "Problem while loading. fscanf function has encountered an error." ;
            my_positions.reserve(total);
        }
        else if (!strcmp(cmd,"sphe"))
        {
            int index;
            double cx=0,cy=0,cz=0,r=1;
            if (fscanf(file, "%d %lf %lf %lf %lf\n",
                    &index, &cx, &cy, &cz, &r) == EOF)
                msg_error("SphereLoader") << "Problem while loading. fscanf function has encountered an error." ;
            my_positions.push_back(Vec3((SReal)cx,(SReal)cy,(SReal)cz));
            my_radius.push_back((SReal)r);
        }
        else if (cmd[0]=='#')
        {
            skipToEOL(file);
        }
        else			// it's an unknown keyword
        {
            msg_warning("SphereLoader") << "Unknown Sphere keyword: "<< cmd << " in file '"<<filename<< "'" ;
            skipToEOL(file);
        }
    }

    m_savedPositions.clear();
    m_savedPositions.resize(my_positions.size());
    for (size_t i = 0; i < my_positions.size(); i++)
    {
        m_savedPositions[i] = my_positions[i];
    }


    (void) fclose(file);

    return true;
}

} // namespace sofa::component::io::mesh
