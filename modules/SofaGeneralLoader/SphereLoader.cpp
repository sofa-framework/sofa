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
#include <SofaGeneralLoader/SphereLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/Locale.h>
#include <sofa/helper/Quater.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/core/ObjectFactory.h>
#include <sstream>

using namespace sofa::core::loader;
using namespace sofa::defaulttype;
namespace sofa
{
namespace component
{
namespace loader
{

int SphereLoaderClass = core::RegisterObject("Loader for sphere model description files")
        .add<SphereLoader>();

SphereLoader::SphereLoader()
    :BaseLoader(),
     d_positions(initData(&d_positions,"position","Sphere centers")),
     d_radius(initData(&d_radius,"listRadius","Radius of each sphere")),
     d_scale(initData(&d_scale, Vec3(1.0, 1.0, 1.0), "scale","Scale applied to sphere positions & radius")),
     d_rotation(initData(&d_rotation, Vec3(), "rotation", "Rotation of the DOFs")),
     d_translation(initData(&d_translation, Vec3(),"translation","Translation applied to sphere positions"))
{
    addAlias(&d_positions,"sphere_centers");
    addAlias(&d_scale, "scale3d");

    addUpdateCallback("filename", { &m_filename }, [this](const core::DataTracker& )
    {
        if (load()) {
            clearLoggedMessages();
            return sofa::core::objectmodel::ComponentState::Valid;
        }
        return sofa::core::objectmodel::ComponentState::Invalid;
    }, { &d_positions, &d_radius });

    addUpdateCallback("updateTransformPosition", { &d_translation, &d_rotation, &d_scale}, [this](const core::DataTracker&)
    {
        applyTransform();
        return sofa::core::objectmodel::ComponentState::Valid;
    }, { &d_positions });

    d_positions.setReadOnly(true);
    d_radius.setReadOnly(true);
}


void SphereLoader::applyTransform()
{
    const Vec3& scale = d_scale.getValue();
    const Vec3& rotation = d_rotation.getValue();
    const Vec3& translation = d_translation.getValue();
    m_internalEngine["updateTransformPosition"].cleanDirty();


    if (d_scale != Vec3(1.0, 1.0, 1.0) || d_rotation != Vec3(0.0, 0.0, 0.0) || d_translation != Vec3(0.0, 0.0, 0.0))
    {
        Matrix4 transformation = Matrix4::transformTranslation(translation) *
            Matrix4::transformRotation(helper::Quater< SReal >::createQuaterFromEuler(rotation * M_PI / 180.0)) *
            Matrix4::transformScale(scale);

        sofa::helper::WriteAccessor <Data< helper::vector<sofa::defaulttype::Vec<3, SReal> > > > my_positions = d_positions;
        for (size_t i = 0; i < my_positions.size(); i++)
        {
            my_positions[i] = transformation.transform(my_positions[i]);
        }
    }
}


bool SphereLoader::load()
{
    m_internalEngine["filename"].cleanDirty();
    auto my_radius = getWriteOnlyAccessor(d_radius);
    auto my_positions = getWriteOnlyAccessor(d_positions);
    my_radius.clear();
    my_positions.clear();

    // Make sure that fscanf() uses a dot '.' as the decimal separator.
    helper::system::TemporaryLocale locale(LC_NUMERIC, "C");

    const char* filename = m_filename.getFullPath().c_str();
    std::string fname = std::string(filename);

    if (!sofa::helper::system::DataRepository.findFile(fname)) return false;

    char cmd[64];
    FILE* file;

    static const char* SPH_FORMAT = "sph 1.0";

    if ((file = fopen(fname.c_str(), "r")) == nullptr)
    {
        msg_error("SphereLoader") << "cannot read file '" << filename << "'. ";
        return false;
    }

    int totalNumSpheres=0;

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
            my_positions.push_back(Vector3((SReal)cx,(SReal)cy,(SReal)cz));
            my_radius.push_back((SReal)r);
            ++totalNumSpheres;
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

    (void) fclose(file);

    if (d_scale.isSet())
    {
        const SReal sx = d_scale.getValue()[0];
        const SReal sy = d_scale.getValue()[1];
        const SReal sz = d_scale.getValue()[2];

        for (unsigned int i = 0; i < my_radius.size(); i++)
        {
            my_radius[i] *= sx;
        }

        for (unsigned int i = 0; i < my_positions.size(); i++)
        {
            my_positions[i].x() *= sx;
            my_positions[i].y() *= sy;
            my_positions[i].z() *= sz;
        }
    }

    if (d_translation.isSet())
    {
        const SReal dx = d_translation.getValue()[0];
        const SReal dy = d_translation.getValue()[1];
        const SReal dz = d_translation.getValue()[2];

        for (unsigned int i = 0; i < my_positions.size(); i++)
        {
            my_positions[i].x() += dx;
            my_positions[i].y() += dy;
            my_positions[i].z() += dz;
        }
    }

    applyTransform();

    return true;
}

}//loader

}//component

}//sofa
