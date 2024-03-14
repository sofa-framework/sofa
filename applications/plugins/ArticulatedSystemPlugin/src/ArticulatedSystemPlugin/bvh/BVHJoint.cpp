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
#include <ArticulatedSystemPlugin/bvh/BVHJoint.h>

#include <sofa/helper/config.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/type/fixed_array.h>
#include <sofa/core/visual/VisualParams.h>

#include <sstream>
#include <iostream>

namespace sofa::helper::io::bvh
{

int BVHJoint::lastId = 0;

BVHJoint::BVHJoint(const char *_name, bool _endSite, BVHJoint *_parent)
    :parent(_parent),endSite(_endSite)
{
    offset = nullptr;
    channels = nullptr;
    motion = nullptr;
    parent = nullptr ;

    if(_parent)
        parent = _parent ;

    id = lastId++;

    if (!endSite)
        strcpy(name,_name);
    else
        strcpy(name,"End Site");
}

BVHJoint::~BVHJoint()
{
    if (offset) delete offset;
    if (channels) delete channels;
    if (motion) delete motion;

    lastId--;
}

void BVHJoint::initMotion(double fTime, unsigned int fCount)
{
    if (channels != nullptr)
        this->getMotion()->init(fTime, fCount, channels->size);

    for (unsigned int i=0; i < children.size(); i++)
        children[i]->initMotion(fTime, fCount);
}

void BVHJoint::display(int frameNum)
{
    const auto drawtool = core::visual::VisualParams::defaultInstance()->drawTool();

    drawtool->pushMatrix();
    drawtool->disableLighting();
    drawtool->drawLine({ 0.0f, 0.0f, 0.0f }, { float(offset->x), float(offset->y), float(offset->z) }, type::RGBAColor::black());

    core::visual::VisualParams::defaultInstance()->getModelViewMatrix(matrix);

    type::Quatf q;
    float rotmat[16];
    if (channels != nullptr)
    {
        for (unsigned int i=0; i<channels->size; i++)
        {
            switch (channels->channels[i])
            {
            case BVHChannels::Xposition:
                drawtool->translate(float(motion->frames[frameNum][i]), 0.0f, 0.0f);
                break;
            case BVHChannels::Yposition:
                drawtool->translate(0.0f, float(motion->frames[frameNum][i]), 0.0f);
                break;
            case BVHChannels::Zposition:
                drawtool->translate(0.0f, 0.0f, float(motion->frames[frameNum][i]));
                break;
            case BVHChannels::Xrotation:
                q = q.axisToQuat({ 1.0f,0.0f,0.0f }, float(motion->frames[frameNum][i] * M_PI / 180));
                q.writeOpenGlMatrix(rotmat);
                drawtool->multMatrix(rotmat);
                break;
            case BVHChannels::Yrotation:
                q = q.axisToQuat({ 0.0f,1.0f,0.0f }, float(motion->frames[frameNum][i] * M_PI / 180));
                q.writeOpenGlMatrix(rotmat);
                drawtool->multMatrix(rotmat);
                break;
            case BVHChannels::Zrotation:
                q = q.axisToQuat({ 0.0f,0.0f,1.0f }, float(motion->frames[frameNum][i] * M_PI / 180));
                q.writeOpenGlMatrix(rotmat);
                drawtool->multMatrix(rotmat);
                break;
            default:
                break;
            }
        }
    }

    drawtool->setMaterial( sofa::type::RGBAColor{ 1.0,0.0,0.0,1.0f });
    drawtool->drawSphere({ 0.0f, 0.0f, 0.0f }, 0.01f);

    for (unsigned int i=0; i<children.size(); i++)
    {
        children[i]->display(frameNum);
    }
}

void BVHJoint::displayInGlobalFrame(void)
{
    const auto drawtool = core::visual::VisualParams::defaultInstance()->drawTool();

    drawtool->pushMatrix();
    drawtool->translate(0.0f, 0.0f, -4.0f);
    float matf[16];
    for (auto i = 0 ; i < 16; i++)
        matf[i] = float(matrix[i]);
    drawtool->multMatrix(matf);

    drawtool->disableLighting();

    drawtool->setMaterial(sofa::type::RGBAColor{ 1.0,0.0,0.0,1.0f });
    drawtool->drawSphere({ 0.0f, 0.0f, 0.0f }, 0.005f);

    for (unsigned int i=0; i<children.size(); i++)
    {
        children[i]->displayInGlobalFrame();
    }
}


int BVHJoint::getNumJoints(char *s)
{
    int tmp(0);

    if (s!=nullptr)
    {
        if (strcmp(name,s) == 0)
            return accumulateNumJoints();

        for (unsigned int i=0; i<children.size(); i++)
            tmp += children[i]->getNumJoints(s);
    }
    else
        return accumulateNumJoints();

    return tmp;
}

int BVHJoint::accumulateNumJoints(void)
{
    int tmp = 1;

    if (children.size() != 0)
    {
        for (unsigned int i=0; i<children.size(); i++)
            tmp += children[i]->accumulateNumJoints();
    }

    return tmp;
}


int BVHJoint::getNumSegments(char *s)
{
    return (getNumJoints(s) - 1);
}


void BVHJoint::dump(char *fName, char *rootJointName)
{
    FILE *f = fopen(fName,"w+");

    fprintf(f,"Catheter_Name Walk\n\n");
    fprintf(f,"Number_of_Nodes %d\n", getNumJoints(rootJointName));
    fprintf(f,"Number_of_Segments %d\n", getNumSegments(rootJointName));

    fprintf(f, "\nList_of_Nodes\n");
    dumpPosition(f, rootJointName);

    fprintf(f, "\nSegments\n");
    dumpSegment(f, rootJointName);

    fprintf(f, "\nRotationAxis\n");
    dumpRotation(f, rootJointName);

    fprintf(f, "\nRotationLimits\n");
    dumpRotationLimit(f, rootJointName);

    fprintf(f, "\nRotationStiffness\n");
    dumpRotationStiffness(f, rootJointName);

    fprintf(f, "\nTranslationAxis\n");
    fprintf(f, "\t0  0 1.000000 0.000000 0.000000\n");
    fprintf(f, "\t1  0 0.000000 1.000000 0.000000\n");
    fprintf(f, "\t2  0 0.000000 0.000000 1.000000\n");

    fprintf(f, "\nTranslationStiffness\n");
    fprintf(f, "\t0 10000.0\n");
    fprintf(f, "\t1 10000.0\n");
    fprintf(f, "\t2 10000.0\n");

    fclose(f);
}


void BVHJoint::dumpPosition(FILE *f, char *s)
{
    if (s!=nullptr)
    {
        if (strcmp(name,s) == 0)
            dumpPosition(f, id);
        else
            for (unsigned int i=0; i<children.size(); i++)
                children[i]->dumpPosition(f, s);
    }
    else
        dumpPosition(f, id);
}


void BVHJoint::dumpPosition(FILE *f, int beginIndex)
{
    fprintf(f, "\t%d %f %f %f\n", id - beginIndex, matrix[12], matrix[13], matrix[14]);
    for (unsigned int i=0; i<children.size(); i++)
        children[i]->dumpPosition(f, beginIndex);
}


void BVHJoint::dumpSegment(FILE *f, char *s)
{
    int cpt = 0;

    if (s!=nullptr)
    {
        if (strcmp(name,s) == 0)
            dumpSegment(f, cpt, id);
        else
            for (unsigned int i=0; i<children.size(); i++)
                children[i]->dumpSegment(f, s);
    }
    else
        dumpSegment(f, cpt, id);
}


void BVHJoint::dumpSegment(FILE *f, int &cpt, int beginIndex)
{
    for (unsigned int i=0; i<children.size(); i++)
    {
        fprintf(f, "\t%d %d %d\n", cpt++, id - beginIndex, children[i]->id - beginIndex);
        children[i]->dumpSegment(f, cpt, beginIndex);
    }
}


void BVHJoint::dumpRotation(FILE *f, char *s)
{
    int cpt = 0;

    if (s!=nullptr)
    {
        if (strcmp(name,s) == 0)
            dumpRotation(f, cpt, id);
        else
            for (unsigned int i=0; i<children.size(); i++)
                children[i]->dumpRotation(f, s);
    }
    else
        dumpRotation(f, cpt, id);
}


void BVHJoint::dumpRotation(FILE *f, int &cpt, int beginIndex)
{
    fprintf(f, "\t%d %d %f %f %f\n", cpt++, id - beginIndex, matrix[0], matrix[1], matrix[2]);
    fprintf(f, "\t%d %d %f %f %f\n", cpt++, id - beginIndex, matrix[4], matrix[5], matrix[6]);
    fprintf(f, "\t%d %d %f %f %f\n", cpt++, id - beginIndex, matrix[8], matrix[9], matrix[10]);

    for (unsigned int i=0; i<children.size(); i++)
        children[i]->dumpRotation(f, cpt, beginIndex);
}


void BVHJoint::dumpRotationLimit(FILE *f, char *s)
{
    int cpt = 0;

    if (s!=nullptr)
    {
        if (strcmp(name,s) == 0)
            dumpRotationLimit(f, cpt);
        else
            for (unsigned int i=0; i<children.size(); i++)
                children[i]->dumpRotationLimit(f, s);
    }
    else
        dumpRotationLimit(f, cpt);
}


void BVHJoint::dumpRotationLimit(FILE *f, int &cpt)
{
    fprintf(f, "\t%d -1000000.0 1000000.0\n", cpt++);
    fprintf(f, "\t%d -1000000.0 1000000.0\n", cpt++);
    fprintf(f, "\t%d -1000000.0 1000000.0\n", cpt++);

    for (unsigned int i=0; i<children.size(); i++)
        children[i]->dumpRotationLimit(f, cpt);
}


void BVHJoint::dumpRotationStiffness(FILE *f, char *s)
{
    int cpt = 0;

    if (s!=nullptr)
    {
        if (strcmp(name,s) == 0)
            dumpRotationStiffness(f, cpt);
        else
            for (unsigned int i=0; i<children.size(); i++)
                children[i]->dumpRotationStiffness(f, s);
    }
    else
        dumpRotationStiffness(f, cpt);
}


void BVHJoint::dumpRotationStiffness(FILE *f, int &cpt)
{
    fprintf(f, "\t%d 1000000000.0\n", cpt++);
    fprintf(f, "\t%d 1000000000.0\n", cpt++);
    fprintf(f, "\t%d 1000000000.0\n", cpt++);

    for (unsigned int i=0; i<children.size(); i++)
        children[i]->dumpRotationStiffness(f, cpt);
}

void BVHJoint::debug(int tab)
{
    std::stringstream tmpmsg ;
    for (int i=0; i<tab; i++)
        tmpmsg << "\t";

    tmpmsg << name << msgendl ;

    if (offset != nullptr)
    {
        for (int i=0; i<tab; i++)
            tmpmsg << "\t";
        tmpmsg << "offset " << offset->x << " " << offset->y << " " << offset->z << msgendl;
    }

    if (channels != nullptr)
    {
        for (int i=0; i<tab; i++)
            tmpmsg << "\t";
        tmpmsg << "channels ";
        for (unsigned int i=0; i<channels->channels.size(); i++)
            tmpmsg << channels->channels[i] << " ";
        msg_info("BVHJoint") << tmpmsg.str() ;
    }

    for (unsigned int i=0; i<children.size(); i++)
    {
        children[i]->debug(tab+1);
    }
}

int BVHJoint::getId()
{
    return id;
}

char* BVHJoint::getName()
{
    return name;
}

BVHOffset* BVHJoint::getOffset()
{
    return offset;
}

} // namespace sofa::helper::io::bvh
