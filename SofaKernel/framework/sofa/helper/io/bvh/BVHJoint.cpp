/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/io/bvh/BVHJoint.h>

#include <sofa/helper/helper.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/gl/BasicShapes.h>

#include <iostream>

namespace sofa
{

namespace helper
{

namespace io
{

namespace bvh
{

int BVHJoint::lastId = 0;

BVHJoint::BVHJoint(const char *_name, bool _endSite, BVHJoint *_parent)
    :parent(_parent),endSite(_endSite)
{
    offset = NULL;
    channels = NULL;
    motion = NULL;
    parent = NULL ;

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
    if (channels != NULL)
        this->getMotion()->init(fTime, fCount, channels->size);

    for (unsigned int i=0; i < children.size(); i++)
        children[i]->initMotion(fTime, fCount);
}

void BVHJoint::display(int frameNum)
{
#ifndef SOFA_NO_OPENGL
	glPushMatrix();
    glDisable(GL_LIGHTING);
    glColor3f(0.0, 0.0, 0.0);
    glBegin(GL_LINES);
    glVertex3d(0.0, 0.0, 0.0);
    glVertex3d(offset->x, offset->y, offset->z);
    glEnd();
    glTranslatef((float)offset->x, (float)offset->y, (float)offset->z);

    glGetDoublev(GL_MODELVIEW_MATRIX,matrix);

    if (channels != NULL)
    {
        for (unsigned int i=0; i<channels->size; i++)
        {
            switch (channels->channels[i])
            {
            case BVHChannels::Xposition:
                glTranslatef((float)motion->frames[frameNum][i],0,0);
                break;
            case BVHChannels::Yposition:
                glTranslatef(0,(float)motion->frames[frameNum][i],0);
                break;
            case BVHChannels::Zposition:
                glTranslatef(0,0,(float)motion->frames[frameNum][i]);
                break;
            case BVHChannels::Xrotation:
                glRotatef((float)motion->frames[frameNum][i],1,0,0);
                break;
            case BVHChannels::Yrotation:
                glRotatef((float)motion->frames[frameNum][i],0,1,0);
                break;
            case BVHChannels::Zrotation:
                glRotatef((float)motion->frames[frameNum][i],0,0,1);
                break;
            default:
                break;
            }
        }
    }

    glColor3f(1.0,0.0,0.0);

	sofa::helper::fixed_array<float, 3> center(0.0, 0.0, 0.0);
    helper::gl::drawSphere(center, 0.01f);

    for (unsigned int i=0; i<children.size(); i++)
    {
        children[i]->display(frameNum);
    }

    glPopMatrix();
#endif /* SOFA_NO_OPENGL */
}

void BVHJoint::displayInGlobalFrame(void)
{
#ifndef SOFA_NO_OPENGL
    glPushMatrix();
    glLoadIdentity();
    glTranslatef(0.0,0.0,-4.0);
    glMultMatrixd(matrix);
    glDisable(GL_LIGHTING);
    glColor3f(1.0, 0.0, 0.0);

	sofa::helper::fixed_array<float, 3> center(0.0, 0.0, 0.0);
    helper::gl::drawSphere(center, 0.005f);

    glPopMatrix();

    for (unsigned int i=0; i<children.size(); i++)
    {
        children[i]->displayInGlobalFrame();
    }
#endif /* SOFA_NO_OPENGL */
}


int BVHJoint::getNumJoints(char *s)
{
    int tmp(0);

    if (s!=NULL)
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
    if (s!=NULL)
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

    if (s!=NULL)
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

    if (s!=NULL)
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

    if (s!=NULL)
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

    if (s!=NULL)
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
    for (int i=0; i<tab; i++)
        std::cout << "\t";

    std::cout << name << std::endl;

    if (offset != NULL)
    {
        for (int i=0; i<tab; i++)
            std::cout << "\t";
        std::cout << "offset " << offset->x << " " << offset->y << " " << offset->z << std::endl;
    }

    if (channels != NULL)
    {
        for (int i=0; i<tab; i++)
            std::cout << "\t";
        std::cout << "channels ";
        for (unsigned int i=0; i<channels->channels.size(); i++)
            std::cout << channels->channels[i] << " ";
        std::cout << std::endl;
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

} // namespace bvh

} // namespace io

} // namespace helper

} // namespace sofa
