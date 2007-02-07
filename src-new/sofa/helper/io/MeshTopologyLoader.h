#ifndef SOFA_HELPER_IO_MESHTOPOLOGYLOADER_H
#define SOFA_HELPER_IO_MESHTOPOLOGYLOADER_H

namespace sofa
{

namespace helper
{

namespace io
{

class MeshTopologyLoader
{
public:
    virtual ~MeshTopologyLoader() {}
    bool load(const char *filename);
    virtual void setNbPoints(int /*n*/) {}
    virtual void setNbLines(int /*n*/) {}
    virtual void setNbEdges(int /*n*/) {}
    virtual void setNbTriangles(int /*n*/) {}
    virtual void setNbQuads(int /*n*/) {}
    virtual void setNbTetras(int /*n*/) {}
    virtual void setNbCubes(int /*n*/) {}
    virtual void addPoint(double /*px*/, double /*py*/, double /*pz*/) {}
    virtual void addLine(int /*p1*/, int /*p2*/) {}
    virtual void addTriangle(int /*p1*/, int /*p2*/, int /*p3*/) {}
    virtual void addQuad(int /*p1*/, int /*p2*/, int /*p3*/, int /*p4*/) {}
    virtual void addTetra(int /*p1*/, int /*p2*/, int /*p3*/, int /*p4*/) {}
    virtual void addCube(int /*p1*/, int /*p2*/, int /*p3*/, int /*p4*/, int /*p5*/, int /*p6*/, int /*p7*/, int /*p8*/) {}
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
