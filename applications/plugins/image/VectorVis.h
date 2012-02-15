#ifndef IMAGE_VECTORVIS_H
#define IMAGE_VECTORVIS_H

namespace sofa
{

namespace defaulttype
{

/**
* Holds data regarding visualization of vector data so that it can be accessed and manipulated by the VectorVisualizationDataWidget
*/
class VectorVis
{
protected:
    int subsampleXY;
    int subsampleZ;
    int shapeScale;
    bool rgb;
    bool shape;

public:

    static const char* Name() { return "Vectors";}

    VectorVis(int _subsampleXY=5, int _subsampleZ=5, int _shapeScale=10, bool _rgb=false, bool _shape=false)
        :subsampleXY(_subsampleXY), subsampleZ(_subsampleZ), shapeScale(_shapeScale), rgb(_rgb), shape(_shape)
    { }

    int getSubsampleXY() const { return subsampleXY; }
    int getSubsampleZ() const { return subsampleZ; }
    int getShapeScale() const {return shapeScale; }
    bool getRgb() const {return rgb; }
    bool getShape() const {return shape;}

    void setSubsampleXY(int _subsampleXY) { subsampleXY = _subsampleXY; }
    void setSubsampleZ(int _subsampleZ) {subsampleZ = _subsampleZ; }
    void setShapeScale(int scale) { shapeScale = scale; }
    void setRgb(bool _rgb) {rgb = _rgb;}
    void setShape(bool vis) { shape = vis; }

    /**
    * Stream operator that allows data to be recieved from the GUI
    */
    inline friend std::istream& operator >> (std::istream& in, VectorVis& v)
    {
        int subsampleXY;
        int subsampleZ;
        int shapeScale;
        bool rgb;
        bool shape;
        in >> subsampleXY >> subsampleZ >> shapeScale >> rgb >> shape;

        v.setSubsampleXY(subsampleXY);
        v.setSubsampleZ(subsampleZ);
        v.setShapeScale(shapeScale);
        v.setRgb(rgb);
        v.setShape(shape);

        return in;
    }

    /**
    * Stream operator that allows data to be sent to the GUI
    */
    friend std::ostream& operator << (std::ostream& out, const VectorVis& v)
    {
        out << v.getSubsampleXY() << v.getSubsampleZ() << v.getShapeScale() << v.getRgb() << v.getShape() ;
        return out;
    }


};

}
}

#endif //IMAGE_VECTORVIS_H
