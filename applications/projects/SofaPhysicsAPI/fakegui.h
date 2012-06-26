#ifndef FAKEGUI_H
#define FAKEGUI_H

#include <sofa/gui/SofaGUI.h>

/// this fake GUI is only meant to manage "sendMessage" from python scripts
class FakeGUI : public sofa::gui::SofaGUI
{
protected:
    /// The destructor should not be called directly. Use the closeGUI() method instead.
    ~FakeGUI() {}

public:
    /// @name methods each GUI must implement
    /// @{
    virtual int mainLoop() {return 0;}
    virtual void redraw() {}
    virtual int closeGUI() {return 0;}
    virtual void setScene(sofa::simulation::Node::SPtr /*groot*/, const char* /*filename*/=NULL, bool /*temporaryFile*/=false) {}
    virtual sofa::simulation::Node* currentSimulation() {return 0;}
    /// @}

    /// @name methods to communicate with the GUI
    /// @{
    virtual void sendMessage(const std::string & /*msgType*/,const std::string & /*msgValue*/);
    /// @}

    static void Create();

};





#endif // FAKEGUI_H
