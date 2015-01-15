.pragma library

function InstanciateComponent(url) {
    console.log("a");
    var component = Qt.createComponent(url);
    console.log("b");
    var incubator = component.incubateObject();
    console.log("c");
    incubator.forceCompletion();
    console.log("d");
    return incubator.object;
}

////////////////////    TOOLS    ////////////////////

var Tools = new InstanciateComponent("qrc:/SofaBasics/Tools.qml");

//////////////////// UI SETTINGS ////////////////////

var Ui = new InstanciateComponent("qrc:/SofaBasics/UISettings.qml");

/////////////////////////////////////////////////////
