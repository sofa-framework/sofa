import QtQuick 2.0
import QtQuick.Controls 1.0
import Qt.labs.settings 1.0
import Scene 1.0
import PickingInteractor 1.0
import "qrc:/SofaCommon/SofaSettingsScript.js" as SofaSettingsScript
import "qrc:/SofaCommon/SofaToolsScript.js" as SofaToolsScript

Scene {
    id: root

    asynchronous: true
    source: ""
    sourceQML: ""
    property string statusMessage: ""

    onStatusChanged: {
        listModel.selectedId = -1;

        var path = source.toString().replace("///", "/").replace("file:", "");
        switch(status) {
        case Scene.Loading:
            statusMessage = 'Loading "' + path + '" please wait';
            break;
        case Scene.Error:
            statusMessage = 'Scene "' + path + '" issued an error during loading';
            break;
        case Scene.Ready:
            statusMessage = 'Scene "' + path + '" loaded successfully';
            SofaSettingsScript.Recent.add(path);
            break;
        }
    }

    property var listModel: SceneListModel {id : listModel}
    onStepEnd: listModel.update()

    // convenience
    readonly property bool ready: status === Scene.Ready

    // allow us to interact with the python script controller
    property var pythonInteractor: PythonInteractor {}

    // allow us to interact with the scene physics
    property var pickingInteractor: PickingInteractor {
        stiffness: 100

        onPickingChanged: SofaToolsScript.Tools.overrideCursorShape = picking ? Qt.BlankCursor : 0
    }

    function keyPressed(event) {
        if(event.modifiers & Qt.ShiftModifier)
            onKeyPressed(event.key);
    }

    function keyReleased(event) {
        //if(event.modifiers & Qt.ShiftModifier)
            onKeyReleased(event.key);
    }

	property var resetAction: Action {
        text: "&Reset"
        shortcut: "Ctrl+Alt+R"
        onTriggered: root.reset();
        tooltip: "Reset the simulation"
    }

    function dataValue(dataName) {
        if(arguments.length == 1) {
            return onDataValue(dataName);
        }

        console.debug("ERROR: Scene - using dataValue with an invalid number of arguments:", arguments.length);
    }

    function setDataValue(dataName) {
        if(arguments.length > 1){
            var packedArguments = [];
            for(var i = 1; i < arguments.length; i++)
                packedArguments.push(arguments[i]);

            return onSetDataValue(dataName, packedArguments);
        }

        console.debug("ERROR: Scene - using setDataValue with an invalid number of arguments:", arguments.length);
    }
}
