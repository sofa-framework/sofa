import QtQuick 2.0
import QtQuick.Controls 1.0
import Qt.labs.settings 1.0
import Scene 1.0
import PickingInteractor 1.0

Scene {
    id: root

    asynchronous: true
    source: ""
    sourceQML: ""
    property string statusMessage: ""

    Component.onCompleted: {
        if(0 !== settings.recent.length)
            source = "file:" + settings.recent.replace(/;.*$/m, "");

        if(0 === source.toString().length)
            source = "file:Demos/caduceus.scn";
    }

    property string recentScenes: ""

    function addRecentScene(sceneSource) {
        recentScenes = sceneSource + ";" + recentScenes.replace(sceneSource + ";", "");
    }

    function clearRecentScenes() {
        recentScenes = "";
    }

    property Settings settings: Settings {
        id: settings
        category: "scene"

        property alias recent: root.recentScenes    // recently opened scenes
    }

    onStatusChanged: {
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
            addRecentScene(path);
            break;
        }
    }

    // convenience
    readonly property bool ready: status === Scene.Ready

    // allow us to interact with the python script controller
    property var pythonInteractor: PythonInteractor {}

    // allow us to interact with the scene physics
    property var pickingInteractor: PickingInteractor {
        stiffness: 100

        onPickingChanged: overrideCursorShape = picking ? Qt.BlankCursor : 0
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
}
