import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Controls.Styles 1.3
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1
import Qt.labs.settings 1.0
import SofaBasics 1.0
import SofaInteractors 1.0
import Qt.labs.folderlistmodel 2.1
import "qrc:/SofaCommon/SofaCommonScript.js" as SofaCommonScript

QtObject {
    id: root

////////////////////////////////////////////////// PRIVATE
    property QtObject d: QtObject {

        property SceneListModel sceneListModel: null

    }

////////////////////////////////////////////////// SCENE

//    readonly property Scene scene: Scene {}

//    // create a sceneListModel only if needed
//    function sceneListModel() {
//        if(null === d.sceneListModel)
//            d.sceneListModel = SofaCommonScript.InstanciateComponent(SceneListModel, root, {"scene": root.scene});

//        return d.sceneListModel;
//    }

////////////////////////////////////////////////// SETTINGS

    //property UISettings uiSettings: UISettings {}
    //property RecentSettings uiSettings: RecentSettings {}

////////////////////////////////////////////////// INTERACTOR

    readonly property string interactorName: {
        if(interactorComponent)
            for(var key in interactorComponentMap)
                if(interactorComponentMap.hasOwnProperty(key) && interactorComponent === interactorComponentMap[key])
                    return key;

        return "";
    }

    property Component interactorComponent: null
    property var interactorComponentMap: null

    property FolderListModel interactorFolderListModel: FolderListModel {
        nameFilters: ["*.qml"]
        showDirs: false
        showFiles: true
        sortField: FolderListModel.Name

        Component.onCompleted: refresh();
        onCountChanged: update();

        function refresh() {
            folder = "";
            folder = "qrc:/SofaInteractors";
        }

        function update() {
            if(root.interactorComponentMap)
                for(var key in root.interactorComponentMap)
                    if(root.interactorComponentMap.hasOwnProperty(key))
                        root.interactorComponentMap[key].destroy();

            var interactorComponentMap = [];
            for(var i = 0; i < count; ++i)
            {
                var fileBaseName = get(i, "fileBaseName");
                var filePath = get(i, "filePath").toString();
                if(-1 !== folder.toString().indexOf("qrc:"))
                    filePath = "qrc" + filePath;

                var name = fileBaseName.slice(fileBaseName.indexOf("_") + 1);
                var interactorComponent = Qt.createComponent(filePath);
                interactorComponentMap[name] = interactorComponent;

                if(null === root.interactorComponent)
                    root.interactorComponent = interactorComponent;
            }

            root.interactorComponentMap = interactorComponentMap;
        }
    }

//////////////////////////////////////////////////


}
