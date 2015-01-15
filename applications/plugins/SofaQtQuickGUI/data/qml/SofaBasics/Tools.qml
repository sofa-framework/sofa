import Qt.labs.settings 1.0
import "qrc:/SofaCommon/ToolsScript.js" as ToolsScript

Settings {
    id: root
    category: "ui"

    property string uiIds: ";"

    function remove(uiId) {
        ToolsScript.Tools.clearSettingGroup("ui_" + uiId.toString());
    }
}
