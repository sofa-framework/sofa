import Qt.labs.settings 1.0
import "qrc:/SofaCommon/SofaToolsScript.js" as SofaToolsScript

Settings {
    category: "ui"

    property string uiIds: ";"

    function generate() {
        var uiId = 1;
        var uiIdList = uiIds.split(";");

        var previousId = 0;
        for(var i = 0; i < uiIdList.length; ++i) {
            if(0 === uiIdList[i].length)
                continue;

            uiId = Number(uiIdList[i]);

            if(previousId + 1 !== uiId) {
                uiId = previousId + 1;
                break;
            }

            previousId = uiId;
            ++uiId;
        }

        return uiId;
    }

    function add(uiId) {
        if(0 === uiId)
            return;

        if(-1 === uiIds.search(";" + uiId.toString() + ";")) {
            uiIds += uiId.toString() + ";";
            update();
        }
    }

    function remove(uiId) {
        if(0 === uiId)
            return;

        uiIds = uiIds.replace(";" + uiId.toString() + ";", ";");

        SofaToolsScript.Tools.clearSettingGroup("ui_" + uiId.toString());
    }

    function replace(previousUiId, uiId) {
        if(0 === uiId)
            return;

        uiIds = uiIds.replace(";" + uiId.toString() + ";", ";");

        if(-1 === uiIds.search(";" + uiId.toString() + ";")) {
            uiIds += uiId.toString() + ";";
            update();
        }
    }

    function update() {
        var uiIdList = uiIds.split(";");
        uiIdList.sort(function(a, b) {return Number(a) - Number(b);});

        uiIds = ";";
        for(var i = 0; i < uiIdList.length; ++i)
            if(0 !== uiIdList[i].length)
                uiIds += uiIdList[i] + ";";
    }
}
