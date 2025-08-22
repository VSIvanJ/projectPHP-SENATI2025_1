function ConsultarContrato() {
    conCli=$('#txtCCliC').val();
    conSop=$('#txtCSC').val();
    if(conCli==='' || conSop===''){
        $('#selectContrato').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procConsultaContrato.php',
        type:'POST',
        data:{txtCCliC:conCli,txtCSC:conSop},
        success: function (respuesta) {
            $('#selectContrato').html(respuesta);
        },
        error: function () {
            $('#selectContrato').text('ERROR AL INTENTAR CONSULTAR');
        }
    });
}

function ActualizarContrato() {
    AiC=$('#txtAICon').val();
    AclC=$('#txtAICCon').val();
    if(AiC==='' || AclC===''){
        $('#updateContrato').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procActualizaContrato.php',
        type:'POST',
        data:{txtAICon:AiC,txtAICCon:AclC},
        success: function (respuesta2) {
            $('#updateContrato').html(respuesta2);
        },
        error: function () {
            $('#updateContrato').text('ERROR AL INTENTAR ACTUALIZAR');
        }
    });
}

function EliminarContrato() {
    EiCon=$('#txtBICon').val();
    EsCon=$('#txtBSCon').val();
    if(EiCon==='' || EsCon===''){
        $('#deleteContrato').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procEliminaContrato.php',
        type:'POST',
        data:{txtBICon:EiCon,txtBSCon:EsCon},
        success: function (respuesta3) {
            $('#deleteContrato').html(respuesta3);
        },
        error: function () {
            $('#deleteContrato').text('ERROR AL INTENTAR ELIMINAR');
        }
    });
}

function LimpiarConsultaCon() {
    $('#txtCCliC').val('');
    $('#txtCSC').val('');
    $('#selectContrato').text('');
}

function LimpiarActualizarCon() {
    $('#txtAICon').val('');
    $('#txtAICCon').val('');
    $('#updateContrato').text('');
}

function LimpiarEliminarCon() {
    $('#txtBICon').val('');
    $('#txtBSCon').val('');
    $('#deleteContrato').text('');
}