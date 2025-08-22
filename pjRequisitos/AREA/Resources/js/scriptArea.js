function ConsultarArea() {
    conA=$('#txtCA').val();
    if(conA===''){
        $('#selectArea').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procConsultaArea.php',
        type:'POST',
        data:{txtCA:conA},
        success: function (respuesta) {
            $('#selectArea').html(respuesta);
        },
        error: function () {
            $('#selectArea').text('ERROR AL INTENTAR CONSULTAR');
        }
    });
}

function ActualizarArea() {
    AidA=$('#txtAI').val();
    AprA=$('#txtAP').val();
    AdeA=$('#txtAD').val();
    if(AidA==='' || AprA==='' || AdeA===''){
        $('#updateArea').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procActualizaArea.php',
        type:'POST',
        data:{txtAI:AidA,txtAP:AprA,txtAD:AdeA},
        success: function (respuesta2) {
            $('#updateArea').html(respuesta2);
        },
        error: function () {
            $('#updateArea').text('ERROR AL INTENTAR ACTUALIZAR');
        }
    });
}

function EliminarArea() {
    EIdA=$('#txtEIA').val();
    if(EIdA===''){
        $('#deleteArea').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procEliminaArea.php',
        type:'POST',
        data:{txtEIA:EIdA},
        success: function (respuesta3) {
            $('#deleteArea').html(respuesta3);
        },
        error: function () {
            $('#deleteArea').text('ERROR AL INTENTAR ELIMINAR');
        }
    });
}

function LimpiarConsultaA() {
    $('#txtCA').val('');
    $('#selectArea').text('');
}

function LimpiarActualizarA() {
    $('#txtAI').val('');
    $('#txtAP').val('');
    $('#txtAD').val('');
    $('#updateArea').text('');
}

function LimpiarEliminarA() {
    $('#txtEIA').val('');
    $('#deleteArea').text('');
}