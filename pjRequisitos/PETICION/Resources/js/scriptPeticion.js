function ConsultarPeticion() {
    conCliP=$('#txtCClP').val();
    conConP=$('#txtCCP').val();
    if(conCliP==='' || conConP===''){
        $('#selectPeticion').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procConsultaPeticion.php',
        type:'POST',
        data:{txtCClP:conCliP,txtCCP:conConP},
        success: function (respuesta) {
            $('#selectPeticion').html(respuesta);
        },
        error: function () {
            $('#selectPeticion').text('ERROR AL INTENTAR CONSULTAR');
        }
    });
}

function ActualizarPeticion() {
    AidPet=$('#txtAIPet').val();
    AconP=$('#txtACPet').val();
    if(AidPet==='' || AconP===''){
        $('#updatePeticion').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procActualizaPeticion.php',
        type:'POST',
        data:{txtAIPet:AidPet,txtACPet:AconP},
        success: function (respuesta2) {
            $('#updatePeticion').html(respuesta2);
        },
        error: function () {
            $('#updatePeticion').text('ERROR AL INTENTAR ACTUALIZAR');
        }
    });
}

function EliminarPeticion() {
    EIdP=$('#txtBIPet').val();
    if(EIdP===''){
        $('#deletePeticion').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procEliminaPeticion.php',
        type:'POST',
        data:{txtBIPet:EIdP},
        success: function (respuesta3) {
            $('#deletePeticion').html(respuesta3);
        },
        error: function () {
            $('#deletePeticion').text('ERROR AL INTENTAR ELIMINAR');
        }
    });
}

function LimpiarConsultaPet() {
    $('#txtCClP').val('');
    $('#txtCCP').val('');
    $('#selectPeticion').text('');
}

function LimpiarActualizarPet() {
    $('#txtAIPet').val('');
    $('#txtACPet').val('');
    $('#updatePeticion').text('');
}

function LimpiarEliminarPet() {
    $('#txtBIPet').val('');
    $('#deletePeticion').text('');
}