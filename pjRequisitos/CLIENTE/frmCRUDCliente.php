<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Acciones Cliente</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js" crossorigin="anonymous"></script>
        <script src="Resources/js/scriptCliente.js" type="text/javascript"></script>
        <link href="../Resources/css/baseCRUD.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>ACCIONES DE CLIENTE</h1>
        <div class="crudTabla">
            <h2>Consultar Cliente:</h2>
            <label for="ruc">Ingrese NºRUC:</label>
            <input type="text" id="txtCRC" name="txtCRC" minlength="11" maxlength="11" required>
            <br><br>
            <button onclick="ConsultarCliente()">CONSULTAR</button>
            <button onclick="LimpiarConsultaCli()">LIMPIAR</button>
            <br><br>
            <div id="selectCliente"></div>
        </div>
        <br><br>
        <div class="crudTabla">
            <h2>Actualizar Dirección Legal:</h2>
            <label for="ruc">Nº RUC:</label>
            <input type="text" id="txtARC" name="txtARC" minlength="11" maxlength="11" required>
            <br><br>
            <label for="direccion">Nueva Dirección:</label><br>
            <input type="text" id="txtADC" name="txtADC" required>
            <br><br>
            <button onclick="ActualizarCliente()">ACTUALIZAR</button>
            <button onclick="LimpiarActualizarCli()">LIMPIAR</button>
            <br><br>
            <div id="updateCliente"></div>
        </div>
        <br><br>
        <div class="crudTabla">
            <h2>Eliminar Cliente:</h2>
            <label for="id_cliente">ID Cliente:</label>
            <input type="text" id="txtEIC" name="txtEIC" required>
            <br><br>
            <label for="ruc">Nº RUC:</label>
            <input type="text" id="txtERC" name="txtERC" required minlength="11" maxlength="11">
            <br><br>
            <button onclick="EliminarCliente()">ELIMINAR</button>
            <button onclick="LimpiarEliminarCli()">LIMPIAR</button>
            <br><br>
            <div id="deleteCliente"></div>
        </div>
    </body>
</html>