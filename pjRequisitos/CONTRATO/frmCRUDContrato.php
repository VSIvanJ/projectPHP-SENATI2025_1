<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Acciones Contrato</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js" crossorigin="anonymous"></script>
        <script src="Resources/js/scriptContrato.js" type="text/javascript"></script>
        <link href="../Resources/css/baseCRUD.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>ACCIONES DE CONTRATO</h1>
        <div class="crudTabla">
            <h2>Consultar Contrato:</h2>
            <label for="id_cliente">ID Cliente:</label>
            <input type="text" id="txtCCliC" name="txtCCliC" required>
            <br><br>
            <label for="id_soporte">ID Personal de Soporte:</label>
            <input type="text" id="txtCSC" name="txtCSC" required>
            <br><br>
            <button onclick="ConsultarContrato()">CONSULTAR</button>
            <button onclick="LimpiarConsultaCon()">LIMPIAR</button>
            <br><br>
            <div id="selectContrato"></div>
        </div>
        <br><br>
        <div class="crudTabla">
            <h2>Actualizar Estado del Contrato:</h2>
            <label for="id_contrato">ID Contrato:</label>
            <input type="text" id="txtAICon" name="txtAICon" required>
            <br><br>
            <label for="id_cliente">ID Cliente:</label>
            <input type="text" id="txtAICCon" name="txtAICCon" required>
            <br><br>
            <button onclick="ActualizarContrato()">ACTUALIZAR</button>
            <button onclick="LimpiarActualizarCon()">LIMPIAR</button>
            <br><br>
            <div id="updateContrato"></div>
        </div>
        <br><br>
        <div class="crudTabla">
            <h2>Eliminar Contrato:</h2>
            <label for="id_contrato">ID Contrato</label>
            <input type="text" id="txtBICon" name="txtBICon" required>
            <br><br>
            <label for="id_soporte">ID Soporte</label>
            <input type="text" id="txtBSCon" name="txtBSCon" required>
            <br><br>
            <button onclick="EliminarContrato()">ELIMINAR</button>
            <button onclick="LimpiarEliminarCon()">LIMPIAR</button>
            <br><br>
            <div id="deleteContrato"></div>
        </div>
    </body>
</html>