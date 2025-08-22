<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Acciones Petición</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js" crossorigin="anonymous"></script>
        <script src="Resources/js/scriptPeticion.js" type="text/javascript"></script>
        <link href="../Resources/css/baseCRUD.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>ACCIONES DE PETICIÓN:</h1>
        <div class="crudTabla">
            <h2>Consultar Petición:</h2>
            <label for="id_cliente">ID Cliente:</label>
            <input type="text" id="txtCClP" name="txtCClP" required>
            <br><br>
            <label for="id_contrato">ID Contrato:</label>
            <input type="text" id="txtCCP" name="txtCCP" required>
            <br><br>
            <button onclick="ConsultarPeticion()">CONSULTAR</button>
            <button onclick="LimpiarConsultaPet()">LIMPIAR</button>
            <br><br>
            <div id="selectPeticion"></div>
        </div>
        <br><br>
        <div class="crudTabla">
            <h2>Actualizar Estado de Petición:</h2>
            <label for="id_peticion">ID Petición:</label>
            <input type="text" id="txtAIPet" name="txtAIPet" required>
            <br><br>
            <label for="id_contrato">ID Contrato:</label>
            <input type="text" id="txtACPet" name="txtACPet" required>
            <br><br>
            <button onclick="ActualizarPeticion()">ACTUALIZAR</button>
            <button onclick="LimpiarActualizarPet()">LIMPIAR</button>
            <br><br>
            <div id="updatePeticion"></div>
        </div>
        <br><br>
        <div class="crudTabla">
            <h2>Eliminar Petición:</h2>
            <label for="id_peticion">ID Petición:</label>
            <input type="text" id="txtBIPet" name="txtBIPet" required>
            <br><br>
            <button onclick="EliminarPeticion()">ELIMINAR</button>
            <button onclick="LimpiarEliminarPet()">LIMPIAR</button>
            <br><br>
            <div id="deletePeticion"></div>
        </div>
    </body>
</html>