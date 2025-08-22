<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Registro Paquete</title>
        <link href="Resources/css/style-reg.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>Registro de Paquete:</h1>
        <hr>
        <div class="registroTabla">
            <form action="procRegPaquete.php" target="_blank">
                <label for="nombre">Nombre del paquete:</label>
                <input type="text" id="txtNombreP" name="txtNombreP" required>
                <br><br>
                <label for="descripcion">Descripción:</label>
                <input type="text" id="txtDescripcionP" name="txtDescripcionP" required>
                <br><br>
                <label for="precio">Precio S/.</label>
                <input type="text" id="txtPrecioP" name="txtPrecioP">
                <br><br>
                <input type="submit">
            </form>
        </div>
    </body>
</html>