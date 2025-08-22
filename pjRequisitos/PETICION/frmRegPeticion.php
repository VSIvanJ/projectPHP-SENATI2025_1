<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Enviar Petición</title>
        <link href="../AREA/Resources/css/style-regArea.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>Enviar Petición:</h1>
        <div class="registroTabla">
            <form action="procRegPeticion.php">
                <label for="id_cliente">ID Cliente:</label>
                <input type="text" id="txtCliP" name="txtCliP" required>
                <br><br>
                <label for="id_contrato">ID Contrato:</label>
                <input type="text" id="txtConP" name="txtConP" required>
                <br><br>
                <label for="detalle">Detalle de Petición:</label>
                <textarea id="txtDetP" name="txtDetP" rows="5" cols="10" required></textarea>
                <br><br>
                <input type="submit">
            </form>
        </div>
    </body>
</html>