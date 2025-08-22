<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Registro Soporte</title>
        <link href="../PERSONAL/Resources/css/style-regPersonal.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>REGISTRO DE SOPORTE AL CLIENTE:</h1>
        <hr>
        <div class="registroTabla">
            <form action="../SOPORTE/procRegSoporte.php" method="post">
                <label for="id_personal">ID Personal:</label>
                <input type="text" id="txtRPS" name="txtRPS" required>
                <br><br>
                <label for="id_area">ID Área:</label>
                <input type="text" id="txtRAS" name="txtRAS" required>
                <br><br>
                <input type="submit">
                <br><br>
            </form>
        </div>
    </body>
</html>