<?php
session_start();
if (!isset($_SESSION['id_usuario'])) {
    header("Location: ../index.php");
    exit();
}
$idUsuario = $_SESSION['id_usuario'];
$paqueteID = isset($_GET['paquete_id']) ? $_GET['paquete_id'] : '';
?>
<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Registro Compra</title>
        <link href="Resources/css/style-regCliente.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>Registro de Compra:</h1>
        <hr>
        <div class="registroTabla">
            <form action="procRegCliente.php" target="_blank">
                <label for="id_usuario">ID Usuario:</label>
                <input type="text" id="txtIDUsu" name="txtIDUsu" value="<?php echo htmlspecialchars($idUsuario); ?>" required readonly>
                <br><br>
                <label for="id_paquete">ID Paquete:</label>
                <input type="text" id="txtPaqueteUsu" name="txtPaqueteUsu" value="<?php echo htmlspecialchars($paqueteID); ?>" required readonly>
                <br><br>
                <label for="razon_social">Razón Social:</label>
                <input type="text" id="txtRSUsu" name="txtRSUsu" required>
                <br><br>
                <label for="ruc">Nº RUC:</label>
                <input type="text" id="txtRUCUsu" name="txtRUCUsu" required minlength="11" maxlength="11">
                <br><br>
                <label for="departamento">Departamento:</label>
                <input type="text" id="txtDepUsu" name="txtDepUsu" required>
                <br><br>
                <label for="direccion">Dirección Legal:</label>
                <input type="text" id="txtDirL" name="txtDirL" required>
                <br><br>
                <input type="submit" value="COMPRAR">
            </form>
        </div>
    </body>
</html>