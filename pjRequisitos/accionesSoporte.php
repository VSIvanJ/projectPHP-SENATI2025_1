<?php
session_start();
if (!isset($_SESSION['id_empleado'])) {
    header("Location: index.php");
    exit();
}
?>
<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Acciones Soporte</title>
        <link href="Resources/css/style-acciones.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>LISTA DE ACCIONES PARA EL PERSONAL DE SOPORTE</h1>
        <hr>
        <nav>
            <ul>
                <li><strong>Clientes</strong>
                    <ul>
                        <li><a href="CLIENTE/frmCRUDCliente.php" target="_blank">Consultar Cliente</a></li>
                    </ul>
                </li>
                
                <li><strong>Contratos</strong>
                    <ul>
                        <li><a href="CONTRATO/frmRegContrato.php" target="_blank">Registrar Contrato</a></li>
                        <li><a href="CONTRATO/frmCRUDContrato.php" target="_blank">Consultar Contratos Cerrados</a></li>
                    </ul>
                </li>

                <li><strong>Peticiones Recibidas</strong>
                    <ul>
                        <li><a href="PETICION/frmCRUDPeticion.php" target="_blank">Consultar Peticiones</a></li>
                    </ul>
                </li>
            </ul>
        </nav>
    </body>
</html>
