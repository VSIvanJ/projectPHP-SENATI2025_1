<?php

include '../PDO/Conexion.php';
include '../PDO/ContratoDAO.php';

$contrato=$_POST['txtBICon'];
$soporte=$_POST['txtBSCon'];

$eCon = new ContratoDAO();

$elimCon = $eCon->eliminarContrato($contrato, $soporte);