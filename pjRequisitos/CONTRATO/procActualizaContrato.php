<?php

include '../PDO/Conexion.php';
include '../PDO/ContratoDAO.php';

$contrato=$_POST['txtAICon'];
$cliente=$_POST['txtAICCon'];

$aCon = new ContratoDAO();

$actCon = $aCon->actualizarContrato($contrato, $cliente);