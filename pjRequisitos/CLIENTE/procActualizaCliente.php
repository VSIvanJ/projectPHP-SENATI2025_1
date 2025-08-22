<?php

include '../PDO/Conexion.php';
include '../PDO/ClienteDAO.php';

$ruc=$_POST['txtARC'];
$direccion=$_POST['txtADC'];

$aCli = new ClienteDAO();

$actCli = $aCli->actualizarCliente($ruc, $direccion);