<?php

include '../PDO/Conexion.php';
include '../PDO/ContratoDAO.php';


$cliente=$_REQUEST['txtCliCon'];
$soporte=$_REQUEST['txtSopCon'];
$contenido=$_REQUEST['txtConCon'];
$inicio=$_REQUEST['fIni'];
$fin=$_REQUEST['fFin'];
$monto=$_REQUEST['txtMonCon'];

$rCon = new ContratoDAO();

$regCon = $rCon->registrarContrato($cliente, $soporte, $contenido, $inicio, $fin, $monto);
