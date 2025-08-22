<?php

include '../PDO/Conexion.php';
include '../PDO/SoporteDAO.php';

$personal=$_POST['txtCIS'];
$area=$_POST['txtCAS'];

$aSop = new SoporteDAO();

//$actSop = $aSop->actualizarSoporte($param);