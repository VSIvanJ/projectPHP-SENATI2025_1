<?php

include '../PDO/Conexion.php';
include '../PDO/SoporteDAO.php';

$soporte=$_POST['txtEIS'];
$personal=$_POST['txtEIPS'];

$eSop = new SoporteDAO();

$elimSop = $eSop->eliminarSoporte($soporte, $personal);