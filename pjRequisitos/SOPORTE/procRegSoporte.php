<?php

include '../PDO/Conexion.php';
include '../PDO/SoporteDAO.php';

$personal=$_REQUEST['txtRPS'];
$area=$_REQUEST['txtRAS'];

$rSop= new SoporteDAO();

$regSop=$rSop->registrarSoporte($personal, $area);