<?php

include '../PDO/Conexion.php';
include '../PDO/AreaDAO.php';

$zona=$_REQUEST['txtZonaA'];
$ht=$_REQUEST['txHorasA'];
$pago=$_REQUEST['txtPagoA'];
$resu=$_REQUEST['txtResumen'];

$rAr=new AreaDAO();

$regArea = $rAr->registrarArea($zona, $ht, $pago, $resu);