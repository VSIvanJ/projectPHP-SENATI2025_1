<?php

include '../PDO/Conexion.php';
include '../PDO/AreaDAO.php';

$idA=$_POST['txtAI'];
$pago=$_POST['txtAP'];
$resumen=$_POST['txtAD'];

$aAr = new AreaDAO();

$actuArea = $aAr->actualizarArea($idA, $pago, $resumen);