<?php

include '../PDO/Conexion.php';
include '../PDO/PaqueteDAO.php';

$des=$_POST['txtActDes'];
$pre=$_POST['txtActPre'];
$id=$_POST['txtActId'];


$actP = new PaqueteDAO();

$actuPaq = $actP->actualizarPaquete($des, $pre, $id);