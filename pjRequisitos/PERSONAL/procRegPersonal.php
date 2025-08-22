<?php

include '../PDO/Conexion.php';
include '../PDO/PersonalDAO.php';

$emp="1";
$nom=$_REQUEST['txtNomb'];
$ap=$_REQUEST['txtApel'];
$dni=$_REQUEST['txtDni'];
$correo=$_REQUEST['txtEmail'];

$rPer= new PersonalDAO();

$regPer=$rPer->registrarPersonal($emp, $nom, $ap, $dni, $correo);