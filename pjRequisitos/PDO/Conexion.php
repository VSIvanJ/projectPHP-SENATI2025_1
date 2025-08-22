<?php

class Conexion {
    private $dbname;
    protected $conn;
    
    public function __construct()
    {
       $this->dbname="D:\\BD2025\\requisitos.db";
       $this->conn= new PDO("sqlite:" . $this->dbname);
       $this->conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    }
}