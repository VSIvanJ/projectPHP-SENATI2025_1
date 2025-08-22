<?php

class AreaDAO extends Conexion{
    public function registrarArea($zona,$ht,$pago,$resu) {
        try {
            $sql="INSERT INTO AREA (zona,horas_trabajo,pago_hora,resumen) "
                    . "VALUES (:zona,:horas_trabajo,:pago_hora,:resumen)";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':zona',$zona);
            $unir->bindParam(':horas_trabajo',$ht);
            $unir->bindParam(':pago_hora',$pago);
            $unir->bindParam(':resumen',$resu);
            $unir->execute();
            
            echo "Se registró el área";
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function consultarArea($zona) {
        try {
            $sql="SELECT * FROM AREA WHERE zona=:zona";
            $unir=$this->conn->prepare($sql);
            $unir->bindParam(':zona',$zona);
            $unir->execute();
            $rpta=$unir->fetch(PDO::FETCH_ASSOC);
            
            if($rpta){
                return $rpta;
            }else{
                echo "El área ingresada no existe";
            }
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function actualizarArea($idA,$pago,$resumen) {
        try {
            $sql="UPDATE AREA SET pago_hora = :pago_hora, resumen = :resumen WHERE id_area = :id_area";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':id_area',$idA);
            $unir->bindParam(':pago_hora',$pago);
            $unir->bindParam(':resumen',$resumen);
            $unir->execute();
            
            echo "Se actualizaron los datos";
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function eliminarArea($area) {
        try {
            $sql="DELETE FROM AREA WHERE id_area = :id_area";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':id_area',$area);
            $unir->execute();
            
            echo "Se eliminó el área";
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
}