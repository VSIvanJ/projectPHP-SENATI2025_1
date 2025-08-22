<?php

class SoporteDAO extends Conexion{
    
    public function registrarSoporte($personal,$area) {
        try {
            
            // Verificar que el personal existe
            $stmt = $this->conn->prepare("SELECT COUNT(*) FROM PERSONAL WHERE id_personal = :id_personal");
            $stmt->bindParam(':id_personal', $personal);
            $stmt->execute();
            
            if ($stmt->fetchColumn() == 0) {
                throw new Exception("El ID del personal ingresado no existe.");
            }
            
            // Verificar que la área existe
            $stmt = $this->conn->prepare("SELECT COUNT(*) FROM AREA WHERE id_area = :id_area");
            $stmt->bindParam(':id_area', $area);
            $stmt->execute();
            
            if ($stmt->fetchColumn() == 0) {
                throw new Exception("El ID de área ingresada no existe.");
            }
            
            //Obtener sueldo del personal: 30 x pago/hora x horas/trabajo           
            $sqlsueldo = $this->conn->prepare("SELECT horas_trabajo, pago_hora FROM AREA WHERE id_area = :id_area");
            $sqlsueldo->bindParam(':id_area', $area);
            $sqlsueldo->execute();
            $datos = $sqlsueldo->fetch(PDO::FETCH_ASSOC);
            
            if($datos){
                $horasTrabajo = $datos['horas_trabajo'];
                $pagoHora = $datos['pago_hora'];
                $sueldo = 30 * $pagoHora * $horasTrabajo;
            }else{
                throw new Exception("Error al intentar obtener los datos para el sueldo");
            }
            
            $sql="INSERT INTO SOPORTE (id_personal,id_area,sueldo) VALUES (:id_personal,:id_area,:sueldo)";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':id_personal',$personal);
            $unir->bindParam(':id_area',$area);
            $unir->bindParam(':sueldo',$sueldo);
            $unir->execute();
            
            echo "Se registró al personal de soporte";
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function consultarSoporte($personal,$area) {
        try {
            $sql="SELECT s.*,a.zona "
                    . "FROM SOPORTE s "
                    . "INNER JOIN AREA a ON s.id_area = a.id_area "
                    . "WHERE id_personal = :id_personal AND s.id_area = :id_area";
            $unir=$this->conn->prepare($sql);
            $unir->bindParam(':id_personal',$personal);
            $unir->bindParam(':id_area',$area);
            $unir->execute();
            $rpta=$unir->fetch(PDO::FETCH_ASSOC);
            
            if($rpta){
                return $rpta;
            }else{
                echo "El personal de soporte no existe";
            }
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function actualizarSoporte($param) {
        try {
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function eliminarSoporte($soporte,$personal) {
        try {
            $sql="DELETE FROM SOPORTE WHERE id_soporte = :id_soporte AND id_personal = :id_personal";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':id_soporte',$soporte);
            $unir->bindParam(':id_personal',$personal);
            $unir->execute();
            
            echo "Se eliminó al personal de soporte";
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
}
