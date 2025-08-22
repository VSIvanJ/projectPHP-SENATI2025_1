<?php

class EmpleadoDAO extends Conexion{
    public function registrarEmpleado($soporte,$user,$hash) {
        try {
            //Verificar que existe el ID SOPORTE
            $stmt = $this->conn->prepare("SELECT COUNT(*) FROM SOPORTE WHERE id_soporte = :id_soporte");
            $stmt->bindParam(':id_soporte', $soporte);
            $stmt->execute();
            
            if ($stmt->fetchColumn() == 0) {
                throw new Exception("El ID del personal de soporte ingresado no existe.");
            }
            
            $sql="INSERT INTO EMPLEADO (id_soporte,usuario,clave) VALUES (:id_soporte,:usuario,:clave)";
            $stmt=$this->conn->prepare($sql);
            
            $stmt->bindParam(':id_soporte',$soporte);
            $stmt->bindParam(':usuario',$user);
            $stmt->bindParam(':clave',$hash);
            $stmt->execute();
            
            echo "Se registró correctamente al empleado...";
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }        
    }
    
    
    public function loginEmpleado($user, $clave) {
        try {
            $sql = "SELECT id_empleado,clave FROM EMPLEADO WHERE usuario = :usuario";
            $stmt = $this->conn->prepare($sql);
            $stmt->bindParam(':usuario', $user);
            $stmt->execute();

            // Obtener la contraseña almacenada
            $resultado = $stmt->fetch(PDO::FETCH_ASSOC);

            if ($resultado && password_verify($clave, $resultado['clave'])) {
                echo "¡Inicio de sesión exitoso! Bienvenido, $user";
                return [
                    'id_empleado' => $resultado['id_empleado'],
                    'usuario' => $user
                ];
            } else {
                echo "Usuario y/o contraseña incorrectos. Inténtelo de nuevo.";
                return false;
            }
        } catch (Exception $ex) {
            echo "ERROR: " . $ex->getMessage();
        }
    }
}
