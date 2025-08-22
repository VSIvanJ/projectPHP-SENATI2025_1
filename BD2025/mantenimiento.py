import sqlite3
import bcrypt

conexion=sqlite3.connect('./requisitos.db')
cursor=conexion.cursor()

# Crear las tablas para la BD
def crear_tablas():
    try:
        cursor.execute('''CREATE TABLE IF NOT EXISTS ADMINISTRADOR(
            id_admin INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            correo TEXT NOT NULL,
            clave TEXT NOT NULL
        )''')
        cursor.execute('''
    CREATE TABLE IF NOT EXISTS EMPRESA (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        razon_social TEXT NOT NULL,
        ruc TEXT NOT NULL,
        ubicacion TEXT NOT NULL
        );
        
    CREATE TABLE IF NOT EXISTS AREA (
        id_area INTEGER PRIMARY KEY AUTOINCREMENT,
        zona TEXT NOT NULL,
        horas_trabajo INTEGER NOT NULL,
        pago_hora DECIMAL(10,2) NOT NULL,
        resumen TEXT NOT NULL
        );
        
    CREATE TABLE IF NOT EXISTS PAQUETE (
        id_paquete INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        descripcion TEXT NOT NULL,
        precio DECIMAL(10,2) NOT NULL
        );
        
    CREATE TABLE IF NOT EXISTS USUARIO (
        id_usuario INTEGER PRIMARY KEY AUTOINCREMENT,
        usuario TEXT NOT NULL,
        email TEXT NOT NULL,
        clave TEXT NOT NULL
        );
        
    CREATE TABLE IF NOT EXISTS PERSONAL (
        id_personal INTEGER PRIMARY KEY AUTOINCREMENT,
        id_empresa INTEGER NOT NULL,
        nombre TEXT NOT NULL,
        apellido TEXT NOT NULL,
        dni TEXT NOT NULL,
        correo TEXT NOT NULL,
        CONSTRAINT fk_empresa FOREIGN KEY (id_empresa) REFERENCES EMPRESA(id) ON DELETE CASCADE
        );
        
    CREATE TABLE IF NOT EXISTS CLIENTE (
        id_cliente INTEGER PRIMARY KEY AUTOINCREMENT,
        id_usuario INTEGER NOT NULL,
        id_paquete INTEGER NOT NULL,
        razon_social TEXT NOT NULL,
        ruc TEXT NOT NULL,
        departamento TEXT NOT NULL,
        direccion TEXT NOT NULL,
        CONSTRAINT fk_usuario FOREIGN KEY (id_usuario) REFERENCES USUARIO(id_usuario) ON DELETE CASCADE,
        CONSTRAINT fk_paquete FOREIGN KEY (id_paquete) REFERENCES PAQUETE(id_paquete) ON DELETE CASCADE
        );
        
    CREATE TABLE IF NOT EXISTS SOPORTE (
        id_soporte INTEGER PRIMARY KEY AUTOINCREMENT,
        id_personal INTEGER NOT NULL,
        id_area INTEGER NOT NULL,
        sueldo DECIMAL(10,2) NOT NULL,
        CONSTRAINT fk_personal FOREIGN KEY (id_personal) REFERENCES PERSONAL(id_personal) ON DELETE CASCADE,
        CONSTRAINT fk_area FOREIGN KEY (id_area) REFERENCES AREA(id_area) ON DELETE CASCADE
        );
        
    CREATE TABLE IF NOT EXISTS CONTRATO (
        id_contrato INTEGER PRIMARY KEY AUTOINCREMENT,
        id_cliente INTEGER NOT NULL,
        id_soporte INTEGER NOT NULL,
        contenido TEXT NOT NULL,
        fecha_inicio DATE NOT NULL,
        fecha_fin DATE NOT NULL,
        monto DECIMAL(10,2) NOT NULL,
        CONSTRAINT fk_cliente FOREIGN KEY (id_cliente) REFERENCES CLIENTE(id_cliente) ON DELETE CASCADE,
        CONSTRAINT fk_soporte FOREIGN KEY (id_soporte) REFERENCES SOPORTE(id_soporte) ON DELETE CASCADE
        );
        
    CREATE TABLE IF NOT EXISTS PETICION (
        id_peticion INTEGER PRIMARY KEY AUTOINCREMENT,
        id_cliente INTEGER NOT NULL,
        id_contrato INTEGER NOT NULL,
        detalle TEXT NOT NULL,
        fecha_peticion DATE NOT NULL,
        estado TEXT NOT NULL,
        CONSTRAINT fk_cliente_peticion FOREIGN KEY (id_cliente) REFERENCES CLIENTE(id_cliente) ON DELETE CASCADE,
        CONSTRAINT fk_contrato FOREIGN KEY (id_contrato) REFERENCES CONTRATO(id_contrato) ON DELETE CASCADE
        );
        
        CREATE TABLE IF NOT EXISTS EMPLEADO(
            id_empleado INTEGER PRIMARY KEY AUTOINCREMENT,
            id_soporte INTEGER NOT NULL,
            usuario TEXT NOT NULL,
            clave TEXT NOT NULL,
            CONSTRAINT fk_soporte_empleado FOREIGN KEY (id_soporte) REFERENCES SOPORTE(id_soporte)
        )
        
        ''')
        print('Tabla creada correctamente')
    except Exception as e:
        print('No se pudo crear la tabla:',e)

def cambiar_tabla():
    try:
        cursor.execute('''ALTER TABLE ADMINISTRADOR ADD COLUMN dni TEXT NOT NULL''')
        print('Tabla modificada correctamente')
    except Exception as e:
        print('No se pudo modificar la tabla:',e)

def hashing_sal_contraseña(clave):
    sal=bcrypt.gensalt()
    h_contra=bcrypt.hashpw(clave.encode('utf-8'),sal)
    return h_contra

def verificarContraseña(clave_request,hash_clave):
    return bcrypt.checkpw(clave_request.encode('utf-8'),hash_clave)

def insertar_administrador(nombre, correo, clave,dni):
    try:
        h=hashing_sal_contraseña(clave)
        u=(nombre, correo, h,dni)
        cursor.execute('INSERT INTO ADMINISTRADOR (nombre, correo, clave,dni) VALUES (?, ?, ?, ?)', u)
        print('Administrador registrado correctamente...')
    except Exception as e:
        print(f'ERROR AL INTENTAR REGISTRAR: {e}')
        
    finally:
        cursor.close()

#insertar_administrador('admin','correodeejemplo@gmail.com','HolaMyGentez*102*','12345678')
#insertar_administrador('admin3','correodeejemplo@gmail.com','S0y4dm1n','12345678')
conexion.commit()
conexion.close()