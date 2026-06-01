import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = "data/database/fotorradar.db"

DATOS_PRUEBA = [
    ("PBC1234", "Chevrolet", "Aveo", "Blanco", "Juan Pérez", "juan.perez@example.com", "Activo"),
    ("AAA1234", "Toyota", "Yaris", "Gris", "María López", "maria.lopez@example.com", "Activo"),
    ("PBH6824", "Kia", "Rio", "Rojo", "Carlos Sánchez", "carlos.sanchez@example.com", "Activo"),
    ("AAC0123", "Hyundai", "Tucson", "Negro", "Ana Torres", "ana.torres@example.com", "Activo"),
]


def conectar(ruta_bd: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    Path(ruta_bd).parent.mkdir(parents=True, exist_ok=True)
    conexion = sqlite3.connect(ruta_bd)
    conexion.row_factory = sqlite3.Row
    return conexion


def inicializar_bd(ruta_bd: str = DEFAULT_DB_PATH) -> None:
    with conectar(ruta_bd) as conexion:
        conexion.execute(
            """
            CREATE TABLE IF NOT EXISTS vehiculos (
                placa TEXT PRIMARY KEY,
                marca TEXT,
                modelo TEXT,
                color TEXT,
                propietario TEXT,
                correo TEXT,
                estado TEXT
            )
            """
        )
        _asegurar_columnas_vehiculos(conexion)

        conexion.execute(
            """
            CREATE TABLE IF NOT EXISTS eventos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha_hora TEXT,
                placa TEXT,
                marca TEXT,
                modelo TEXT,
                color TEXT,
                propietario TEXT,
                correo TEXT,
                velocidad_kmh REAL,
                limite_kmh REAL,
                estado_difuso TEXT,
                nivel_infraccion TEXT,
                sancion TEXT,
                horas_suspension INTEGER,
                mensaje TEXT,
                evidencia_frame TEXT,
                evidencia_placa TEXT,
                fuente TEXT
            )
            """
        )
        _asegurar_columnas_eventos(conexion)

        conexion.executemany(
            """
            INSERT OR IGNORE INTO vehiculos
                (placa, marca, modelo, color, propietario, correo, estado)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            DATOS_PRUEBA,
        )


def _asegurar_columnas_vehiculos(conexion: sqlite3.Connection) -> None:
    columnas = {fila["name"] for fila in conexion.execute("PRAGMA table_info(vehiculos)").fetchall()}
    requeridas = {
        "modelo": "TEXT",
        "estado": "TEXT",
    }
    for columna, tipo in requeridas.items():
        if columna not in columnas:
            conexion.execute(f"ALTER TABLE vehiculos ADD COLUMN {columna} {tipo}")


def _asegurar_columnas_eventos(conexion: sqlite3.Connection) -> None:
    columnas = {fila["name"] for fila in conexion.execute("PRAGMA table_info(eventos)").fetchall()}
    requeridas = {
        "marca": "TEXT",
        "modelo": "TEXT",
        "color": "TEXT",
        "propietario": "TEXT",
        "correo": "TEXT",
        "velocidad_kmh": "REAL",
        "limite_kmh": "REAL",
        "estado_difuso": "TEXT",
        "nivel_infraccion": "TEXT",
        "horas_suspension": "INTEGER",
        "mensaje": "TEXT",
        "evidencia_frame": "TEXT",
        "evidencia_placa": "TEXT",
        "fuente": "TEXT",
    }
    for columna, tipo in requeridas.items():
        if columna not in columnas:
            conexion.execute(f"ALTER TABLE eventos ADD COLUMN {columna} {tipo}")


def normalizar_placa(placa: str) -> str:
    return (placa or "").strip().replace(" ", "").upper()


def buscar_vehiculo_por_placa(placa: str, ruta_bd: str = DEFAULT_DB_PATH) -> dict | None:
    placa_normalizada = normalizar_placa(placa)
    inicializar_bd(ruta_bd)
    with conectar(ruta_bd) as conexion:
        fila = conexion.execute("SELECT * FROM vehiculos WHERE placa = ?", (placa_normalizada,)).fetchone()
    return dict(fila) if fila else None


def listar_vehiculos(ruta_bd: str = DEFAULT_DB_PATH) -> list[dict]:
    inicializar_bd(ruta_bd)
    with conectar(ruta_bd) as conexion:
        filas = conexion.execute("SELECT * FROM vehiculos ORDER BY placa").fetchall()
    return [dict(fila) for fila in filas]


def listar_eventos(ruta_bd: str = DEFAULT_DB_PATH, limit: int = 50, limite: int | None = None) -> list[dict]:
    inicializar_bd(ruta_bd)
    cantidad = int(limite if limite is not None else limit)
    with conectar(ruta_bd) as conexion:
        filas = conexion.execute(
            "SELECT * FROM eventos ORDER BY id DESC LIMIT ?",
            (cantidad,),
        ).fetchall()
    return [dict(fila) for fila in filas]


def guardar_evento(evento: dict, ruta_bd: str = DEFAULT_DB_PATH) -> int:
    inicializar_bd(ruta_bd)
    placa = normalizar_placa(evento.get("placa"))
    vehiculo = evento.get("vehiculo") or {}

    with conectar(ruta_bd) as conexion:
        cursor = conexion.execute(
            """
            INSERT INTO eventos (
                fecha_hora,
                placa,
                marca,
                modelo,
                color,
                propietario,
                correo,
                velocidad_kmh,
                limite_kmh,
                estado_difuso,
                nivel_infraccion,
                sancion,
                horas_suspension,
                mensaje,
                evidencia_frame,
                evidencia_placa,
                fuente
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evento.get("fecha_hora"),
                placa,
                evento.get("marca") or vehiculo.get("marca"),
                evento.get("modelo") or vehiculo.get("modelo"),
                evento.get("color") or vehiculo.get("color"),
                evento.get("propietario") or vehiculo.get("propietario"),
                evento.get("correo") or vehiculo.get("correo"),
                evento.get("velocidad_kmh", evento.get("velocidad")),
                evento.get("limite_kmh"),
                evento.get("estado_difuso", evento.get("estado")),
                evento.get("nivel_infraccion"),
                evento.get("sancion"),
                evento.get("horas_suspension", 0),
                evento.get("mensaje"),
                evento.get("evidencia_frame", evento.get("evidencia")),
                evento.get("evidencia_placa"),
                evento.get("fuente"),
            ),
        )
        return int(cursor.lastrowid)
